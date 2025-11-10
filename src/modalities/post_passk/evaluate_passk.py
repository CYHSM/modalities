import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dataset_loader import load_gsm8k
from utils import extract_answer, check_answer
from typing import List, Dict
import os


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_fewshot_examples() -> List[Dict[str, str]]:
    return [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs every day. She makes 9 * $2 = $18 every day at the farmers' market. \\boxed{18}"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "The robe takes 2 / 2 = 1 bolt of white fiber. So the total amount of fabric is 2 + 1 = 3 bolts. \\boxed{3}"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "The cost of the house and repairs is 80,000 + 50,000 = $130,000. He increased the value of the house by 80,000 * 1.5 = $120,000. So the new value is 80,000 + 120,000 = $200,000. So he made a profit of 200,000 - 130,000 = $70,000. \\boxed{70000}"
        },
        {
            "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
            "answer": "He runs 3 * 3 = 9 sprints a week. So he runs 9 * 60 = 540 meters. \\boxed{540}"
        }
    ]


def create_prompt(question: str, n_fewshots: int = 0) -> str:
    prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n\n"
    
    if n_fewshots > 0:
        examples = get_fewshot_examples()[:n_fewshots]
        for example in examples:
            prompt += f"User: {example['question']}\n"
            prompt += f"Assistant: {example['answer']}\n\n"
    
    prompt += f"User: {question}\n"
    prompt += "Please reason step by step, and put your final answer within \\boxed{{}}. \n\n"
    prompt += "Assistant:"
    return prompt


def generate_samples(
    model,
    tokenizer,
    question: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_fewshots: int,
    verbose: bool
) -> List[str]:
    prompt = create_prompt(question, n_fewshots)
    
    if verbose:
        print(f"\n{'='*80}")
        print("PROMPT:")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs_list = []
    batch_size = min(n_samples, 8)
    
    for i in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - i)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=current_batch,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for output in outputs:
            text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            outputs_list.append(text)
    
    return outputs_list


def evaluate_problem(
    model,
    tokenizer,
    problem: Dict,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_fewshots: int,
    verbose: bool
) -> Dict:
    question = problem['question']
    ground_truth = problem['answer']
    
    if verbose:
        print(f"\n{'#'*80}")
        print(f"Problem {problem['idx']}: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'#'*80}")
    
    responses = generate_samples(
        model, tokenizer, question, n_samples, 
        temperature, top_p, max_tokens, n_fewshots, verbose
    )
    
    correct_count = 0
    for idx, response in enumerate(responses):
        pred_answer = extract_answer(response)
        is_correct = pred_answer and check_answer(pred_answer, ground_truth)
        
        if is_correct:
            correct_count += 1
        
        if verbose and idx < 3:
            print(f"\nSample {idx + 1}:")
            print(f"{'-'*80}")
            print(response)
            print(f"{'-'*80}")
            print(f"Extracted Answer: {pred_answer}")
            print(f"Correct: {is_correct}")
    
    if verbose:
        print(f"\nCorrect: {correct_count}/{n_samples}")
    
    return {
        'idx': problem['idx'],
        'question': question,
        'answer': ground_truth,
        'correct_count': correct_count,
        'total_samples': n_samples,
        'temperature': temperature,
        'n_fewshots': n_fewshots
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate pass@k on GSM8K')
    
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--n_samples', type=int, default=512,
                        help='Number of samples per problem (default: 512)')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of problems to evaluate (default: all)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.6],
                        help='List of temperatures to evaluate (default: [0.6])')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling threshold (default: 0.95)')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Maximum generation length (default: 2048)')
    parser.add_argument('--n_fewshots', type=int, default=0,
                        help='Number of few-shot examples to include (default: 0, max: 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print generated outputs and detailed evaluation')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    problems = load_gsm8k(args.subset_size)
    
    for temperature in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Evaluating with temperature={temperature}, n_fewshots={args.n_fewshots}")
        print(f"{'='*60}\n")
        
        results = []
        
        for problem in tqdm(problems, desc=f"T={temperature}", disable=args.verbose):
            result = evaluate_problem(
                model, tokenizer, problem,
                args.n_samples, temperature,
                args.top_p, args.max_tokens,
                args.n_fewshots, args.verbose
            )
            results.append(result)
        
        output_file = os.path.join(
            args.output_dir,
            f"results_temp{temperature}_n{args.n_samples}_fewshot{args.n_fewshots}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to: {output_file}")
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
