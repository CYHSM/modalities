import torch
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

def capture_activations(model_path, input_text, max_new_tokens=5, verbose=True):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    model.eval()
    
    generation_activations = []
    step_texts = []
    generated_tokens = input_ids.clone()
    
    initial_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    step_texts.append(initial_text)
    
    if verbose:
        print(f"Initial text: '{initial_text}'")
        print(f"Generating {max_new_tokens} tokens...\n")
    
    for step in range(max_new_tokens):
        step_activations = OrderedDict()
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    step_activations[name] = output[:, -1, ...].detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        step_activations[name] = output[0][:, -1, ...].detach().cpu()
            return hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        with torch.no_grad():
            outputs = model(generated_tokens)
        
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, keepdim=True).unsqueeze(0)
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
        
        current_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        step_texts.append(current_text)
        
        if verbose:
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f"Step {step + 1}: Generated token '{new_token}'")
            print(f"  Full text: '{current_text}'")
            print()
        
        generation_activations.append(step_activations)
        
        for hook in hooks:
            hook.remove()
    
    if verbose:
        print(f"✓ Completed generation: {len(generation_activations)} steps")
        print(f"✓ Final text length: {len(step_texts[-1])} characters")
    
    return generated_tokens, generation_activations, step_texts

def save_activations(activations, step_texts, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {'activations': activations, 'texts': step_texts}
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_activations(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return data, None
    return data['activations'], data.get('texts', None)

if __name__ == "__main__":
    model_path = "/raid/s3/opengptx/mfrey/instruct/hf_model"
    input_text = "The quick brown fox"
    
    tokens, activations, texts = capture_activations(model_path, input_text, max_new_tokens=10, verbose=True)
    save_activations(activations, texts, "data/activations.pkl")
    print(f"\nSaved {len(activations)} steps to data/activations.pkl")