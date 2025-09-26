import torch
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

def capture_activations(model_path, input_text, max_new_tokens=10):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    model.eval()
    
    generation_activations = []
    step_texts = []
    generated_tokens = input_ids.clone()
    
    initial_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    step_texts.append(initial_text)
    
    print(f"Initial: '{initial_text}'")
    print(f"Generating {max_new_tokens} tokens...")
    
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
        
        new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(f"  Step {step + 1}: +'{new_token}'")
        
        generation_activations.append(step_activations)
        
        for hook in hooks:
            hook.remove()
    
    return generation_activations, step_texts

def save_data(activations, texts, output_dir="output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "activations.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump({'activations': activations, 'texts': texts}, f)
    return filepath

def load_data(output_dir="output"):
    filepath = Path(output_dir) / "activations.pkl"
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['activations'], data['texts']