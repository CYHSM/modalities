import torch
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

def capture_activations(model_path, input_text, max_new_tokens=5):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    model.eval()
    
    generation_activations = []
    generated_tokens = input_ids.clone()
    
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
        
        generation_activations.append(step_activations)
        
        for hook in hooks:
            hook.remove()
    
    return generated_tokens, generation_activations

def save_activations(activations, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(activations, f)

def load_activations(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model_path = "/raid/s3/opengptx/mfrey/instruct/hf_model"
    input_text = "The quick brown fox"
    
    tokens, activations = capture_activations(model_path, input_text, max_new_tokens=10)
    save_activations(activations, "data/activations.pkl")
    print(f"Saved {len(activations)} steps to data/activations.pkl")