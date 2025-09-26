import pickle
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationCapture:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def capture_single(self, input_text, max_new_tokens=1, layers_to_capture=None):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        activations = OrderedDict()
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu().numpy()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        activations[name] = output[0].detach().cpu().numpy()

            return hook

        for name, module in self.model.named_modules():
            if layers_to_capture is None or any(pattern in name for pattern in layers_to_capture):
                if len(list(module.children())) == 0:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            outputs = self.model(input_ids)

        for hook in hooks:
            hook.remove()

        logits = outputs.logits.detach().cpu().numpy()

        return {"activations": activations, "input_ids": input_ids.cpu().numpy(), "logits": logits, "text": input_text}

    def capture_generation(self, input_text, *, max_new_tokens=10, layers_to_capture=None):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        generation_activations = []
        step_texts = []
        generated_tokens = input_ids.clone()
        
        initial_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        step_texts.append(initial_text)
        
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
            
            for name, module in self.model.named_modules():
                if layers_to_capture is None or any(pattern in name for pattern in layers_to_capture):
                    if len(list(module.children())) == 0:
                        hooks.append(module.register_forward_hook(make_hook(name)))
            
            with torch.no_grad():
                outputs = self.model(generated_tokens)
            
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, keepdim=True).unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
            
            current_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            step_texts.append(current_text)
            
            generation_activations.append(step_activations)
            
            for hook in hooks:
                hook.remove()
        
        return generation_activations, step_texts

    def capture_batch(self, texts, max_new_tokens=1, layers_to_capture=None):
        results = []
        for i, text in enumerate(texts):
            print(f"  Processing {i+1}/{len(texts)}: {text[:50]}...")
            result = self.capture_single(text, max_new_tokens, layers_to_capture)
            results.append(result)
        return results
    
    def capture_prompts(self, prompts, *, max_new_tokens=1, layers_to_capture=None):
        all_activations = []
        all_texts = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Processing {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            if max_new_tokens == 1:
                result = self.capture_single(prompt, max_new_tokens, layers_to_capture)
                activations = [result['activations']]
                texts = [result['text']]
            else:
                activations, texts = self.capture_generation(prompt, max_new_tokens=max_new_tokens, layers_to_capture=layers_to_capture)
            
            all_activations.append(activations)
            all_texts.append(texts)
        
        return all_activations, all_texts


class DataStore:
    @staticmethod
    def save_experiment(data, experiment_name, output_dir="/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests"):
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / "data.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        metadata_file = output_path / "metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Conditions: {list(data.keys())}\n")
            for condition, samples in data.items():
                f.write(f"  {condition}: {len(samples)} samples\n")

        return output_path

    @staticmethod
    def load_experiment(experiment_name, output_dir="/raid/s3/opengptx/mfrey/cp_analysis/inference_vis/tests"):
        filepath = Path(output_dir) / experiment_name / "data.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_generation_data(*, activations, texts, output_path):
        filepath = output_path / "generation_data.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump({'activations': activations, 'texts': texts}, f)
        return filepath

    @staticmethod
    def load_generation_data(output_path):
        filepath = Path(output_path) / "generation_data.pkl"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['activations'], data['texts']


def extract_layer_activations(activation_dict, layer_pattern):
    layer_acts = {}
    for name, act in activation_dict.items():
        if layer_pattern in name:
            layer_acts[name] = act
    return layer_acts


def get_activation_at_position(activation, position=-1):
    if len(activation.shape) == 3:
        return activation[:, position, :]
    elif len(activation.shape) == 2:
        return activation[position, :]
    else:
        return activation.flatten()