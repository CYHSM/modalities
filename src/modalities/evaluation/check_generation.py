import torch
import os
from pathlib import Path
# Assuming your new code is saved in src/modalities/evaluation/olmes_evaluator.py
from modalities.evaluation.olmes_evaluator import (
    ModalitiesOLMESWrapper, 
    load_modalities_model
)

def run_sanity_check(checkpoint_path, config_path=None, model_key="model_raw"):
    """
    Tests if the model loads correctly (DCP or standard) and produces text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Phase 1: Loading Model (Device: {device}) ---")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # load_modalities_model now handles the DCP -> Torch conversion internally
    try:
        model, tokenizer, config = load_modalities_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            model_key=model_key
        )
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # Wrap for OLMES
    olmes_model = ModalitiesOLMESWrapper(
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        device=torch.device(device)
    )

    print("\n--- Phase 2: Testing Tokenization ---")
    text = "Hello, how are you?"
    tokens = olmes_model.tok_encode(text)
    decoded = olmes_model.tok_decode(tokens)
    print(f"Input text: {text}")
    print(f"Encoded tokens: {tokens}")
    print(f"Decoded check: {decoded}")

    print("\n--- Phase 3: Testing Loglikelihood (Scoring) ---")
    from oe_eval.components.requests import LoglikelihoodRequest
    
    # We compare a factual continuation vs a nonsensical one
    reqs = [
        LoglikelihoodRequest(context="The capital of Germany is", continuation=" Berlin"),
        LoglikelihoodRequest(context="The capital of Germany is", continuation=" Pizza")
    ]
    
    results = olmes_model.loglikelihood_verbose(reqs)
    for i, res in enumerate(results):
        print(f"Context: '{reqs[i].context}' | Cont: '{reqs[i].continuation}'")
        print(f"  -> Sum Logits: {res['sum_logits']:.4f} | Tokens: {res['num_tokens']}")

    print("\n--- Phase 4: Testing Text Generation ---")
    from oe_eval.components.requests import GenerateUntilRequest
    
    # This triggers the token-by-token loop in your wrapper
    gen_req = [GenerateUntilRequest(
        context="The Milky Way is an",
        #stop_sequences=["\n", "."],
        generation_kwargs={"max_gen_toks": 300, "temperature": 1.0}
    )]
    
    print("Generating...")
    gen_results = olmes_model.generate_until_verbose(gen_req)
    print(f"Prompt: {gen_req[0].context}")
    print(f"Output: {gen_results[0]['continuation']}")
    print("\n--- Sanity Check Complete ---")

if __name__ == "__main__":
    CHECKPOINT_DIR = "/raid/s3/opengptx/mfrey/loop/checkpoints/2026-01-07__09-43-23_17c44997f407444e/eid_2026-01-07__09-43-23_17c44997f407444e-seen_steps_30000-seen_tokens_10813440000-target_steps_151380-target_tokens_54564618240"
    run_sanity_check(checkpoint_path=CHECKPOINT_DIR)