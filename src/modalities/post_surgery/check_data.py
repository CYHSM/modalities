import pickle
from pathlib import Path
import sys

def check_step(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"\n=== Step {data['step']} ===")
    print(f"Loss: {data['loss']:.4f}")
    print(f"Sample text: {data['batch_text'][:100]}...")
    
    print(f"\nGradients captured: {len([k for k, v in data['gradients'].items() if v is not None])}")
    print(f"Weight deltas captured: {len(data['weight_deltas'])}")
    
    # Show a few gradient norms
    print("\nSample gradient norms:")
    for name, grad in list(data['gradients'].items())[:3]:
        if grad is not None:
            print(f"  {name}: {grad.norm().item():.2e}")

if __name__ == "__main__":
    dynamics_dir = Path(sys.argv[1]) / "dynamics"
    files = sorted(dynamics_dir.glob("step_*.pkl"))
    
    print(f"Found {len(files)} dynamics files")
    for f in files[:3]:  # Show first 3 steps
        check_step(f)
