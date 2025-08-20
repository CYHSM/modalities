#!/usr/bin/env python3
"""Interactive model testing script for manual evaluation."""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from evaluation import parse_and_log_results, run_lighteval_cli
from model_utils import generate_response, get_model_info, load_model_and_tokenizer

from config import EvaluationConfig, ModelConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test and evaluate model checkpoints")

    # Model arguments
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    # Generation arguments (for interactive mode)
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")

    # Test mode arguments
    parser.add_argument(
        "--mode",
        choices=["interactive", "evaluate"],
        default="interactive",
        help="Testing mode: interactive chat or LightEval evaluation",
    )

    # Evaluation arguments (for evaluate mode)
    parser.add_argument("--eval-gpu", type=int, default=6, help="GPU for evaluation")
    parser.add_argument("--eval-samples", type=int, default=500, help="Max samples for evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument(
        "--eval-tasks", default="leaderboard|hellaswag|0|1,leaderboard|gsm8k|0|1", help="LightEval tasks to run"
    )
    parser.add_argument("--hf-home", default="/raid/s3/opengptx/mfrey/huggingface", help="HF cache directory")

    # Environment arguments
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--cuda-devices", default="0", help="CUDA visible devices")

    return parser.parse_args()


def interactive_mode(model, tokenizer, args):
    """Run interactive testing mode."""
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE MODEL TESTING")
    print("=" * 60)
    print("Type your questions/problems and see model responses.")
    print("Commands:")
    print("  /quit or /exit - Exit the program")
    print("  /info - Show model information")
    print("  /settings - Show current generation settings")
    print("  /help - Show this help message")
    print("=" * 60 + "\n")

    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "do_sample": not args.no_sample,
        "top_p": args.top_p,
    }

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "/info":
                info = get_model_info(model, tokenizer)
                print("\n📊 Model Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                continue
            elif user_input.lower() == "/settings":
                print("\n⚙️  Generation Settings:")
                for key, value in generation_kwargs.items():
                    print(f"  {key}: {value}")
                continue
            elif user_input.lower() == "/help":
                print("\n❓ Commands:")
                print("  /quit or /exit - Exit the program")
                print("  /info - Show model information")
                print("  /settings - Show current generation settings")
                print("  /help - Show this help message")
                continue

            # Generate response
            print("\n🤖 Model: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input, **generation_kwargs)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def evaluate_mode(model_path: str, args):
    """Run LightEval evaluation on the checkpoint."""
    print("\n" + "=" * 60)
    print("🔍 LIGHTEVAL BENCHMARK EVALUATION")
    print("=" * 60)

    # Create evaluation config
    eval_config = EvaluationConfig(
        gpu_id=args.eval_gpu, max_samples=args.eval_samples, batch_size=args.eval_batch_size, tasks=args.eval_tasks
    )

    print(f"Evaluating checkpoint: {model_path}")
    print(f"Tasks: {eval_config.tasks}")
    print(f"Max samples per task: {eval_config.max_samples}")
    print(f"GPU: {eval_config.gpu_id}")
    print("=" * 60 + "\n")

    # Run evaluation
    print("🚀 Starting LightEval evaluation...")
    results = run_lighteval_cli(
        checkpoint_path=model_path,
        step=0,  # Use 0 since this is standalone evaluation
        eval_config=eval_config,
        hf_home=args.hf_home,
    )

    if results is None:
        print("❌ Evaluation failed!")
        return

    print("✅ Evaluation completed successfully!")

    # Parse and display results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    parsed_results = parse_and_log_results(results, step=0)

    if parsed_results:
        # Display results in a nice format
        for metric_name, value in parsed_results.items():
            if metric_name.startswith("eval/"):
                clean_name = metric_name.replace("eval/", "")
                if isinstance(value, float):
                    print(f"  {clean_name:<30}: {value:.4f}")
                else:
                    print(f"  {clean_name:<30}: {value}")

    # Save detailed results
    results_file = Path(model_path).parent / f"lighteval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file}")
    print("=" * 60)


def main():
    """Main function."""
    args = parse_args()

    # Setup environment
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    logger.info(f"Testing model: {args.model_path}")
    logger.info(f"Mode: {args.mode}")

    try:
        if args.mode == "evaluate":
            # For evaluation mode, we don't need to load the model in Python
            # LightEval will handle model loading
            evaluate_mode(args.model_path, args)

        elif args.mode == "interactive":
            # Load model and tokenizer for interactive mode
            logger.info("Loading model and tokenizer...")

            model_config = ModelConfig(model_path=args.model_path, device=args.device, torch_dtype=args.torch_dtype)

            model, tokenizer = load_model_and_tokenizer(
                model_config.model_path,
                device=model_config.device,
                torch_dtype=model_config.get_torch_dtype(),
                trust_remote_code=model_config.trust_remote_code,
                device_map=model_config.device_map,
            )

            logger.info("Model loaded successfully")
            interactive_mode(model, tokenizer, args)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
