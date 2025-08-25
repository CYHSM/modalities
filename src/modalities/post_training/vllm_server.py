#!/usr/bin/env python3
"""Helper script to start vLLM server for GRPO training."""
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Start vLLM server for GRPO training")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID for vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")
    parser.add_argument("--hf-home", default="/raid/s3/opengptx/mfrey/huggingface", help="HF cache dir")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for models")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HOME"] = args.hf_home

    # Build command
    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    # Conditionally add the flag if it's passed to your script
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"🚀 Starting vLLM server on GPU {args.gpu}")
    print(f"📍 Server will be available at http://{args.host}:{args.port}")
    print(f"📦 Model: {args.model_path}")
    print(f"💾 GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"🔒 Trust remote code: {args.trust_remote_code}")
    print("-" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        # Run the server
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down vLLM server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting vLLM server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
