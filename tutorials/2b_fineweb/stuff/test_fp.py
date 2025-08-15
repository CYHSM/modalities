import torch


# Check if your CPU actually executes in bfloat16
def test_actual_computation():
    x = torch.randn(100, 100, dtype=torch.bfloat16)

    # Force a computation that would show precision differences
    x_cpu = x.cpu()
    x_gpu = x.cuda()

    # Direct bfloat16 softmax
    sm_cpu = torch.softmax(x_cpu, dim=-1)
    sm_gpu = torch.softmax(x_gpu, dim=-1)

    # Compare CPU vs GPU results
    diff = (sm_cpu.cuda() - sm_gpu).abs().max()
    print(f"CPU vs GPU bfloat16 softmax diff: {diff}")

    # Now test if CPU is secretly using fp32
    sm_cpu_as_fp32 = torch.softmax(x_cpu.float(), dim=-1).to(torch.bfloat16)
    diff_fp32 = (sm_cpu - sm_cpu_as_fp32).abs().max()
    print(f"CPU bf16 vs CPU fp32->bf16 diff: {diff_fp32}")


# Run the test
if __name__ == "__main__":
    test_actual_computation()
