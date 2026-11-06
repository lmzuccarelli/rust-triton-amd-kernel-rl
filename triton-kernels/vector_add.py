import torch
import triton
import triton.language as tl


# Deliberately un-optimized baseline: tiny BLOCK, no contiguity hints,
# no autotune. Intended as input for the optimizer agent.
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK = 64
    grid = (triton.cdiv(n_elements, BLOCK),)
    add_kernel[grid](x, y, out, n_elements, BLOCK=BLOCK)
    return out


def main():
    # PyTorch exposes ROCm devices through its "cuda" device API. Confirm that
    # this is a ROCm build so the example cannot silently run on NVIDIA CUDA.
    if torch.version.hip is None:
        raise RuntimeError(
            "This example targets AMD GPUs via ROCm; install a ROCm-enabled "
            "PyTorch build. (PyTorch still names ROCm devices 'cuda'.)"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("No AMD GPU is available to the ROCm-enabled PyTorch build.")

    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"Running on AMD GPU via ROCm: {torch.cuda.get_device_name(device)}")
    x = torch.rand(1_000_003, device=device, dtype=torch.float32)
    y = torch.rand(1_000_003, device=device, dtype=torch.float32)
    out = add(x, y)
    assert torch.allclose(out, x + y), "mismatch vs torch reference"
    print("ok")


if __name__ == "__main__":
    main()
