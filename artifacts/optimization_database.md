# Comprehensive Triton / ROCm (AMD) Optimization Strategy Database

This database contains optimization strategies with their expected performance gains for different GPU performance states, targeting **OpenAI Triton kernels running on AMD GPUs via ROCm** (CDNA2/CDNA3 — MI250/MI300, and RDNA).

## Database Philosophy

This database serves as a **living knowledge base** that:

- Maps performance states to Triton/ROCm optimization strategies
- Tracks actual vs predicted performance improvements
- Learns from optimization results to improve future predictions
- Provides comprehensive context for LLM-powered optimization
- Unknown performance improvements are listed as 0%

## Performance State Categories

<!-- PERFORMANCE_STATE_CATEGORIES: generated from optimization_database.json -->

## Usage Guidelines for LLM Integration

### For State Analysis

When analyzing rocprof / omniperf (rocprofiler-compute) reports, the LLM should:

1. Identify primary and secondary bottlenecks
2. Map to the closest state in this database
3. Consider hardware context (CDNA2/CDNA3, wavefront = 64) and kernel characteristics
4. Account for potential state transitions after optimization

### For Optimization Selection

When selecting optimizations, the LLM should:

1. Prioritize high-confidence, high-impact optimizations
2. Prefer Triton-level knobs (BLOCK_M/N/K, num_warps, num_stages, GROUP_SIZE_M, waves_per_eu, matrix_instr_nonkdim, kpack) before restructuring the algorithm
3. Account for potential side effects and trade-offs (e.g. larger tiles raise VGPR/LDS pressure and lower occupancy)
4. Adapt parameters based on kernel-specific characteristics

### For Performance Prediction

When predicting improvements, the LLM should:

1. Use base predictions as starting points
2. Adjust based on confidence scores and historical accuracy
3. Consider kernel context and similarity to previous cases
4. Provide uncertainty ranges rather than point estimates

## Integration with Comprehensive GPU Optimization Knowledge

This database integrates with the GPU optimization decision tree to provide:

- **Hierarchical optimization strategies**: From high-level decisions to specific Triton implementations
- **Context-aware recommendations**: Based on profiling data and performance characteristics
- **Multi-objective optimization**: Balancing performance, accuracy, and maintainability
- **Hardware-specific guidance**: Tailored recommendations for different AMD GPU architectures

The LLM agents use this database as a **living reference** that evolves based on actual optimization results.

## IMPORTANT

When targeting kernels for AMD CDNA3 (MI300X / MI300A, `gfx942`) take note of the following.

CRITICAL HARDWARE CONSTRAINTS:

1. Target the ROCm backend. The wavefront (warp) size is **64 lanes**, not 32 — reductions, masks and any lane-index logic must assume 64.
2. Matrix math must go through `tl.dot`, which lowers to **MFMA** (Matrix Core) instructions. Do NOT hand-write `wmma`/`mma.sync`/`tcgen05` — those are NVIDIA PTX and will not compile for ROCm.
3. Keep per-workgroup **LDS (shared memory) under ~64 KB** (CDNA3 has 64 KB LDS per CU); exceeding it fails to launch or kills occupancy.
4. Prefer **bf16/fp16** inputs with **fp32 accumulation**; CDNA3 adds native **fp8 (OCP e4m3/e5m2)** for `tl.dot` — use it where accuracy allows to feed the Matrix Cores and relieve HBM bandwidth.
5. AMD-specific Triton autotune knobs that matter most on CDNA: `matrix_instr_nonkdim` (16 or 32 — the MFMA tile size), `waves_per_eu` (occupancy hint), `kpack` (K-packing for LDS), alongside the portable `num_warps` / `num_stages` / `BLOCK_*` / `GROUP_SIZE_M`.
6. Do not set `allow_tf32` / rely on TF32 — it is an NVIDIA Ampere+ concept and has no CDNA equivalent.

For CDNA2 (MI250, `gfx90a`) the same rules apply except there is **no fp8** path and the MFMA shapes differ; stay on bf16/fp16.

### Learned Optimization Strategies

#### Expert Technique: matrix_core_utilization

The outline below targets AMD **Matrix Cores (MFMA)** through Triton's `tl.dot`.
In Triton you never emit MFMA directly — you write a blocked GEMM with `tl.dot`
and the ROCm backend selects the MFMA instruction for the operand dtypes and the
`matrix_instr_nonkdim` setting.

Key points (NVIDIA WMMA / CuTe `cute.gemm` -> Triton `tl.dot`):

1. Load `BLOCK_M x BLOCK_K` and `BLOCK_K x BLOCK_N` operand tiles with `tl.load`
   (masked on the edges), then `acc += tl.dot(a, b)` in an fp32 accumulator.
2. Let the autotuner pick the MFMA tile via `matrix_instr_nonkdim` (16 or 32).
3. Use `num_stages >= 2` so the next K-tile is prefetched while the current MFMA runs.
4. Reorder program ids with a `GROUP_SIZE_M` super-group so co-scheduled
   workgroups reuse A/B rows out of L2 (the single biggest matmul win on AMD).
5. Inputs bf16/fp16 (or fp8 on CDNA3), accumulate fp32, cast on store.

**Usage Examples** (Triton + ROCm, `triton` 3.x):

```python
# matrix_core_utilization - blocked GEMM C = A @ B on AMD Matrix Cores (MFMA).
# A is (M,K) row-major, B is (K,N) row-major, C is (M,N). Inputs fp16/bf16,
# accumulation fp32. Tails handled by masked tl.load / tl.store (no zero-pad copy).
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Portable knobs + AMD-specific (matrix_instr_nonkdim, waves_per_eu, kpack)
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8,
             "matrix_instr_nonkdim": 16, "waves_per_eu": 2, "kpack": 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8,
             "matrix_instr_nonkdim": 16, "waves_per_eu": 2, "kpack": 2},
            num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # L2 cache locality: group programs so neighbours share A rows / B cols.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 accumulator
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask.T, other=0.0)
        acc += tl.dot(a, b)                                 # lowers to MFMA
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

**Notes / pitfalls (Triton / ROCm):**

- **`tl.dot` is the MFMA path.** There is no `wmma`/`mma.sync`; do not import `cutlass`.
- **Tails use masks, not packing.** Pass `mask=` to `tl.load`/`tl.store`; out-of-range reads return `other`.
- **`GROUP_SIZE_M` is the L2 lever.** On AMD this super-grouping is usually the largest single matmul speedup — keep it in the config space.
- **Occupancy.** `waves_per_eu`, `BLOCK_*` and `num_stages` trade VGPR/LDS against resident waves; always autotune, do not hard-code one config.
- **Wavefront = 64.** Any manual reduction or lane logic must assume 64 lanes.

#### Expert Technique: software_pipelining

**Performance Impact**: 0% improvement
**Confidence Score**: 0.90
**Applicable States**: memory_latency_bound, compute_throughput_saturated, memory_compute_balanced

**Implementation Hints**:

- `num_stages >= 2` prefetches the next tile while the current one computes
- Most effective when the K-loop is long and global-load latency dominates
- Costs LDS/registers; back off if it drops occupancy (check with omniperf)

**Usage Examples** (Triton + ROCm):

```python
# Software pipelining is a scheduling knob, not kernel code: expose num_stages
# in the autotune space and let the backend overlap loads with compute.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel(...):
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=..., other=0.0)   # prefetched across stages
        b = tl.load(b_ptrs, mask=..., other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
```

#### Expert Technique: l2_cache_locality

**Performance Impact**: 0% improvement
**Confidence Score**: 0.95
**Applicable States**: memory_latency_bound, cache_inefficient, memory_bandwidth_saturated

**Implementation Hints**:

- Super-group program ids by `GROUP_SIZE_M` (try 4 / 8 / 16)
- Co-scheduled workgroups then reuse the same A rows / B cols from L2
- Usually the highest-ROI change for GEMM-like kernels on AMD

**Usage Examples** (Triton + ROCm):

```python
# Workgroup swizzling for L2 reuse (the grouped-launch map used in the GEMM above).
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

#### Expert Technique: vectorized_memory_access

**Performance Impact**: 0% improvement
**Confidence Score**: 0.85
**Applicable States**: memory_bandwidth_saturated, memory_latency_bound

**Implementation Hints**:

- Make the last tensor dim stride-1 so block loads are contiguous
- Hint the compiler with `tl.max_contiguous` / `tl.multiple_of` to unlock wide (dwordx4) global loads
- Larger `BLOCK` along the contiguous dim improves transaction width

**Usage Examples** (Triton + ROCm):

```python
# Elementwise/copy kernel with contiguity hints so the backend emits wide loads.
@triton.jit
def scale_kernel(x_ptr, y_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)  # vectorization hint
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x * scale, mask=mask)
```

#### Expert Technique: occupancy_tuning

**Performance Impact**: 0% improvement
**Confidence Score**: 0.85
**Applicable States**: low_occupancy_vgpr_pressure, low_occupancy_lds, latency_occupancy_limited

**Implementation Hints**:

- `waves_per_eu` steers the register allocator toward more resident waves
- Smaller `BLOCK_*` and fewer `num_stages` cut VGPR/LDS, raising occupancy
- Verify waves/SIMD and VGPR/LDS usage with omniperf, not by guessing

**Usage Examples** (Triton + ROCm):

```python
# Occupancy is driven by block shape + the AMD waves_per_eu hint; autotune them.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024, "waves_per_eu": 0}, num_warps=4),   # 0 = no hint
        triton.Config({"BLOCK": 512,  "waves_per_eu": 2}, num_warps=4),
        triton.Config({"BLOCK": 256,  "waves_per_eu": 4}, num_warps=2),
    ],
    key=["n"],
)
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    ...
```

#### Expert Technique: packed_math

**Performance Impact**: 0% improvement
**Confidence Score**: 0.60
**Applicable States**: compute_throughput_saturated, hybrid_bound, memory_bandwidth_saturated

**Implementation Hints**:

- Keep data in fp16/bf16 so the VALU uses packed (v_pk_*) math (2 FLOPs/lane/cycle)
- Only up-cast to fp32 for the final accumulation / reduction
- Halves both register footprint and HBM traffic versus fp32

**Usage Examples** (Triton + ROCm):

```python
# Fused bf16 elementwise with packed math; accumulate/reduce in fp32 only at the end.
@triton.jit
def gelu_bf16_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)          # bf16 in -> packed VALU
    xf = x.to(tl.float32)                           # up-cast only for the math that needs it
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(y_ptr + offs, y.to(x.dtype), mask=mask)
```

### Expert Technique Combinations

**Combination**: l2_cache_locality + software_pipelining + matrix_core_utilization

- Average Performance: 0% speedup
- Confidence: 0.70
- Frequency in top solutions: common for GEMM on CDNA

**Combination**: block_tiling + occupancy_tuning

- Average Performance: 0% speedup
- Confidence: 0.60
- Frequency in top solutions: 3 times

## Expert-Learned Optimizations (Auto-Generated)

### Expert Technique: l2_cache_locality (auto-learned)

**Source**: AMD Triton GEMM tuning (GROUP_SIZE_M)
**Performance Impact**: 0% improvement
**Confidence**: 0.95

**Implementation Strategy**:

- Super-group program ids (GROUP_SIZE_M = 4/8/16)
- Highly effective for matmul / attention projection workloads

**Usage Context**:
Best applied to: memory_bandwidth_saturated, memory_latency_bound, cache_inefficient

### Expert Technique: matrix_core_utilization (auto-learned)

**Source**: AMD Triton GEMM tuning (tl.dot + matrix_instr_nonkdim)
**Performance Impact**: 0% improvement
**Confidence**: 0.90

**Implementation Strategy**:

- Route matmul through tl.dot; autotune matrix_instr_nonkdim (16/32)
- bf16/fp16 in, fp32 accumulate (fp8 on CDNA3)

**Usage Context**:
Best applied to: compute_throughput_saturated, memory_compute_balanced

### Expert Technique: software_pipelining (auto-learned)

**Source**: AMD Triton GEMM tuning (num_stages)
**Performance Impact**: 0% improvement
**Confidence**: 0.90

**Implementation Strategy**:

- num_stages >= 2 to overlap global loads with MFMA
- Back off if occupancy drops

**Usage Context**:
Best applied to: memory_latency_bound, memory_compute_balanced, compute_throughput_saturated
