# tinyfuse

[![tests](https://github.com/PS12007/tinyfuse/actions/workflows/tests.yml/badge.svg)](https://github.com/PS12007/tinyfuse/actions/workflows/tests.yml)

tinyfuse is a small ML compiler I wrote in Python and C. You write a model as
ordinary Python using a NumPy-like API. tinyfuse traces it into a graph,
optimizes and fuses the graph, generates C with tiled, vectorized loops,
compiles that C with `gcc -O3 -march=native`, and calls it through `ctypes`.

It handles a 2-layer MLP and a full transformer block (multi-head attention,
GELU MLP, layernorm, residuals). On one core it runs them 1.2 to 1.7 times
faster than the same code in NumPy. On elementwise-heavy code, where fusion
has the most to do, the fused kernels are 2 to 7 times faster than PyTorch eager.

![tinyfuse demo: compile an MLP, print the fused IR, check against NumPy](docs/img/demo.gif)

**Contents:** [Quick start](#quick-start) ·
[How it works](#how-it-works) · [Results](#results) ·
[Things that went wrong](#things-that-went-wrong-along-the-way) ·
[Tests](#tests) · [Limitations](#limitations) · [Repo layout](#repo-layout)

## Quick start

You need Python 3.9+, NumPy and GCC. For the benchmarks you also need
matplotlib and PyTorch.

```bash
pip install numpy pytest
```

```python
import numpy as np
import tinyfuse as tf

def mlp(x, w1, b1, w2, b2):
    return tf.relu(x @ w1 + b1) @ w2 + b2

f = tf.compile(mlp, x=(32, 64), w1=(64, 128), b1=(128,), w2=(128, 10), b2=(10,))

print(f.graph)       # IR as traced
print(f.optimized)   # IR after the passes
print(f.source)      # the generated C
y = f(x, w1, b1, w2, b2)   # NumPy arrays in, NumPy arrays out
```

Run the tests (329 of them, about 20 seconds; the first run spends most of
that compiling C):

```bash
python -m pytest -q
```

Other entry points:

```bash
python examples/demo.py                      # the run in the GIF above
```

```bash
python examples/show_transformer_ir.py       # transformer IR before/after fusion
```

```bash
python benchmarks/run_benchmarks.py --threads 1
```

```bash
python -m tinyfuse.autotune --shape 256x1024x4096 --db my_tuning.json
```

```bash
python docs/make_figures.py                  # regenerate every image in this README
```

Supported ops: `matmul` (2-D and batched), `add`, `sub`, `mul`, `div`,
`maximum`, `relu`, `gelu` (tanh form), `sigmoid`, `tanh`, `exp`, `neg`, `sqrt`,
`rsqrt`, `sum`/`mean`/`max` (last axis), `softmax` and `layernorm` (last
axis), `reshape`, `transpose`.

## How it works

There are four stages: trace, optimize, generate C, run. The reasoning behind
each decision is in [DESIGN.md](DESIGN.md). This section is the short version.

### 1. Tracing into a graph

`tf.compile` calls your function with symbolic tensors. Each operation adds a
node to a graph, and Python control flow simply runs during tracing, the same
way `torch.fx` or `jax.jit` work. Every node has a static float32 shape, which
is inferred the moment the node is created. So a shape mismatch fails on the
line that caused it, not later inside the compiler.

The IR is a flat SSA graph. Every op has a NumPy reference implementation. That
implementation is the definition of what the op means: the interpreter, the
constant folder and the tests all use it. There is a verifier that checks
operand order, arity, shapes and fused-op programs, and the pass manager runs
it after every pass.

### 2. Optimizing and fusing

The `fused` pipeline runs constant folding, CSE and DCE, then three fusion
passes and a layout pass:

* **Matmul epilogue fusion.** Starting at each matmul, it follows the chain of
  elementwise ops that consume the result: bias add, activation, residual add,
  scaling. Each must be the previous op's only user, and the shape must not
  change. The whole chain becomes the matmul's *epilogue*, a small expression
  evaluated on each output tile.
* **Elementwise fusion.** Whatever elementwise chains are left get merged into
  single loops. A node joins a group only if all of its users are already in
  that group. That rule is what guarantees fusion can never create a cycle.
* **Reduction fusion.** An elementwise chain whose only consumer is a `sum`,
  `mean` or `max` is evaluated inside the reduction loop, so `mean(x*x)` reads
  `x` once and never writes `x*x` to memory.
* **Layout folding.** A transpose, or a reshape of contiguous data, is just a
  strided view of its input. When one feeds a matmul, the matmul reads the
  original buffer through those strides during packing, and the transpose
  kernel disappears. In attention that removes the q/k/v head splits and `k^T`.

On the transformer block this takes 29 ops down to 13, and 25 kernels down
to 12:

![Transformer block graph before and after fusion](docs/img/fusion_graph.png)

The printed IR shows the same thing. Here is a 2-layer MLP before:

```
graph mlp(%x: f32[32,64], %w1: f32[64,128], %b1: f32[128], %w2: f32[128,10], %b2: f32[10]) {
  %0: f32[32,128] = matmul(%x, %w1)
  %1: f32[32,128] = add(%0, %b1)
  %2: f32[32,128] = relu(%1)
  %3: f32[32,10] = matmul(%2, %w2)
  %4: f32[32,10] = add(%3, %b2)
  return %4
}
```

and after:

```
graph mlp(%x: f32[32,64], %w1: f32[64,128], %b1: f32[128], %w2: f32[128,10], %b2: f32[10]) {
  %0: f32[32,128] = fused_matmul(%x, %w1, %b1) {epilogue=[t0=add(acc,in0); t1=relu(t0)]}
  %1: f32[32,10] = fused_matmul(%0, %w2, %b2) {epilogue=[t0=add(acc,in0)]}
  return %1
}
```

In the transformer the residual add is absorbed too, and the attention scores
read q and `k^T` straight out of the projection outputs:

```
%4: f32[4,64,64] = fused_matmul(%1, %2) {a_view=f32[4,64,64] strides [64,256,1],
                     b_view=f32[4,64,64] strides [64,1,256], epilogue=[t0=mul(acc,0.125)]}
%9: f32[64,256]  = fused_matmul(%8, %wo, %bo, %x) {epilogue=[t0=add(acc,in0); t1=add(in1,t0)]}
```

Reductions mean you can write your own normalization layers. An RMSNorm:

```python
def rmsnorm(x, g):
    return x * tf.rsqrt(tf.mean(x * x) + 1e-6) * g
```

compiles to three loops: one fused reduction and two elementwise kernels.

```
%0: f32[64,1]   = fused_reduce(%x) {kind=mean, program=[t0=mul(in0,in0)], ...}
%1: f32[64,1]   = fused_elementwise(%0) {program=[t0=add(in0,1e-06); t1=rsqrt(t0)]}
%2: f32[64,256] = fused_elementwise(%x, %1, %g) {program=[t0=mul(in0,in1); t1=mul(t0,in2)]}
```

`%1` is left as its own small kernel on purpose. Fusing it into `%2` would mean
computing the `rsqrt` once per element instead of once per row.

Layout folding is correct and tested, but I couldn't measure a speedup from
it: the transposes were only about 2% of the runtime, and reading `k^T`
through a strided gather while packing costs some of that back.

### 3. Generating C

The whole graph becomes one C file. Each node gets its own `static` function
with every shape baked in as a constant, and a single `tf_run()` calls them in
order, so one call from Python runs the entire model.

The matmul kernel is where most of the time goes, so it got most of the
effort. It follows the GotoBLAS structure. Blocks of A and B are copied
("packed") into small contiguous panels. Then a micro-kernel computes a 6×64
tile of C while keeping all of it in 24 AVX-512 registers, written with GCC
vector extensions. When the last K block of a cache block is done, the fused
epilogue runs over that block while it is still in cache:

![Animation of the blocked matmul: packing panels, micro-kernel tiles, epilogue](docs/img/matmul_tiling.gif)

My first version left the vectorizing to GCC, using a plain `float acc[6][32]`.
GCC kept that array on the stack and the kernel ran at 1.8 GFLOP/s. With
explicit vector types and direct loads it reaches 95 to 125 GFLOP/s on one core
of this machine, slightly ahead of OpenBLAS at these sizes.

Elementwise kernels, fused or not, merge every dimension that is contiguous for
all of their operands and emit a single `#pragma omp simd` loop that evaluates
the fused expression in registers. Softmax and layernorm are row kernels with
SIMD reductions. Transpose is either a vectorized row copy or a 32×32 tiled
copy. Reshape compiles to nothing.

The tile sizes can be autotuned. For each matmul shape, including its
epilogue, the tuner times up to 32 candidate tiles, re-times the best few
against the default (interleaved, to cancel out noise), and stores the winner
in `benchmarks/tuning.json`.

### 4. Memory

Intermediate tensors live in one preallocated workspace. A greedy best-fit
planner assigns offsets based on when each tensor is first written and last
read, so tensors whose lifetimes don't overlap share memory. Outputs are
written straight into the arrays returned to the caller.

![Workspace layout for the transformer block, unfused vs fused](docs/img/memory_plan.png)

Fusion removes the elementwise intermediates altogether, which is why the
fused workspace is 40% smaller.

## Results

Setup: one cloud VM with an Intel Xeon @ 2.80 GHz (4 vCPUs, AVX-512), GCC 13.3,
NumPy 2.5.3 (OpenBLAS), PyTorch 2.15 on CPU in eager mode. Each variant's output
is checked against NumPy before it is timed. The numbers are medians. Raw data
is in `docs/benchmarks_*.csv`. The 1-thread MLP and transformer numbers were
measured before layout folding was added; an A/B run with and without it
differed by less than 1.5% on every transformer size.

The five variants: tinyfuse with cleanup passes but no fusion (`unfused`), with
fusion (`fused`), with fusion and autotuned tiles (`fused+tuned`), the same
model in plain NumPy, and PyTorch.

### One thread

![MLP throughput, 1 thread](docs/bench_mlp_1t.png)
![Transformer block throughput, 1 thread](docs/bench_transformer_1t.png)

| ms per call, 1 thread | unfused | fused | fused+tuned | NumPy | PyTorch |
|---|---:|---:|---:|---:|---:|
| MLP 16 × 256→1024→256 | 0.337 | 0.317 | 0.282 | 0.398 | **0.261** |
| MLP 64 × 512→2048→512 | 3.114 | 3.001 | **2.965** | 3.698 | 3.105 |
| MLP 256 × 1024→4096→1024 | 38.95 | 37.45 | 37.46 | 44.62 | **35.34** |
| Transformer seq 64, d 256 | 1.381 | 1.333 | **1.267** | 2.098 | 1.626 |
| Transformer seq 128, d 512 | 9.074 | 8.893 | **8.458** | 13.97 | 11.78 |
| Transformer seq 256, d 768 | 39.20 | 37.48 | **36.22** | 51.14 | 38.13 |

tinyfuse beats NumPy on every row, and it beats PyTorch eager on all three
transformer sizes. PyTorch is faster on the smallest and the largest MLP.

Fusion is not where most of that speedup comes from. The profiler built into
the generated code (`CodegenOptions(profile=True)`) shows why:

![Per-kernel time, unfused vs fused](docs/img/kernel_profile.png)

These models spend about 90% of their time in matmul. Fusion deletes the
separate elementwise kernels, but their arithmetic still runs, now inside the
matmul epilogue. What's saved is the memory traffic and the extra kernels,
which comes to 2 to 6% end to end here. Most of the gap over NumPy comes from
the matmul kernel itself, and from running the whole model in one native call
instead of one Python-dispatched op at a time. A graph made of long elementwise
chains would benefit far more from fusion than these do.

### Four threads

With `CodegenOptions(parallel=True)` tinyfuse uses OpenMP, and every library
gets 4 threads. Here each variant is timed in its own block rather than
interleaved: when the variants alternated, the three thread pools (OpenMP,
OpenBLAS and PyTorch's) kept fighting each other for the 4 cores.

![MLP throughput, 4 threads](docs/bench_mlp_4t.png)
![Transformer block throughput, 4 threads](docs/bench_transformer_4t.png)

| ms per call, 4 threads | unfused | fused | fused+tuned | NumPy | PyTorch |
|---|---:|---:|---:|---:|---:|
| MLP 16 × 256→1024→256 | 0.163 | 0.151 | 0.129 | 0.196 | **0.101** |
| MLP 64 × 512→2048→512 | 1.150 | 1.159 | 1.271 | 1.404 | **0.810** |
| MLP 256 × 1024→4096→1024 | 14.93 | 15.09 | 14.12 | 17.30 | **10.84** |
| Transformer seq 64, d 256 | 0.823 | **0.766** | 0.786 | 1.329 | 0.862 |
| Transformer seq 128, d 512 | 3.810 | 3.811 | 4.345 | 8.669 | **2.865** |
| Transformer seq 256, d 768 | 18.42 | 17.20 | 15.44 | 26.65 | **11.02** |

At 4 threads tinyfuse beats NumPy on every row (1.2 to 2.3 times) and beats
PyTorch on the smallest transformer. PyTorch is ahead on the other five, by
1.3 to 1.6 times. That gap used to be up to 2.6 times. Most of the
improvement came from rewriting the parallel matmul so threads never wait on
each other (see [Things that went wrong](#things-that-went-wrong-along-the-way)).

### An elementwise-heavy graph

The models above are dominated by matmul, so fusion barely shows up in them.
To see what fusion does when it has real work, here is an activation block of
8 elementwise ops over a big tensor:

```python
def ew_chain(x, a, b):
    h = x * a + b
    return tf.gelu(h) * tf.sigmoid(x) + x * 0.5 - tf.tanh(h) * 0.1
```

Unfused, that is 8 kernels and 8 full passes over memory. Fused, it is one loop
with every temporary held in registers.

![Elementwise chain, 1 thread](docs/bench_elementwise_1t.png)
![Elementwise chain, 4 threads](docs/bench_elementwise_4t.png)

| ms per call | unfused | fused | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| 256 × 1024, 1 thread | 1.658 | **0.903** | 2.187 | 2.398 |
| 1024 × 1024, 1 thread | 7.869 | **3.422** | 15.48 | 10.69 |
| 4096 × 1024, 1 thread | 38.65 | **13.41** | 61.70 | 47.55 |
| 256 × 1024, 4 threads | 0.346 | **0.226** | 2.196 | 0.512 |
| 1024 × 1024, 4 threads | 1.806 | **0.913** | 11.71 | 6.408 |
| 4096 × 1024, 4 threads | 14.18 | **4.404** | 61.79 | 13.98 |

Here fusion is worth 1.8 to 2.9 times on one thread and 1.5 to 3.2 times on
four, and the fused kernel beats PyTorch eager by 2 to 7 times. The fused loop
runs at about 3.5 ns per element on one core, almost all of it spent in the
three transcendental functions (`exp` for GELU and sigmoid, plus `tanh`), so
this chain is compute-bound once fused. The unfused version is the one paying
for memory traffic.

## Things that went wrong along the way

These are written up because they took a while to track down, and the first
two had me drawing the wrong conclusions for a while.

**Misaligned buffers made matmul 1.4× slower, but only some of the time.**
NumPy only guarantees 16-byte alignment, and big arrays usually start 16 bytes
past a page boundary. Every 64-byte AVX-512 load from a misaligned buffer
straddles two cache lines. Whether a compiled model was fast or slow depended
on where the allocator happened to put its workspace. For a while it looked as
if fusion was making matmuls slower and the autotuner was picking bad tiles.
Now every buffer tinyfuse allocates is 64-byte aligned, and a test checks it.

**GELU was 10× slower on big activations.** GELU is computed as
`x / (1 + exp(-2y))`. With large inputs the `exp` underflows, and glibc's
vectorized `expf` falls back to a slow scalar path when that happens.
Clamping the exponent to ±80 fixed it; the effect on the result is below
1e-34.

**Interleaved multithreaded benchmarks measured the wrong thing.** The first
4-thread run gave medians 2 to 25 times larger than the minimums, because each
library's idle threads kept spinning and stole cores from whichever library ran
next. I switched to timing each variant in its own block.

**The first parallel matmul was slower than the serial one** on small
problems, because it started a new OpenMP team for every cache block. The
second version used one parallel region per kernel but still had all threads
pack B together and then wait at two barriers per block, and it scaled only
1.35x on 4 cores for a 64-row matmul. The current version has no barriers at
all. Each thread owns whole output blocks (a column block times a group of
row blocks), packs its own panels, and runs the full K loop. On 4 cores it
now scales 2.9x on that matmul and 3.4x on a 256×1024×4096 one.

**Single-loop elementwise kernels ignored `parallel=True`.** The generator only
parallelized the outer loop of a loop nest. A contiguous op like `gelu(h)`
collapses to a single loop, so it never got split across threads. That went
unnoticed until the elementwise benchmark showed the unfused version not
scaling at all from 1 to 4 threads. A test now checks for the pragma.

## Tests

Every op, the MLP and the transformer block are compared with NumPy in float64,
with `rtol = atol = 1e-4`.

| File | What it covers |
|---|---|
| `test_frontend.py` | tracing, operator overloading, trace-time shape errors |
| `test_ir.py` | shape inference, exact printer output, the verifier rejecting 10 kinds of broken IR |
| `test_pass_*.py` | each pass, layout folding included: exact IR before and after, same results through the interpreter, cases that must *not* fuse or fold |
| `test_codegen_ops.py` | every op through C at odd sizes and with every broadcast pattern; tiny tiles that force every edge path in the matmul; aliased and duplicated outputs; alignment; the profiler |
| `test_models.py` | MLP and transformer at 4 sizes × 3 pipelines, a PyTorch cross-check, OpenMP builds |
| `test_fuzz.py` | 40 random graphs, each compiled 4 ways, compared with the interpreter |
| `test_reductions.py` | sum/mean/max vs NumPy, reduction fusion IR, hand-written RMSNorm and layernorm (checked against the built-in kernel) |
| `test_memory.py` | live buffers never overlap, and reuse actually saves memory |
| `test_autotune.py` | the search space, the tuning database, precedence of explicit tiles, the CLI |

## Limitations

* Static shapes, float32 and inference only. A new input shape means
  re-tracing and recompiling (compiled code is cached by a hash of its source).
  There is no autodiff and no low-precision types.
* The op set is small. There is no concat, slicing, gather or convolution.
  Reductions, softmax and layernorm only work over the last axis, and batched
  matmul needs identical batch dimensions.
* Fusion is local. Layernorm and softmax are never fused with their
  neighbours, and there is nothing like flash attention. Transposes that feed a
  matmul are folded away, but a transpose of a matmul's *output* (the head
  merge after `att @ v`) is still materialized.
* The matmul is good but not BLAS-grade everywhere: no prefetching, no
  assembly, and weights are re-packed on every call even when they are
  constants. It is tuned for AVX-512; AVX2 machines run the same code with
  much less tuning behind it.
* Threading is basic. There is one fork/join per kernel rather than one per
  graph, no NUMA awareness, and the autotuner only tunes single-threaded. A
  compiled function is not safe to call from several threads at once, because
  its workspace is shared.
* The C is built with `-ffast-math`, which glibc needs before it will use its
  vectorized `expf`/`tanhf`. Results can differ from NumPy in the last few bits,
  and NaN or inf inputs give unspecified results.
* It assumes GCC on Linux with glibc. The compiled `.so` files are specific to
  the CPU that built them.
* All benchmarks come from one shared cloud VM, with run-to-run swings of up to
  30%. PyTorch was run in eager mode, not with `torch.compile`.

## Repo layout

| Path | Contents |
|---|---|
| `tinyfuse/frontend.py` | `Tensor`, operator overloading, `trace()` |
| `tinyfuse/ops.py` | op registry: shape rules plus NumPy semantics |
| `tinyfuse/ir.py` | graph, printer, verifier |
| `tinyfuse/passes/` | constant folding, CSE, DCE, matmul/elementwise/reduction fusion, layout folding, pass manager |
| `tinyfuse/codegen/` | C emitter and memory planner |
| `tinyfuse/runtime.py` | gcc invocation, `.so` cache, `ctypes` wrapper, aligned buffers, profiler |
| `tinyfuse/autotune.py` | tile search and the tuning database |
| `tinyfuse/models.py` | the MLP and transformer block used by tests and benchmarks |
| `benchmarks/` | benchmark runner (`--models mlp,transformer,elementwise`), chart script, tuning database |
| `docs/` | benchmark CSVs and charts, `make_figures.py`, images in `docs/img/` |
| `DESIGN.md` | design decisions and why |
