# tinyfuse design notes

This file records the decisions made while building tinyfuse, and why. It is
meant to be read alongside the code; each section names the module it covers.

## 1. Scope and non-goals

* **float32 only, static shapes only.** Every tensor has a fixed shape known at
  trace time. This keeps shape inference trivial, lets the C generator bake every
  loop bound into the source as a constant (so GCC can unroll and vectorise
  freely), and lets memory be planned once at compile time. Dynamic shapes would
  need symbolic shape inference plus runtime dispatch, which is out of scope.
* **Inference only.** No autodiff and no training ops.
* **CPU / C only.** One backend: portable GNU C11 compiled with the system `gcc`.

## 2. Frontend (`tinyfuse/frontend.py`)

* **Tracing, not parsing.** `trace(fn, *shapes)` calls `fn` with symbolic `Tensor`
  objects; every operation appends a node to the current graph. Python control
  flow is resolved during tracing, so the graph is straight-line code (the
  `torch.fx` / `jax.jit` model). We went this way because it supports arbitrary
  Python for free and the IR stays simple.
* Python scalars and NumPy arrays used in expressions become `const` nodes.
  `Tensor.__array_ufunc__ = None` makes `ndarray @ Tensor` and friends defer to
  `Tensor`'s reflected operators.
* Parameters with default values and no shape (for example `softmax(x, axis=-1)`)
  keep their default and are not graph inputs.
* Shape errors are raised **at trace time**, at the line that caused them,
  because nodes infer their shapes as soon as they are created.
* `gelu` is the **tanh approximation**, the same as
  `torch.nn.functional.gelu(x, approximate="tanh")`. NumPy has no `erf`, and the
  tanh form is what most transformer inference stacks use.
* `softmax` and `layernorm` only reduce over the **last axis**. That covers every
  transformer use; other axes can be reached with a transpose.

## 3. IR (`tinyfuse/ir.py`, `tinyfuse/ops.py`)

* **Single SSA dataflow graph.** Every `Node` produces exactly one tensor. A
  `Graph` holds `nodes` in topological order, plus `inputs` (in signature order)
  and `outputs`. There are no regions or blocks, because there is no control flow
  to represent.
* **An op registry is the single source of truth.** Each `OpDef` has a shape
  function and a **NumPy reference implementation**. The interpreter, the
  constant folder and the tests all use the NumPy implementation, so "what does
  this op mean" is answered in one place, and the C backend is checked against it.
* **Fused ops are ordinary IR ops.** `fused_elementwise` carries a `program`: a
  tuple of `(op, operands)` instructions whose operands are `("in", i)`,
  `("t", j)` (an earlier instruction), `("c", float)` (an inlined scalar
  constant) or `("acc",)`. `fused_matmul` carries an `epilogue` program in which
  `("acc",)` is the matmul result. Programs are plain hashable tuples, so CSE,
  printing, interpretation and codegen all handle them without special classes.
* **Matmul forms.** `[M,K]@[K,N]`, `[*,M,K]@[K,N]` and `[*,M,K]@[*,K,N]` with
  identical batch dims. There is no batch broadcasting beyond "B is 2-D", which
  is enough for MLPs and multi-head attention.
* **Pretty printer.** Values are renumbered `%0, %1, …` in node order, and inputs
  print under their parameter names. That makes the output deterministic, so
  pass tests can compare exact before/after text.
* **Verifier.** It checks that ids are unique, operands are defined before use,
  ops are known with the right arity, stored shapes match re-inferred shapes,
  constants are float32 arrays of the right shape, fused programs are well formed
  (no forward references, operands in range, `acc` only in epilogues), inputs
  are listed with unique names, and outputs exist. The pass manager runs it after
  every pass by default.

## 4. Passes (`tinyfuse/passes/`)

Each pass is `fn(graph) -> changed: bool` and mutates the graph in place.
Pipelines (`passes/manager.py`):

| pipeline | passes |
|---|---|
| `none` | nothing: one kernel per traced op |
| `unfused` | constant folding → CSE → DCE |
| `fused` | `unfused` + matmul-epilogue fusion → elementwise fusion → DCE |

`optimize()` runs the pipeline up to twice, stopping once a full round changes
nothing.

* **Constant folding.** Any node whose operands are all constants is evaluated
  with its NumPy implementation, up to 16M elements. The pass also applies a few
  shape-preserving algebraic identities: `x+0`, `x-0`, `x*1`, `x/1`, `1*x`,
  `0+x`, `--x`, no-op reshape/transpose, and composition of
  `reshape∘reshape` and `transpose∘transpose`. An identity is skipped when it
  would change the result shape through broadcasting, for example
  `x[3] * ones[2,3]`. Dead constants are left behind for DCE.
* **CSE.** Nodes are keyed by `(op, operand ids, frozen attrs, shape)`. Operands
  of commutative ops (`add`, `mul`, `maximum`) are sorted, so `a+b ≡ b+a`.
  Constants are keyed by a SHA-1 of their bytes, so equal-valued constants merge.
  Graph inputs never merge.
* **DCE.** Removes everything not reachable from the outputs. Graph inputs are
  always kept, because they are the call signature.
* **Matmul epilogue fusion** (`matmul_fusion.py`). This generalises the
  matmul + bias + activation fusion the task asked for. Starting at a matmul, it
  walks forward while the chain tail is not a graph output, has exactly one user,
  and that user is elementwise with the matmul's output shape. The whole chain
  becomes the epilogue. So `relu(x@W+b)`, `x + gelu(h@W1+b1)@W2 + b2` (bias and
  residual) and attention's `(q@k)*scale` all fuse. Epilogue operands can be any
  shape that broadcasts to the output; the kernel indexes them with broadcast
  strides.
* **Elementwise fusion** (`fusion.py`). The graph is walked in reverse
  topological order. An elementwise node joins a group only if (a) it is not a
  graph output, (b) **all** its users are already in that same group, and (c) it
  has the group root's exact shape. Rule (b) means every edge that leaves a group
  starts at the group's root, which makes cycles impossible: a path out of the
  group and back in would have to leave through the root and re-enter an
  ancestor of the root. Rule (c) means a small broadcast producer is never
  recomputed once per element of a larger output. Multi-use values whose users
  all sit in one group (`relu(x)*exp(x)`) still fuse. Scalar constants become
  literals in the program.
* **Order.** Matmul fusion runs first so that bias and activation end up in the
  matmul epilogue rather than in a separate fused-elementwise kernel. The matmul
  pass also absorbs an already-fused `fused_elementwise` consumer, so the order is
  a preference, not a correctness requirement.

## 5. C code generation (`tinyfuse/codegen/`)

* **One translation unit per graph.** Each non-alias node becomes a `static`
  kernel function whose shapes are compile-time constants. The entry point is
  `void tf_run(float* const* bufs)` with `bufs = [inputs…, outputs…, constants…,
  workspace]`. One `ctypes` call runs the whole graph, so Python overhead is paid
  once per call, not once per kernel.
* **Elementwise kernels** (primitive and fused). Each operand gets broadcast
  strides against the output shape. Adjacent dimensions are merged when they are
  contiguous for *every* operand, and size-1 dims are dropped. The result is a
  short loop nest whose innermost loop has stride 0 or 1 for every operand and
  carries `#pragma omp simd`. The program is emitted as a sequence of
  `const float tN = …;` statements, so temporaries live in registers.
* **Matmul.** This is a GotoBLAS-style kernel:
  * loops `j0 (NB) → k0 (KB) → pack B panel → i0 (MB) → pack A panel → micro-tiles`;
  * B is packed into zero-padded `[kb][NR]` panels and A into `[kb][MR]` panels,
    so the micro-kernel sees unit-stride data and every micro-tile is full size.
    Edge tiles are handled once, at store time, through a small temporary;
  * the micro-kernel keeps an `MR × NV` block of SIMD vectors in registers, using
    GCC vector extensions (`float __attribute__((vector_size(VW*4)))`).
  * **Why vector extensions and not plain loops.** Our first version used a
    plain-C accumulator array, `float acc[MR][NR]`, and relied on the
    autovectoriser. GCC vectorised it but kept the accumulators on the stack:
    **1.8 GFLOP/s**. Vector types loaded through `memcpy` reached 20 GFLOP/s but
    still spilled. Direct vector loads and stores keep all accumulators in
    registers and reach **~115–135 GFLOP/s** single-threaded on AVX-512, on par
    with OpenBLAS. Vector extensions are portable C: on AVX2 or SSE machines GCC
    splits each vector into narrower ones. `VW` comes from probing
    `gcc -march=native` for `__AVX512F__` / `__AVX__`.
  * **Epilogue placement.** After the last K block of an `MB×NB` cache block, the
    epilogue runs over that block while it is still in L2, in a `#pragma omp simd`
    loop. There is no extra pass over C in DRAM and no intermediate buffer. We
    also tried running it per micro-tile, in L1: it measured the same, and the
    block form keeps the micro-kernel loop body minimal.
  * **64-byte alignment matters (a lot).** NumPy only guarantees 16-byte
    alignment, and large blocks usually start 16 bytes past a page boundary. With
    a misaligned workspace or output, every 64-byte AVX-512 load and store in the
    micro-kernel splits a cache line. That made matmuls **~1.4× slower**,
    *bimodally*, depending on where the allocator happened to put a module's
    buffers. It first looked like "fusion makes matmul slower" and "the autotuner
    picks bad tiles". The workspace, constants and outputs are now allocated
    64-byte aligned (`runtime.aligned_empty`), and a test checks this.
  * **Default tile:** `mb=48, nb=512, kb=256, mr=6, nv=4` for AVX-512, which uses
    24 of the 32 zmm registers for accumulators. In measurements it was the most
    consistent across the benchmark shapes. The default AVX2 tile is `6×2`
    vectors. Tiles are clamped to the problem size.
* **Softmax / LayerNorm.** These are row-wise kernels with `omp simd reduction`
  loops. Softmax subtracts the row max, which keeps it stable for large inputs
  (tested). The max starts at `-FLT_MAX`, not `-INFINITY`; see the fast-math note
  below.
* **Transpose.** Dimensions are merged when they stay adjacent after the
  permutation. If the innermost output dim is unit-stride in the input (like the
  `(1,0,2)` head split in attention), it is a vectorised row copy. Otherwise the
  two dims involved are tiled 32×32.
* **Reshape is free.** It aliases its operand's storage. If an output is an
  input, a constant, an alias of one of those, or a duplicate of another output,
  it gets a `memcpy`, so the caller never receives aliased memory.
* **GELU lowering.** `0.5·x·(1+tanh(y)) == x·sigmoid(2y)`, so we emit
  `x / (1 + expf(-2y))`, with the exponent clamped to `[-80, 80]`. The upper
  clamp prevents overflow. The lower clamp matters for speed: libmvec's `expf`
  falls back to a slow scalar path when its result underflows, which made GELU
  on large activations ~10× slower (≈10 ns vs ≈1 ns per element). `sigmoid` is
  clamped the same way. The error is below 1e-34.
  With libmvec, vectorised `expf` costs about half as much as vectorised `tanhf`
  (≈0.8 vs ≈2.0 ns/element in a micro-benchmark).

### Compiler flags

`-O3 -march=native -ffast-math -fopenmp-simd -std=gnu11 -fPIC -shared`, plus
`-fopenmp` when `parallel=True`.

* **`-ffast-math` is required for vectorised `expf`/`tanhf`.** glibc only
  advertises its libmvec SIMD variants under `__FAST_MATH__`. We checked that
  adding `-fno-finite-math-only` turns them off again. Fast-math also allows
  reassociation, which the reductions need.
* **Consequences, handled:** kernels never depend on inf or NaN semantics. The
  softmax max starts at `-FLT_MAX` and the GELU exponent is clamped. Results can
  differ from NumPy in the last bits because of reassociation and FMA
  contraction. The tests use `rtol = atol = 1e-4` against a float64 NumPy
  reference.
* **Not handled:** NaN or inf *inputs* give unspecified results.

## 6. Memory planning (`codegen/memory.py`)

Each value maps to a storage: input, const, output, or a workspace slice.
Workspace slices are assigned by greedy best-fit over live ranges, with
coalescing of freed blocks. A slice is freed only *after* the kernel that last
reads it has had its own output allocated. So a kernel's output never overlaps
its inputs, which keeps `restrict` on kernel arguments valid. Graph outputs are
written straight into the caller's arrays when possible. On the 128×256
transformer block the fused graph needs a smaller workspace than the unfused one
(a test asserts this).

Matmul packing buffers come from a scratch region at the end of the workspace,
sized for the largest matmul. **So a compiled function is not reentrant:** do
not call one compiled object from several threads at once. Compile separate
objects instead.

## 7. Runtime (`tinyfuse/runtime.py`)

* The generated source is written to a cache directory (`$TINYFUSE_CACHE`, or
  `$TMPDIR/tinyfuse_cache`) under a SHA-256 of compiler, flags and source.
  Compiling into a temporary file followed by `os.replace` makes concurrent
  compiles safe. Recompiling an identical graph costs nothing.
* `ctypes.CDLL` loads the shared object. The constant arrays and the workspace
  are owned by the `CompiledModule`. On each call the input pointers are written
  into a preallocated pointer array; inputs are converted to C-contiguous float32
  if needed, and outputs are freshly allocated unless `out=` is passed.
* `CodegenOptions(profile=True)` wraps every kernel call in `clock_gettime`.
  `module.profile()` returns cumulative seconds per kernel.

## 8. Testing strategy (`tests/`)

* **Reference = NumPy in float64.** Inputs are float32 and the reference is
  computed in float64, so any differences come from tinyfuse, not from NumPy's
  own float32 rounding. Tolerance is `rtol=atol=1e-4` throughout.
* **Every op through C**, with odd sizes (1, 7, 17, 33…), every broadcast pattern,
  2-D and batched matmul, and all three pipelines.
* **Tiny tiles.** Every matmul test also runs with `mb=6, nb=VW, kb=5, mr=3,
  nv=1`. This forces many K blocks (the epilogue must run only after the last
  one), partial micro-tiles in both dimensions, and multiple cache blocks, so
  edge paths are exercised even at small sizes.
* **Passes.** Each pass has tests that compare the exact before/after IR text,
  plus a semantic-equivalence check through the interpreter. There are negative
  tests too: no fusion across outputs, multi-use matmuls, or shape changes, and
  no CSE of non-commutative swaps.
* **Models.** 2-layer MLP and transformer block at four sizes each, on all
  pipelines, plus a PyTorch cross-check of the transformer block.
* **Differential fuzzing.** 40 random graphs mixing elementwise, matmul,
  softmax, layernorm, transpose and reshape, each compiled three ways and
  compared with the interpreter.
* **Memory planner.** Tests check that no two simultaneously-live slices overlap,
  and that reuse actually shrinks the workspace.

## 9. Benchmarks (`benchmarks/`)

* Thread counts are pinned through environment variables *before* NumPy and
  torch are imported, and with `torch.set_num_threads`. tinyfuse uses OpenMP only
  when `--threads > 1`.
* Every variant's output is checked against NumPy before timing starts.
* **Interleaved timing.** Each round calls every variant once, for a time budget
  of at least 10 rounds. The benchmark VM is noisy (±30% swings within minutes),
  and interleaving spreads that drift over all variants equally. We report the
  median (used in the charts) and the minimum.
* PyTorch runs eagerly under `torch.no_grad()` with the same tanh GELU.
* The NumPy MLP and transformer are the same functions the tests use as
  references.

## 10. Autotuner (`tinyfuse/autotune.py`)

* For each `(batch, M, N, K)` it tries up to 32 tiles sampled from a pruned space.
  Micro-tiles `mr ∈ {4,6,8,12}`, `nv ∈ {1..4}` must fit the register file
  (`mr·nv + nv + 1 ≤ #regs`); `mb = mr·{4,8,16,32}`, `nb ∈ {128…1024}`,
  `kb ∈ {64…512}`; candidates are de-duplicated after clamping. The default tile
  is always included.
* Each candidate is compiled as a standalone matmul, checked for correctness, and
  timed by the minimum of several runs. The best is stored in a JSON
  `TuningDB` keyed by CPU model hash, SIMD width and shape.
  `benchmarks/tuning.json` is the database produced on the benchmark machine.
* `compile(..., tuning=db)` applies the stored tiles to every matmul and
  fused_matmul of that shape. Tiles set explicitly in
  `CodegenOptions.tiles` take precedence.
* **Known simplification:** tuning times the plain matmul, not the fused one, and
  it is single-threaded.
