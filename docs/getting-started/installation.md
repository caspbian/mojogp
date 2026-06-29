# Installation

MojoGP's current public install path is a source build. MojoGP currently
supports Python 3.10 and 3.11.

```bash
git clone https://github.com/caspbian/mojogp.git
cd mojogp
pip install -e .
mojo build mojogp/kernels/jit/jit_engine_bindings.mojo \
  --emit shared-lib \
  -I mojogp/ \
  -o mojogp_jit_engine.so \
  --target-accelerator sm_89
```

After package publication, the base `mojogp` package installs the Python API
and shared runtime dependencies. Normal GPU use should install the accelerator
extra for the GPU target you will run on. `pip` cannot reliably choose that
extra from local hardware because installation often happens in CI, containers,
or login nodes that do not match the runtime GPU.

Installing the matching extra is normally enough. MojoGP detects the CUDA compute
capability at runtime and loads the matching installed accelerator package.
Set `GPU_TARGET` only when you need to force a specific target, such as in CI,
containers, login-node workflows, or environments with multiple installed
accelerator packages.

After package publication, choose the matching accelerator extra:

```bash
pip install "mojogp[sm80]"  # A100
pip install "mojogp[sm86]"  # A40 / RTX 30-series
pip install "mojogp[sm89]"  # L4 / RTX 40-series
pip install "mojogp[sm90]"  # H100 / H200
```

Underscored extra aliases such as `mojogp[sm_80]` and `mojogp[sm_90]` are also
accepted.

Additional candidate package targets are listed on the accelerator target page.

The base package and accelerator packages are version-locked. For example,
`mojogp==0.26.6.0` must be installed with `mojogp-cuda-sm89==0.26.6.0`. The
current release pins Mojo `0.25.7.0` and MAX `25.7.0` because generated Mojo
code is version-sensitive.

Minimum runtime dependencies are Python 3.10 or 3.11, NumPy, SymPy, tqdm,
Mojo, and MAX.

## Optional Extras

Install comparison and test dependencies only when you need them:

```bash
pip install -e ".[test]"
```

Install notebook dependencies when running the marimo notebooks:

```bash
pip install -e ".[notebooks]"
```

Install documentation build dependencies when building the docs locally:

```bash
pip install -e ".[docs]"
```

## Native Mojo Kernels

For source builds, MojoGP loads the repository-local `mojogp_jit_engine.so`
compiled by the `mojo build` command above. After package publication, MojoGP
can also load a prebuilt JIT engine from the matching accelerator package.

Set `GPU_TARGET` before running only if you need to force a specific accelerator
target:

```bash
GPU_TARGET=sm_80 python train.py
```

If `GPU_TARGET` is not set, MojoGP tries to detect the CUDA compute capability
and load the matching installed accelerator package. See
[Accelerator Targets](accelerator-targets.md) for the GPU target table,
source-build commands, and local docs preview commands.

## What Is Not Required

Training and prediction with MojoGP do not require PyTorch or GPyTorch. Those
packages are useful for reference comparisons and validation workflows only, and
they are installed by the optional `test` extra.

## Troubleshooting

### Missing Accelerator Package

If MojoGP reports that an accelerator package is missing, install the matching
extra for the detected or requested target after packages are published:

```bash
pip install --upgrade "mojogp[sm89]"
```

If MojoGP reports a version mismatch, upgrade the same extra so the base package
and accelerator package versions match exactly.

### Wrong GPU Target

If MojoGP selects the wrong GPU target, set `GPU_TARGET` for the current machine
before running Python.

### CUDA Not Visible

MojoGP is GPU-backed. Check the NVIDIA driver, CUDA visibility, and whether the
current environment can see the GPU before running large models.
