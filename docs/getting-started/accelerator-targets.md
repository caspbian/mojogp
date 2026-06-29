# Accelerator Targets

MojoGP loads a prebuilt native engine for a GPU compute capability target. If you
know the GPU you will run on, set `GPU_TARGET` before running Python so the
loader selects the intended accelerator package.

```bash
GPU_TARGET=sm_80 python train.py
```

If `GPU_TARGET` is not set, MojoGP tries to detect the CUDA compute capability
and load the matching installed accelerator package.

## Current Tested Targets

The current local evidence is limited to these targets:

| Target | GPUs covered by that target | Status |
|---|---|---|
| `sm_89` | L4 / RTX 40-series consumer cards | target tested locally on RTX 4050-class hardware |

Other targets in the reference table below are candidate package targets, not
public support claims, until MojoGP has a matching package install smoke and a
minimal fit/predict smoke on hardware with that compute capability.

## NVIDIA Target Reference

Build by compute capability, not by PCIe/SXM form factor. For example, A100 PCIe
and A100 SXM both use `sm_80`.

| GPU family | Compute capability | Target | Planned package extra | Runtime example | Source-build target |
|---|---:|---|---|---|---|
| GTX 10-series consumer cards | 6.1 | `sm_61` | not available in Mojo 0.25.7.0 build check | n/a | build failed in current toolchain |
| V100 | 7.0 | `sm_70` | not available in Mojo 0.25.7.0 build check | n/a | build failed in current toolchain |
| T4 / RTX 20-series consumer cards | 7.5 | `sm_75` | `mojogp[sm75]` | `GPU_TARGET=sm_75 python train.py` | `--target-accelerator sm_75` |
| A100 / A30 | 8.0 | `sm_80` | `mojogp[sm80]` | `GPU_TARGET=sm_80 python train.py` | `--target-accelerator sm_80` |
| A40 / A10 / A16 / A2 | 8.6 | `sm_86` | `mojogp[sm86]` | `GPU_TARGET=sm_86 python train.py` | `--target-accelerator sm_86` |
| RTX 30-series / RTX A-series | 8.6 | `sm_86` | `mojogp[sm86]` | `GPU_TARGET=sm_86 python train.py` | `--target-accelerator sm_86` |
| L4 / L40 / L40S | 8.9 | `sm_89` | `mojogp[sm89]` | `GPU_TARGET=sm_89 python train.py` | `--target-accelerator sm_89` |
| RTX 40-series / RTX Ada | 8.9 | `sm_89` | `mojogp[sm89]` | `GPU_TARGET=sm_89 python train.py` | `--target-accelerator sm_89` |
| GH200 / H100 / H200 | 9.0 | `sm_90` | `mojogp[sm90]` | `GPU_TARGET=sm_90 python train.py` | `--target-accelerator sm_90` |
| B200 / GB200 | 10.0 | `sm_100` | `mojogp[sm100]` | `GPU_TARGET=sm_100 python train.py` | `--target-accelerator sm_100` |
| RTX PRO Blackwell / GeForce RTX 50-series | 12.0 | `sm_120` | `mojogp[sm120]` | `GPU_TARGET=sm_120 python train.py` | `--target-accelerator sm_120` |

Check the local Mojo toolchain before relying on a target:

```bash
mojo build --print-supported-accelerators
```

Only targets that are documented in the release notes and covered by matching
hardware smoke tests should be treated as release-supported. Newer GPU families
and additional Blackwell compute capabilities should remain unclaimed until the
Mojo toolchain lists the target and MojoGP has passing accelerator smoke evidence
for that target.

## Installed Package Workflow

After package publication, no repository-local build step is required. Install
the matching accelerator extra, then set the target when launching the program if
you want to force a specific route:

```bash
pip install "mojogp[sm89]"
GPU_TARGET=sm_89 python train.py
```

The underscored extra alias will also work, for example
`pip install "mojogp[sm_89]"`.

Accelerator package versions must exactly match the base package version. For
release `0.26.6.0`, the package set is:

```bash
mojogp==0.26.6.0
mojogp-cuda-sm75==0.26.6.0
mojogp-cuda-sm80==0.26.6.0
mojogp-cuda-sm86==0.26.6.0
mojogp-cuda-sm89==0.26.6.0
mojogp-cuda-sm90==0.26.6.0
mojogp-cuda-sm100==0.26.6.0
mojogp-cuda-sm120==0.26.6.0
```

MojoGP raises a clear reinstall command if the installed accelerator package is
missing or does not match the base package version.

## Source Build Workflow

When building MojoGP from a checkout, pass the target you need to `mojo build`:

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

Then run a small import or fit/predict smoke on hardware with the matching GPU.
Do not reuse a shared library compiled for one target on a different compute
capability.

## Local Documentation Review

Before publishing docs changes, build and review the site locally:

```bash
pip install -e ".[docs]"
mkdocs build --strict -f docs/mkdocs.yml
mkdocs serve -f docs/mkdocs.yml
```

Open the local MkDocs URL printed by `mkdocs serve`, review the accelerator page,
and publish only after the strict build passes.
