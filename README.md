# MojoGP

> **Active development:** MojoGP is pre-1.0 software. Expect sharp edges, API changes, incomplete routes.

MojoGP is a Python-first exact Gaussian Process regression library backed by JIT-compiled Mojo GPU kernels. It exists to make exact GP training practical without depending on Torch: the runtime package depends on NumPy, SymPy, tqdm, and Mojo/MAX.

## Why MojoGP

1. No torch dependency: NumPy, SymPy, tqdm, and Mojo/MAX only.
2. Materialized and matrix-free routes: dense materialized kernels and matrix-free kernel matvecs.
3. JIT-compiled GP models: models JIT-compile and kernels cached before training/prediction.
4. Multi-output GPs: ICM-style and LMC-style.
5. Discrete/categorical kernels: initial support for mixed continuous-categorical kernels.
6. Support for NVIDIA GPUs - support for AMD and Apple Silicon is on the roadmap.

## Features

Mixed means continuous plus categorical/discrete inputs.

| Feature | Single-output continuous | Single-output mixed | Multi-output ICM continuous | Multi-output ICM mixed | Multi-output LMC continuous | Multi-output LMC mixed |
|---|---|---|---|---|---|---|
| Materialized training | alpha | experimental | experimental | experimental | alpha | experimental |
| Matrix-free training | alpha | experimental | experimental | experimental | alpha | experimental |
| Mean-only prediction | alpha | experimental | experimental | experimental | alpha | experimental |
| Exact variance | alpha | experimental | experimental | experimental | alpha | experimental |
| LOVE variance | alpha | experimental | experimental | experimental | experimental | experimental |
| Heterogeneous latent kernels | n/a | n/a | n/a | n/a | alpha | experimental |
| Active dimensions | alpha | experimental | experimental | experimental | alpha | experimental |
| ARD lengthscales | alpha | experimental | experimental | in development | alpha | in development |
| Additive kernel composites | alpha | experimental | experimental | experimental | alpha | in development |
| Product kernel composites | alpha | experimental | experimental | experimental | alpha | experimental |
| Save / load | alpha | experimental | experimental | experimental | alpha | experimental |
| Learned homoskedastic noise | alpha | experimental | experimental | experimental | alpha | experimental |
| Fixed observation noise | alpha | in development | alpha | in development | alpha | in development |
| Learned heteroskedastic noise | alpha | in development | not started | not started | not started | not started |
| Grouped noise | alpha | in development | alpha | in development | unsupported | unsupported |
| Posterior sampling | alpha | experimental | experimental | experimental | alpha | experimental |

## Examples

See [`notebooks/examples/`](notebooks/examples/) for runnable examples covering:

- single-output GPs
- multi-output workflows
- predictive uncertainty
- categorical variables
- observation-noise variants
- posterior sampling
- model persistence

## Install

PyPI packages are in development. For now, install MojoGP from source.

Minimum runtime dependencies are Python 3.10 or 3.11, NumPy, SymPy, tqdm,
Mojo, and MAX. The current build has been tested with Mojo 0.25.7.0 and MAX
25.7.0.


## Build From Source

For now, build MojoGP from source. This workflow clones the source, installs the
Python package, and builds one JIT engine locally for the GPU target you choose:

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

Use the Mojo accelerator target that matches your GPU:


| GPU family | Target |
|---|---|
| T4 / RTX 20-series / Quadro RTX | `sm_75` |
| A100 / A30 | `sm_80` |
| A40 / A10 / A16 / A2 / RTX 30-series / RTX A-series | `sm_86` |
| L4 / L40 / L40S / RTX 40-series / RTX Ada | `sm_89` |
| GH200 / H100 / H200 | `sm_90` |
| B200 / GB200 | `sm_100` |
| RTX PRO Blackwell / GeForce RTX 50-series | `sm_120` |


## Hello World

```python
import numpy as np
from mojogp import RBF, SingleOutputGP

rng = np.random.default_rng(0)
X = np.linspace(-3, 3, 2000, dtype=np.float32).reshape(-1, 1)
y = (np.sin(2.0 * X[:, 0]) + 0.05 * rng.standard_normal(len(X))).astype(np.float32)

gp = SingleOutputGP(RBF())
gp.fit(X, y, max_iterations=50, method="matrix_free")

X_test = np.linspace(-4, 4, 128, dtype=np.float32).reshape(-1, 1)
mean, std = gp.predict(X_test, return_std=True, variance_method="love")
```

## References
Bonilla, E.V., Chai, K. and Williams, C. (2007). Multi-task Gaussian Process Prediction. [online] Neural Information Processing Systems. Available at: https://papers.nips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html.

Bruinsma, W.P., Perim, E., Tebbutt, W., Scott, H.J., Solin, A. and Turner, R.E. (2019). Scalable Exact Inference in Multi-Output Gaussian Processes. [online] arXiv.org. Available at: https://arxiv.org/abs/1911.06287 [Accessed 22 May 2026].

Charlier, B., Feydy, J., Glaunès, J.A., Collin, F.-D. and Durif, G. (2020). Kernel Operations on the GPU, with Autodiff, without Memory Overflows. [online] arXiv.org. Available at: https://arxiv.org/abs/2004.11127 [Accessed 22 May 2026].

Chen, T., Huber, C., Lin, E. and Zaid, H. (2026). Preconditioning without a preconditioner using randomized block Krylov subspace methods. ETNA - Electronic Transactions on Numerical Analysis, [online] 65, pp.63–92. doi:https://doi.org/10.1553/etna_vol65s63.

Dong, K., Eriksson, D., Nickisch, H., Bindel, D. and Wilson, A.G. (2017). Scalable Log Determinants for Gaussian Process Kernel Learning. [online] arXiv.org. Available at: https://arxiv.org/abs/1711.03481 [Accessed 22 May 2026].

Gardner, J.R., Pleiss, G., Bindel, D., Weinberger, K.Q. and Wilson, A.G. (2021). GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. arXiv:1809.11165 [cs, stat]. [online] Available at: https://arxiv.org/abs/1809.11165.

Godoy, W., Melnichenko, T., Valero-Lara, P., Elwasif, W., Fackler, P., Ferreira Da Silva, R., Teranishi, K. and Vetter, J. (2025). Mojo: MLIR-based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem. Proceedings of the SC ’25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, [online] pp.2114–2128. doi:https://doi.org/10.1145/3731599.3767573.

Harbrecht, H., Peters, M. and Schneider, R. (2012). On the low-rank approximation by the pivoted Cholesky decomposition. Applied numerical mathematics, 62(4), pp.428–440. doi:https://doi.org/10.1016/j.apnum.2011.10.001.

Perez, R.C., Veiga, D. and Garnier, J. (2025). A reproducible comparative study of categorical kernels for Gaussian process regression, with new clustering-based nested kernels. [online] arXiv.org. Available at: https://arxiv.org/abs/2510.01840 [Accessed 22 May 2026].

Peter, Wu, H. and Wu, C.Y. (2008). Gaussian Process Models for Computer Experiments With Qualitative and Quantitative Factors. 50(3), pp.383–396. doi:https://doi.org/10.1198/004017008000000262.

Pleiss, G., Gardner, J.R., Weinberger, K.Q. and Wilson, A.G. (2018). Constant-Time Predictive Distributions for Gaussian Processes. [online] arXiv.org. Available at: https://arxiv.org/abs/1803.06058 [Accessed 22 May 2026].

Rakitsch, B., Lippert, C., Borgwardt, K. and Stegle, O. (2026). It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals. Advances in Neural Information Processing Systems, [online] 26. Available at: https://proceedings.neurips.cc/paper/2013/hash/59c33016884a62116be975a9bb8257e3-Abstract.html [Accessed 22 May 2026].

Rasmussen, C.E. and Williams, C.K.I. (2008). Gaussian processes for machine learning. Cambridge, Mass. Mit Press.

Roustant, O., Padonou, E., Deville, Y., Clément, A., Perrin, G., Giorla, J. and Wynn, H. (2018). Group kernels for Gaussian process metamodels with categorical inputs. [online] arXiv.org. Available at: https://arxiv.org/abs/1802.02368 [Accessed 22 May 2026].

Saves, P., Diouane, Y., Bartoli, N., Lefebvre, T. and Morlier, J. (2023). A mixed-categorical correlation kernel for Gaussian process. Neurocomputing, [online] 550, p.126472. doi:https://doi.org/10.1016/j.neucom.2023.126472.

Shashanka Ubaru, Chen, J. and Saad, Y. (2017). Fast Estimation of $tr(f(A))$ via Stochastic Lanczos Quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), pp.1075–1099. doi:https://doi.org/10.1137/16m1104974.

Wilson, A.G. and Nickisch, H. (2026). Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP). [online] arXiv.org. Available at: https://arxiv.org/abs/1503.01057 [Accessed 22 May 2026].

Wilson, J.T., Borovitskiy, V., Terenin, A., Mostowsky, P. and Deisenroth, M.P. (2020). Pathwise Conditioning of Gaussian Processes. [online] arXiv.org. Available at: https://arxiv.org/abs/2011.04026 [Accessed 22 May 2026].

Zhou, Q., Peter Z.G. Qian and Zhou, S. (2011). A Simple Approach to Emulation for Computer Models With Qualitative and Quantitative Factors. Technometrics, 53(3), pp.266–273. doi:https://doi.org/10.1198/tech.2011.10025.

## License

MIT
