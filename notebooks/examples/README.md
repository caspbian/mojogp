# Notebook Examples

These marimo notebooks are intended to be read in order. The sequence starts
with the smallest single-output ExactGP workflow, then expands into kernel
choice, route choice, multi-output structure, mixed inputs, uncertainty,
persistence, posterior sampling, and documented observation-noise routes.

The notebook titles are the authoritative user-facing labels. The numbered
filenames are stable example IDs; use the table below as the recommended reading
order.

## Recommended Order

| Step | Source | Notebook | HTML | Main concept | Dataset style |
|---|---|---|---|---|---|
| 1 | [`01_hello_gp.py`](01_hello_gp.py) | [`ipynb`](__marimo__/ipynb/01_hello_gp.ipynb) | [`html`](__marimo__/html/01_hello_gp.html) | Smallest end-to-end `ExactGP` regression | 1D synthetic |
| 2 | [`02_diabetes_ard.py`](02_diabetes_ard.py) | [`ipynb`](__marimo__/ipynb/02_diabetes_ard.ipynb) | [`html`](__marimo__/html/02_diabetes_ard.html) | ARD on a real tabular regression task | sklearn diabetes |
| 3 | [`03_mauna_loa_co2.py`](03_mauna_loa_co2.py) | [`ipynb`](__marimo__/ipynb/03_mauna_loa_co2.ipynb) | [`html`](__marimo__/html/03_mauna_loa_co2.html) | Composite kernels for seasonal structure | CO2-style synthetic time series |
| 4 | [`04_multi_output.py`](04_multi_output.py) | [`ipynb`](__marimo__/ipynb/04_multi_output.ipynb) | [`html`](__marimo__/html/04_multi_output.html) | `MultiOutputLMCGP` with correlated tasks | 2-task synthetic |
| 5 | [`05_scaling.py`](05_scaling.py) | [`ipynb`](__marimo__/ipynb/05_scaling.ipynb) | [`html`](__marimo__/html/05_scaling.html) | Choosing `materialized` vs `matrix_free` | medium synthetic tabular |
| 6 | [`06_constant_mean_function.py`](06_constant_mean_function.py) | [`ipynb`](__marimo__/ipynb/06_constant_mean_function.ipynb) | [`html`](__marimo__/html/06_constant_mean_function.html) | Constant mean initialization and centering workflows | shifted 1D synthetic |
| 7 | [`08_ard_vs_isotropic.py`](08_ard_vs_isotropic.py) | [`ipynb`](__marimo__/ipynb/08_ard_vs_isotropic.ipynb) | [`html`](__marimo__/html/08_ard_vs_isotropic.html) | Controlled ARD vs isotropic comparison | synthetic feature-relevance |
| 8 | [`09_discrete_variables.py`](09_discrete_variables.py) | [`ipynb`](__marimo__/ipynb/09_discrete_variables.ipynb) | [`html`](__marimo__/html/09_discrete_variables.html) | Mixed continuous plus categorical inputs | synthetic mixed-input |
| 9 | [`10_predictive_uncertainty_and_love_variance.py`](10_predictive_uncertainty_and_love_variance.py) | [`ipynb`](__marimo__/ipynb/10_predictive_uncertainty_and_love_variance.ipynb) | [`html`](__marimo__/html/10_predictive_uncertainty_and_love_variance.html) | Predictive uncertainty, LOVE variance, and route metadata | 1D synthetic uncertainty |
| 10 | [`11_fixed_observation_noise.py`](11_fixed_observation_noise.py) | [`ipynb`](__marimo__/ipynb/11_fixed_observation_noise.ipynb) | [`html`](__marimo__/html/11_fixed_observation_noise.html) | Fixed per-sample observation noise | 1D heteroskedastic synthetic |
| 11 | [`12_grouped_observation_noise.py`](12_grouped_observation_noise.py) | [`ipynb`](__marimo__/ipynb/12_grouped_observation_noise.ipynb) | [`html`](__marimo__/html/12_grouped_observation_noise.html) | Fixed grouped observation noise | grouped synthetic |
| 12 | [`14_input_dependent_observation_noise.py`](14_input_dependent_observation_noise.py) | [`ipynb`](__marimo__/ipynb/14_input_dependent_observation_noise.ipynb) | [`html`](__marimo__/html/14_input_dependent_observation_noise.html) | Learned input-dependent observation noise | 1D heteroskedastic synthetic |
| 13 | [`15_categorical_ablation.py`](15_categorical_ablation.py) | [`ipynb`](__marimo__/ipynb/15_categorical_ablation.ipynb) | [`html`](__marimo__/html/15_categorical_ablation.html) | Discrete-kernel progression and shuffled-category controls | simulated categorical structures |
| 14 | [`17_model_persistence_roundtrip.py`](17_model_persistence_roundtrip.py) | [`ipynb`](__marimo__/ipynb/17_model_persistence_roundtrip.ipynb) | [`html`](__marimo__/html/17_model_persistence_roundtrip.html) | Save/load prediction consistency | 1D synthetic persistence |
| 15 | [`18_posterior_sampling_methods.py`](18_posterior_sampling_methods.py) | [`ipynb`](__marimo__/ipynb/18_posterior_sampling_methods.ipynb) | [`html`](__marimo__/html/18_posterior_sampling_methods.html) | Diagonal and pathwise posterior sampling | 1D synthetic sampling |

Rendered notebooks and HTML files are generated artifacts with executed outputs
for previewing figures and result tables. The marimo `.py` files are the source
of truth.
