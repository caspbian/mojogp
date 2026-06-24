# Notebook Examples

These marimo notebooks are intended to be read in order. The sequence starts
with the smallest single-output ExactGP workflow, then expands into kernel
choice, route choice, multi-output structure, mixed inputs, uncertainty,
persistence, posterior sampling, and documented observation-noise routes.

The notebook titles are the authoritative user-facing labels. The numbered
filenames are stable example IDs; use the table below as the recommended reading
order.

## Recommended Order

| Step | File | Main concept | Dataset style |
|---|---|---|---|
| 1 | `01_hello_gp.py` | Smallest end-to-end `ExactGP` regression | 1D synthetic |
| 2 | `02_diabetes_ard.py` | ARD on a real tabular regression task | sklearn diabetes |
| 3 | `03_mauna_loa_co2.py` | Composite kernels for seasonal structure | CO2-style synthetic time series |
| 4 | `04_multi_output.py` | `MultiOutputLMCGP` with correlated tasks | 2-task synthetic |
| 5 | `05_scaling.py` | Choosing `materialized` vs `matrix_free` | medium synthetic tabular |
| 6 | `06_constant_mean_function.py` | Constant mean initialization and centering workflows | shifted 1D synthetic |
| 7 | `08_ard_vs_isotropic.py` | Controlled ARD vs isotropic comparison | synthetic feature-relevance |
| 8 | `09_discrete_variables.py` | Mixed continuous plus categorical inputs | synthetic mixed-input |
| 9 | `10_predictive_uncertainty_and_love_variance.py` | Predictive uncertainty, LOVE variance, and route metadata | 1D synthetic uncertainty |
| 10 | `11_fixed_observation_noise.py` | Fixed per-sample observation noise | 1D heteroskedastic synthetic |
| 11 | `12_grouped_observation_noise.py` | Fixed grouped observation noise | grouped synthetic |
| 12 | `14_input_dependent_observation_noise.py` | Learned input-dependent observation noise | 1D heteroskedastic synthetic |
| 13 | `15_categorical_ablation.py` | Discrete-kernel progression and shuffled-category controls | simulated categorical structures |
| 14 | `17_model_persistence_roundtrip.py` | Save/load prediction consistency | 1D synthetic persistence |
| 15 | `18_posterior_sampling_methods.py` | Diagonal and pathwise posterior sampling | 1D synthetic sampling |
