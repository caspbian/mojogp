import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Pathwise Posterior Sampling

        This notebook uses **pathwise sampling only** and checks the core sampling
        contract against the GP predictive distribution.

        For a fitted ExactGP, the latent posterior at test inputs is

        \[
        f_* \mid y \sim \mathcal{N}(\mu_*, \Sigma_*).
        \]

        The cumulative diagnostic below draws more posterior functions over time.
        It compares finite-RFF pathwise sample moments against the exact latent
        GP posterior moments. The pathwise samples are latent-function draws, so
        this notebook subtracts the learned observation-noise variance from the
        backend predictive variance before drawing the exact comparison band.
        """
    )
    return (mo,)


@app.cell
def _():
    import time
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np

    from mojogp import RBF, SingleOutputGP

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(18)
    X_train = np.concatenate(
        [
            np.linspace(-3.5, -1.15, 34, dtype=np.float32),
            np.linspace(1.15, 3.5, 34, dtype=np.float32),
        ]
    ).reshape(-1, 1)
    X_test = np.linspace(-4.0, 4.0, 96, dtype=np.float32).reshape(-1, 1)
    y_true = (np.sin(1.3 * X_test[:, 0]) + 0.2 * np.cos(2.1 * X_test[:, 0])).astype(
        np.float32
    )
    y_train = (
        np.sin(1.3 * X_train[:, 0])
        + 0.2 * np.cos(2.1 * X_train[:, 0])
        + 0.1 * rng.standard_normal(len(X_train))
    ).astype(np.float32)
    return RBF, SingleOutputGP, X_test, X_train, np, plt, time, y_train, y_true


@app.cell
def _(RBF, SingleOutputGP, X_test, X_train, np, time, y_train):
    gp = SingleOutputGP(RBF(lengthscale=0.9, outputscale=1.0))
    train_result = gp.fit(
        X_train,
        y_train,
        max_iterations=20,
        learning_rate=0.035,
        method="materialized",
        verbose=False,
        progress=True,
    )

    predictive = gp.predict(X_test, variance_method="exact", progress=True)
    n_samples = 256
    n_rff_features = 2048

    _bulk_start = time.perf_counter()
    pathwise_samples = gp.sample_posterior(
        X_test,
        n_samples=n_samples,
        method="pathwise",
        n_rff_features=n_rff_features,
        rng=np.random.default_rng(8128),
    )
    bulk_sample_seconds = time.perf_counter() - _bulk_start
    bulk_amortized_ms = 1000.0 * bulk_sample_seconds / n_samples
    pathwise_info = dict(gp.backend_sample_info or {})

    _single_start = time.perf_counter()
    _single_sample = gp.sample_posterior(
        X_test,
        n_samples=1,
        method="pathwise",
        n_rff_features=n_rff_features,
        rng=np.random.default_rng(9128),
    )
    standalone_sample_ms = 1000.0 * (time.perf_counter() - _single_start)
    return (
        bulk_amortized_ms,
        bulk_sample_seconds,
        n_rff_features,
        n_samples,
        pathwise_info,
        pathwise_samples,
        predictive,
        standalone_sample_ms,
        train_result,
    )


@app.cell
def _(n_samples, np, pathwise_samples, predictive, train_result):
    predicted_variance = np.maximum(
        predictive.variance - float(train_result.noise),
        1e-12,
    )
    predicted_std = np.sqrt(predicted_variance)

    cumulative_sum = np.cumsum(pathwise_samples, axis=0)
    cumulative_sum_sq = np.cumsum(pathwise_samples**2, axis=0)
    early_frames = np.arange(2, min(32, n_samples) + 1, 2, dtype=int)
    later_frames = np.round(
        np.geomspace(min(40, n_samples), n_samples, 24)
    ).astype(int)
    frame_counts = np.unique(np.concatenate([early_frames, later_frames]))
    frame_counts[-1] = n_samples

    mean_rmse_trace = []
    variance_rel_rmse_trace = []
    kl_trace = []
    for _k in frame_counts:
        _sample_sum = cumulative_sum[_k - 1]
        _sample_sum_sq = cumulative_sum_sq[_k - 1]
        _sample_mean = _sample_sum / _k
        _sample_variance = np.maximum(
            (_sample_sum_sq - (_sample_sum**2) / _k) / max(_k - 1, 1),
            1e-12,
        )
        _mean_error = _sample_mean - predictive.mean
        _variance_error = _sample_variance - predicted_variance
        _pointwise_kl = 0.5 * (
            np.log(predicted_variance / _sample_variance)
            + (_sample_variance + _mean_error**2) / predicted_variance
            - 1.0
        )
        mean_rmse_trace.append(float(np.sqrt(np.mean(_mean_error**2))))
        variance_rel_rmse_trace.append(
            float(np.sqrt(np.mean(_variance_error**2)) / (np.mean(predicted_variance) + 1e-12))
        )
        kl_trace.append(float(np.mean(_pointwise_kl)))

    empirical_mean = cumulative_sum[-1] / n_samples
    empirical_variance = np.maximum(
        (cumulative_sum_sq[-1] - (cumulative_sum[-1] ** 2) / n_samples) / (n_samples - 1),
        1e-12,
    )
    mean_error = empirical_mean - predictive.mean
    variance_error = empirical_variance - predicted_variance
    mean_mc_se = predicted_std / np.sqrt(n_samples)
    variance_mc_se = predicted_variance * np.sqrt(2.0 / max(n_samples - 1, 1))
    mean_z = np.abs(mean_error) / np.maximum(mean_mc_se, 1e-12)
    variance_z = np.abs(variance_error) / np.maximum(variance_mc_se, 1e-12)

    mean_rmse = float(np.sqrt(np.mean(mean_error**2)))
    variance_rel_rmse = float(
        np.sqrt(np.mean(variance_error**2)) / (np.mean(predicted_variance) + 1e-12)
    )
    mean_within_3se = float(np.mean(mean_z <= 3.0))
    variance_within_3se = float(np.mean(variance_z <= 3.0))
    moment_check_pass = bool(mean_within_3se >= 0.85 and variance_within_3se >= 0.70)
    return (
        cumulative_sum,
        cumulative_sum_sq,
        frame_counts,
        kl_trace,
        mean_rmse,
        mean_rmse_trace,
        mean_within_3se,
        moment_check_pass,
        predicted_variance,
        variance_rel_rmse,
        variance_rel_rmse_trace,
        variance_within_3se,
    )


@app.cell
def _(
    X_test,
    X_train,
    cumulative_sum,
    cumulative_sum_sq,
    frame_counts,
    kl_trace,
    mean_rmse_trace,
    mo,
    np,
    pathwise_samples,
    plt,
    predicted_variance,
    predictive,
    variance_rel_rmse_trace,
    y_train,
    y_true,
):
    from matplotlib.animation import FuncAnimation

    _frame_stats = []
    for _k in frame_counts:
        _sample_sum = cumulative_sum[_k - 1]
        _sample_sum_sq = cumulative_sum_sq[_k - 1]
        _sample_mean = _sample_sum / _k
        _sample_variance = np.maximum(
            (_sample_sum_sq - (_sample_sum**2) / _k) / max(_k - 1, 1),
            1e-12,
        )
        _pointwise_kl = 0.5 * (
            np.log(predicted_variance / _sample_variance)
            + (_sample_variance + (_sample_mean - predictive.mean) ** 2) / predicted_variance
            - 1.0
        )
        _frame_stats.append((_sample_mean, _sample_variance, _pointwise_kl))

    _fig = plt.figure(figsize=(9.6, 8.8), dpi=90)
    _grid = _fig.add_gridspec(
        3,
        2,
        height_ratios=[2.25, 1.1, 1.25],
        hspace=0.64,
        wspace=0.36,
    )
    _sample_ax = _fig.add_subplot(_grid[0, :])
    _variance_ax = _fig.add_subplot(_grid[1, 0])
    _kl_ax = _fig.add_subplot(_grid[1, 1])
    _trace_ax = _fig.add_subplot(_grid[2, :])
    _x = X_test[:, 0]

    _sample_ax.scatter(X_train[:, 0], y_train, s=10, alpha=0.28, label="train")
    _sample_ax.plot(_x, y_true, "k--", alpha=0.75, label="truth")
    _sample_ax.plot(_x, predictive.mean, color="tab:green", linewidth=2, label="GP exact mean")
    _latent_std = np.sqrt(predicted_variance)
    _sample_ax.plot(
        _x,
        predictive.mean - 2 * _latent_std,
        color="black",
        linewidth=1.3,
        alpha=0.75,
        linestyle="--",
        label="exact latent 95% bounds",
    )
    _sample_ax.plot(
        _x,
        predictive.mean + 2 * _latent_std,
        color="black",
        linewidth=1.3,
        alpha=0.75,
        linestyle="--",
    )
    _draw_lines = [
        _sample_ax.plot([], [], color="0.35", alpha=0.14, linewidth=0.8)[0]
        for _ in range(8)
    ]
    (_mean_line,) = _sample_ax.plot(
        [], [], color="tab:blue", linestyle=":", linewidth=2, label="sample mean"
    )
    _sample_band = [
        _sample_ax.fill_between(
            _x,
            _x * 0.0,
            _x * 0.0,
            color="tab:orange",
            alpha=0.18,
            label="RFF pathwise spread around GP mean",
        )
    ]
    _sample_ax.set_xlim(float(_x.min()), float(_x.max()))
    _sample_ax.set_ylim(-2.2, 2.2)
    _sample_ax.set_xlabel("x")
    _sample_ax.set_ylabel("latent f(x)")
    _sample_ax.legend(
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.22),
        mode="expand",
        ncol=4,
        borderaxespad=0.0,
        framealpha=0.9,
    )

    _variance_ax.plot(
        _x, predicted_variance, color="tab:green", linewidth=2, label="GP exact variance"
    )
    (_variance_line,) = _variance_ax.plot(
        [], [], color="tab:blue", linestyle=":", linewidth=2, label="sample variance"
    )
    _variance_ax.set_xlim(float(_x.min()), float(_x.max()))
    _variance_ax.set_ylim(
        0,
        float(max(predicted_variance.max(), np.max([_s[1].max() for _s in _frame_stats])) * 1.2),
    )
    _variance_ax.set_title("RFF pathwise variance\nvs exact GP variance")
    _variance_ax.set_xlabel("x")
    _variance_ax.set_ylabel("variance")
    _variance_ax.legend(fontsize=8)

    (_kl_line,) = _kl_ax.plot([], [], color="tab:orange", linewidth=2)
    _kl_ax.set_xlim(float(_x.min()), float(_x.max()))
    _kl_ax.set_ylim(0, float(max(np.max([_s[2].max() for _s in _frame_stats]), 1e-6) * 1.1))
    _kl_ax.set_title("Pointwise KL: sample Normal\nvs GP predictive Normal")
    _kl_ax.set_xlabel("x")
    _kl_ax.set_ylabel("KL")

    _trace_ax.plot(frame_counts, mean_rmse_trace, color="tab:purple", label="mean RMSE")
    _trace_ax.plot(frame_counts, variance_rel_rmse_trace, color="tab:red", label="variance rel. RMSE")
    _trace_ax.plot(frame_counts, kl_trace, color="tab:orange", label="mean pointwise KL")
    _trace_marker = _trace_ax.axvline(frame_counts[0], color="0.2", linestyle="--", linewidth=1)
    _trace_ax.set_xscale("log")
    _trace_ax.set_xlabel("cumulative sample count")
    _trace_ax.set_ylabel("diagnostic value")
    _trace_ax.set_title("Moment diagnostics over cumulative samples")
    _trace_ax.legend(fontsize=8)

    _title = _sample_ax.set_title("", pad=58, fontsize=11, color="0.25")

    def _update(_frame_index):
        _k = int(frame_counts[_frame_index])
        _sample_mean, _sample_variance, _pointwise_kl = _frame_stats[_frame_index]
        _sample_std = np.sqrt(_sample_variance)

        for _idx, _line in enumerate(_draw_lines):
            if _idx < min(_k, len(_draw_lines)):
                _line.set_data(_x, pathwise_samples[_idx])
            else:
                _line.set_data([], [])

        _mean_line.set_data(_x, _sample_mean)
        _sample_band[0].remove()
        _sample_band[0] = _sample_ax.fill_between(
            _x,
            predictive.mean - 2 * _sample_std,
            predictive.mean + 2 * _sample_std,
            color="tab:orange",
            alpha=0.18,
        )
        _variance_line.set_data(_x, _sample_variance)
        _kl_line.set_data(_x, _pointwise_kl)
        _trace_marker.set_xdata([_k, _k])
        _title.set_text(f"Pathwise sample accumulation ({_k}/{len(pathwise_samples)} samples)")
        return (
            *_draw_lines,
            _mean_line,
            _sample_band[0],
            _variance_line,
            _kl_line,
            _trace_marker,
            _title,
        )

    _fig.tight_layout()
    _animation = FuncAnimation(
        _fig,
        _update,
        frames=len(frame_counts),
        interval=260,
        repeat=True,
        blit=False,
    )
    _video_html = _animation.to_html5_video().replace(
        " controls autoplay loop>", " controls autoplay loop muted playsinline>"
    )
    _plot = mo.vstack(
        [
            mo.md(
                """
                ## Cumulative Sampling Animation

                The video below is generated when this notebook runs. It shows
                pathwise samples accumulating and compares their empirical moments
                with the exact GP predictive moments.

                Exact pathwise samples target the latent posterior variance. The
                backend predictive variance includes the learned observation-noise
                term in this route, so this notebook subtracts that noise before
                drawing the green exact latent band.
                """
            ),
            mo.Html(_video_html),
        ]
    )
    plt.close(_fig)
    _plot
    return


@app.cell
def _(
    bulk_amortized_ms,
    bulk_sample_seconds,
    mean_rmse,
    mean_within_3se,
    mo,
    moment_check_pass,
    n_rff_features,
    n_samples,
    pathwise_info,
    pathwise_samples,
    standalone_sample_ms,
    train_result,
    variance_rel_rmse,
    variance_within_3se,
):
    status = "PASS" if moment_check_pass else "CHECK"
    summary_rows = [
        {"Quantity": "Status", "Value": status},
        {"Quantity": "Final NLL", "Value": f"{train_result.nll:.4f}"},
        {"Quantity": "Pathwise sample shape", "Value": f"{pathwise_samples.shape}"},
        {"Quantity": "Number of samples", "Value": f"{n_samples}"},
        {"Quantity": "RFF features", "Value": f"{n_rff_features}"},
        {"Quantity": "Sampling route", "Value": f"{pathwise_info.get('actual_sampling_route', 'n/a')}"},
        {"Quantity": "Bulk sampling time", "Value": f"{bulk_sample_seconds:.3f}s"},
        {"Quantity": "Bulk amortized time per sample", "Value": f"{bulk_amortized_ms:.3f} ms"},
        {"Quantity": "Standalone one-sample call time", "Value": f"{standalone_sample_ms:.3f} ms"},
        {"Quantity": "Sample mean RMSE vs GP mean", "Value": f"{mean_rmse:.5f}"},
        {"Quantity": "Variance relative RMSE vs GP variance", "Value": f"{variance_rel_rmse:.4f}"},
        {"Quantity": "Mean points within 3 Monte Carlo SE", "Value": f"{100.0 * mean_within_3se:.1f}%"},
        {"Quantity": "Variance points within 3 Monte Carlo SE", "Value": f"{100.0 * variance_within_3se:.1f}%"},
    ]

    mo.ui.table(summary_rows, label="Pathwise sampling moment check")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Reference

        Wilson, J.T., Borovitskiy, V., Terenin, A., Mostowsky, P. and Deisenroth, M.P. (2020). Pathwise Conditioning of Gaussian Processes. [online] arXiv.org. Available at: https://arxiv.org/abs/2011.04026 [Accessed 22 May 2026].
        """
    )
    return


if __name__ == "__main__":
    app.run()
