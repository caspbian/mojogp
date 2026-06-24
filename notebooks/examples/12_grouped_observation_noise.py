import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Grouped Observation Noise

        Fixed grouped noise maps each row to a known group variance and expands
        exactly to a diagonal vector internally. This preserves the same exact GP
        objective as fixed per-sample noise while keeping group metadata for
        observed prediction.

        In this notebook, **latent** means the underlying noise-free function
        `f(x)`. **Observed** means a future noisy measurement `y(x) = f(x) + eps`.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D as _Line2D
    import numpy as np

    rng = np.random.default_rng(32)
    X_train = np.linspace(-5, 5, 150, dtype=np.float32).reshape(-1, 1)
    X_test = np.linspace(-5.5, 5.5, 100, dtype=np.float32).reshape(-1, 1)
    train_groups = np.digitize(X_train[:, 0], bins=[-1.5, 1.5]).astype(np.int32)
    test_groups = np.digitize(X_test[:, 0], bins=[-1.5, 1.5]).astype(np.int32)
    group_noise = np.array([0.015, 0.045, 0.09], dtype=np.float32)
    group_palette = np.array(["tab:blue", "tab:green", "tab:purple"])
    group_labels = [
        "group 0: low noise",
        "group 1: medium noise",
        "group 2: high noise",
    ]

    y_true = (np.sin(1.2 * X_test[:, 0]) + 0.1 * X_test[:, 0]).astype(np.float32)
    latent_train = (np.sin(1.2 * X_train[:, 0]) + 0.1 * X_train[:, 0]).astype(np.float32)
    y_train = (
        latent_train
        + rng.standard_normal(len(X_train)).astype(np.float32)
        * np.sqrt(group_noise[train_groups])
    ).astype(np.float32)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(
        X_train[:, 0],
        y_train,
        c=group_palette[train_groups],
        s=13,
        alpha=0.6,
    )
    handles = [
        _Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=7,
        )
        for label, color in zip(group_labels, group_palette)
    ]
    ax.set_title("Grouped-noise training data")
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return (
        X_test,
        X_train,
        group_labels,
        group_noise,
        group_palette,
        np,
        plt,
        test_groups,
        train_groups,
        y_train,
        y_true,
    )


@app.cell
def _(
    X_test,
    X_train,
    group_labels,
    group_noise,
    group_palette,
    mo,
    np,
    plt,
    test_groups,
    train_groups,
    y_train,
    y_true,
):
    from mojogp import RBF, SingleOutputGP

    gp = SingleOutputGP(RBF())
    train_result = gp.fit(
        X_train,
        y_train,
        noise_model="grouped",
        noise_group_train=train_groups,
        group_noise=group_noise,
        learn_noise=False,
        max_iterations=20,
        learning_rate=0.035,
        method="materialized",
        progress=True,
    )

    latent = gp.predict_latent(X_test, variance_method="exact", progress=True)
    observed = gp.predict_observed(
        X_test,
        noise_group_test=test_groups,
        variance_method="exact",
        progress=True,
    )
    rmse = float(np.sqrt(np.mean((latent.mean - y_true) ** 2)))
    test_noise = group_noise[test_groups]
    observed_variance_from_sum = latent.variance + test_noise
    observed_variance_gap = float(np.max(np.abs(observed.variance - observed_variance_from_sum)))
    noise_interval_width = 2 * np.sqrt(test_noise)
    observed_interval_width = 2 * observed.std

    from matplotlib.lines import Line2D as _Line2D

    fig_pred, pred_ax = plt.subplots(figsize=(9, 4.6))
    pred_ax.scatter(
        X_train[:, 0],
        y_train,
        c=group_palette[train_groups],
        s=10,
        alpha=0.35,
        zorder=3,
    )
    pred_ax.plot(X_test[:, 0], y_true, "k--", label="latent truth", zorder=4)
    pred_ax.plot(X_test[:, 0], latent.mean, color="tab:blue", label="latent mean", zorder=5)
    pred_ax.fill_between(
        X_test[:, 0],
        latent.mean - noise_interval_width,
        latent.mean + noise_interval_width,
        color="tab:orange",
        alpha=0.18,
        label="group observation-noise component",
        zorder=1,
    )
    pred_ax.plot(
        X_test[:, 0],
        latent.mean - noise_interval_width,
        color="tab:orange",
        linestyle="--",
        linewidth=0.9,
        alpha=0.85,
        zorder=2,
    )
    pred_ax.plot(
        X_test[:, 0],
        latent.mean + noise_interval_width,
        color="tab:orange",
        linestyle="--",
        linewidth=0.9,
        alpha=0.85,
        label="group noise boundary",
        zorder=2,
    )
    pred_ax.fill_between(
        X_test[:, 0],
        latent.mean + noise_interval_width,
        latent.mean + observed_interval_width,
        color="tab:blue",
        alpha=0.18,
        label="latent uncertainty component",
        zorder=2,
    )
    pred_ax.fill_between(
        X_test[:, 0],
        latent.mean - observed_interval_width,
        latent.mean - noise_interval_width,
        color="tab:blue",
        alpha=0.18,
        zorder=2,
    )
    pred_ax.plot(
        X_test[:, 0],
        latent.mean - observed_interval_width,
        color="tab:blue",
        linewidth=0.9,
        alpha=0.75,
        label="total predictive boundary",
        zorder=2,
    )
    pred_ax.plot(
        X_test[:, 0],
        latent.mean + observed_interval_width,
        color="tab:blue",
        linewidth=0.9,
        alpha=0.75,
        zorder=2,
    )
    pred_ax.axvline(-1.5, color="0.45", linestyle=":", linewidth=1.0, label="group boundary", zorder=4)
    pred_ax.axvline(1.5, color="0.45", linestyle=":", linewidth=1.0, zorder=4)
    pred_ax.set_title("Predictive uncertainty decomposed by observation-noise group", pad=70)
    group_handles = [
        _Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=6,
        )
        for label, color in zip(group_labels, group_palette)
    ]
    pred_ax.legend(
        handles=pred_ax.get_legend_handles_labels()[0] + group_handles,
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.25),
        mode="expand",
        ncol=4,
        borderaxespad=0.0,
        framealpha=0.9,
    )
    fig_pred.tight_layout()

    fig_var, var_ax = plt.subplots(figsize=(9, 4.4))
    var_ax.fill_between(
        X_test[:, 0],
        0.0,
        test_noise,
        color="tab:orange",
        alpha=0.22,
        step="post",
        label="group observation-noise variance",
    )
    var_ax.fill_between(
        X_test[:, 0],
        test_noise,
        observed_variance_from_sum,
        color="tab:blue",
        alpha=0.22,
        label="latent variance stacked on group noise",
    )
    var_ax.plot(
        X_test[:, 0],
        test_noise,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        drawstyle="steps-post",
        label="group noise boundary",
    )
    var_ax.plot(
        X_test[:, 0],
        test_noise + latent.variance,
        color="tab:blue",
        linewidth=1.4,
        label="group noise + latent boundary",
    )
    var_ax.plot(
        X_test[:, 0],
        observed.variance,
        color="black",
        linewidth=1.8,
        label="observed variance",
    )
    var_ax.plot(
        X_test[:, 0],
        observed_variance_from_sum,
        color="tab:green",
        linestyle=":",
        linewidth=1.4,
        label="latent + group noise",
    )
    var_ax.axvline(-1.5, color="0.45", linestyle=":", linewidth=1.0, label="group boundary")
    var_ax.axvline(1.5, color="0.45", linestyle=":", linewidth=1.0)
    var_ax.set_title("Variance decomposition by observation-noise group", pad=58)
    var_ax.set_xlabel("x")
    var_ax.set_ylabel("predictive variance")
    var_ax.legend(
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.22),
        mode="expand",
        ncol=4,
        borderaxespad=0.0,
        framealpha=0.9,
    )
    fig_var.tight_layout()
    mo.vstack(
        [
            mo.mpl.interactive(fig_pred),
            mo.mpl.interactive(fig_var),
            mo.md(
                f"""
                ## Fit Summary

                | Quantity | Value |
                |---|---:|
                | RMSE vs latent truth | {rmse:.4f} |
                | Final NLL | {train_result.nll:.4f} |
                | Group variances | {group_noise.tolist()} |
                | Mean latent std | {float(np.mean(latent.std)):.4f} |
                | Mean observed std | {float(np.mean(observed.std)):.4f} |
                | Max variance decomposition error | {observed_variance_gap:.8f} |
                """
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
