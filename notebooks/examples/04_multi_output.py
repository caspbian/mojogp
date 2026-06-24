import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Multi-Output GP with Correlated Tasks

        A user-facing `MultiOutputLMCGP` example on two related signals.

        MojoGP also exposes `MultiOutputGP` for ICM-style models. A minimal ICM
        call uses the same data shape: `MultiOutputGP(kernel=Kernel.rbf()).fit(X, Y)`.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(12)
    X_train = np.linspace(-4, 4, 80, dtype=np.float32).reshape(-1, 1)
    X_test = np.linspace(-4.5, 4.5, 120, dtype=np.float32).reshape(-1, 1)

    latent_train = np.sin(1.4 * X_train[:, 0]) + 0.3 * np.cos(3.0 * X_train[:, 0])
    latent_test = np.sin(1.4 * X_test[:, 0]) + 0.3 * np.cos(3.0 * X_test[:, 0])

    y0_train = (latent_train + 0.15 * rng.standard_normal(len(X_train))).astype(
        np.float32
    )
    y1_train = (
        0.7 * latent_train - 0.4 + 0.35 * rng.standard_normal(len(X_train))
    ).astype(np.float32)
    Y_train = np.column_stack([y0_train, y1_train]).astype(np.float32)
    Y_test = np.column_stack([latent_test, 0.7 * latent_test - 0.4]).astype(np.float32)

    fig, train_axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    train_axes[0].scatter(X_train[:, 0], y0_train, s=12, alpha=0.5)
    train_axes[0].set_title("Task 0")
    train_axes[1].scatter(X_train[:, 0], y1_train, s=12, alpha=0.5)
    train_axes[1].set_title("Task 1")
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return X_test, X_train, Y_test, Y_train, np, plt


@app.cell
def _(X_test, X_train, Y_test, Y_train, mo, np, plt):
    from mojogp import Kernel, MultiOutputLMCGP

    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=30,
        preconditioner_rank=8,
    )
    train_result = gp.fit(
        X_train,
        Y_train,
        max_iterations=20,
        learning_rate=0.03,
        method="materialized",
    )
    mean, std = gp.predict(X_test, return_std=True, variance_method="exact")
    scores = gp.score(X_test, Y_test)
    samples = gp.sample_posterior(X_test[::12], n_samples=3, method="diagonal")
    B = gp.task_covariance

    fig_pred, pred_axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for task in range(2):
        pred_axes[task].plot(
            X_test[:, 0], Y_test[:, task], "k--", alpha=0.7, label="truth"
        )
        pred_axes[task].scatter(
            X_train[:, 0], Y_train[:, task], s=10, alpha=0.3, label="train"
        )
        pred_axes[task].plot(
            X_test[:, 0], mean[:, task], color=f"C{task}", label="LMC mean"
        )
        pred_axes[task].fill_between(
            X_test[:, 0],
            mean[:, task] - 2 * std[:, task],
            mean[:, task] + 2 * std[:, task],
            alpha=0.2,
            color=f"C{task}",
        )
        pred_axes[task].set_title(f"Task {task}")
        pred_axes[task].legend(fontsize=8)
    fig_pred.tight_layout()
    display_items = [
        mo.mpl.interactive(fig_pred),
        mo.md(
            f"""
            ## Fit Summary

            | Quantity | Value |
            |---|---:|
            | Overall RMSE | {scores["rmse"]:.4f} |
            | Task 0 RMSE | {scores["rmse_per_task"][0]:.4f} |
            | Task 1 RMSE | {scores["rmse_per_task"][1]:.4f} |
            | Mean noise | {float(np.mean(train_result.noise_per_task)):.4f} |
            | Diagonal sample shape | `{samples.shape}` |

            This executed example uses the materialized LMC route to keep the notebook
            lightweight. Use `method="matrix_free"` for larger training sets where the
            dense train kernel no longer fits comfortably.
            """
        ),
    ]
    fig_cov = None
    if B is not None:
        fig_cov, cov_ax = plt.subplots(figsize=(4, 4))
        im = cov_ax.imshow(B, cmap="viridis")
        cov_ax.set_title("Learned task covariance")
        fig_cov.colorbar(im, ax=cov_ax, shrink=0.8)
        fig_cov.tight_layout()
        display_items.append(mo.mpl.interactive(fig_cov))
    mo.vstack(display_items)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## References

        Bonilla, E.V., Chai, K. and Williams, C. (2007). Multi-task Gaussian Process Prediction. [online] Neural Information Processing Systems. Available at: https://papers.nips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html.

        Bruinsma, W.P., Perim, E., Tebbutt, W., Scott, H.J., Solin, A. and Turner, R.E. (2019). Scalable Exact Inference in Multi-Output Gaussian Processes. [online] arXiv.org. Available at: https://arxiv.org/abs/1911.06287 [Accessed 22 May 2026].
        """
    )
    return


if __name__ == "__main__":
    app.run()
