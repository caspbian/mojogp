import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # Model Persistence Roundtrip

    This notebook verifies a narrow persistence workflow: fit a single-output
    ExactGP, save it, reload it, and compare predictions from the original and
    loaded wrappers on the same test inputs.
    """
    )
    return (mo,)


@app.cell
def _():
    import tempfile
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from mojogp import RBF, SingleOutputGP

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(17)
    X_train = np.linspace(-3.5, 3.5, 70, dtype=np.float32).reshape(-1, 1)
    X_test = np.linspace(-4.0, 4.0, 90, dtype=np.float32).reshape(-1, 1)
    y_train = (
        np.sin(1.2 * X_train[:, 0])
        + 0.2 * np.cos(2.4 * X_train[:, 0])
        + 0.1 * rng.standard_normal(len(X_train))
    ).astype(np.float32)
    return (
        Path,
        RBF,
        SingleOutputGP,
        X_test,
        X_train,
        np,
        plt,
        tempfile,
        y_train,
    )


@app.cell
def _(Path, RBF, SingleOutputGP, X_test, X_train, np, tempfile, y_train):
    gp = SingleOutputGP(RBF(lengthscale=0.85, outputscale=1.0))
    train_result = gp.fit(
        X_train,
        y_train,
        max_iterations=18,
        learning_rate=0.035,
        method="materialized",
        verbose=False,
        progress=True,
    )
    original_prediction = gp.predict(X_test, variance_method="exact", progress=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "single_output_exactgp_roundtrip"
        gp.save(str(save_path))
        loaded = SingleOutputGP.load(str(save_path))
        loaded_prediction = loaded.predict(X_test, variance_method="exact", progress=True)

    mean_max_diff = float(
        np.max(np.abs(original_prediction.mean - loaded_prediction.mean))
    )
    std_max_diff = float(np.max(np.abs(original_prediction.std - loaded_prediction.std)))
    return (
        loaded_prediction,
        mean_max_diff,
        original_prediction,
        std_max_diff,
        train_result,
    )


@app.cell
def _(
    X_test,
    X_train,
    loaded_prediction,
    mo,
    original_prediction,
    plt,
    y_train,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(X_train[:, 0], y_train, s=12, alpha=0.35, label="train")
    ax.plot(X_test[:, 0], original_prediction.mean, label="original mean")
    ax.plot(
        X_test[:, 0],
        loaded_prediction.mean,
        linestyle="--",
        label="loaded mean",
    )
    ax.set_title("Original vs loaded predictions")
    ax.legend()
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mean_max_diff, mo, std_max_diff, train_result):
    mo.md(f"""
    ## Roundtrip Check

    | Quantity | Value |
    |---|---:|
    | Final NLL | {train_result.nll:.4f} |
    | Max mean difference after reload | {mean_max_diff:.8f} |
    | Max std difference after reload | {std_max_diff:.8f} |

    The loaded model should reproduce the original model's predictions up to
    numerical roundoff on the same route and inputs.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
