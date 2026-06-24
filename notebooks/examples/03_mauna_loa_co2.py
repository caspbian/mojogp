import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Seasonal Time Series with a Composite Kernel

        A local CO2-style example: smooth long-term drift plus an annual cycle,
        modeled with `SingleOutputGP` and `RBF() + Periodic()`.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    months = np.arange(0, 8 * 12, dtype=np.float32)
    years = months / 12.0
    trend = 0.18 * years
    seasonal = 0.7 * np.sin(2 * np.pi * years)
    y = (320.0 + trend + seasonal + 0.08 * rng.standard_normal(len(years))).astype(
        np.float32
    )

    split = 6 * 12
    X_train = years[:split].reshape(-1, 1)
    y_train = y[:split]
    X_test = years[split:].reshape(-1, 1)
    y_test = y[split:]

    fig, series_ax = plt.subplots(figsize=(10, 4))
    series_ax.plot(years, y, color="0.35")
    series_ax.axvline(years[split], linestyle="--", color="tab:red", alpha=0.6)
    series_ax.set_title("Synthetic seasonal time series")
    series_ax.set_xlabel("years since start")
    fig.tight_layout()
    mo.mpl.interactive(fig)
    return X_test, X_train, np, plt, split, y, y_test, y_train, years


@app.cell
def _(X_test, X_train, mo, np, plt, split, y, y_test, y_train, years):
    from mojogp import SingleOutputGP, Periodic, RBF

    kernel = RBF(lengthscale=3.0) + Periodic(lengthscale=1.0, period=1.0)
    gp = SingleOutputGP(kernel)
    train_result = gp.fit(
        X_train,
        y_train,
        max_iterations=30,
        learning_rate=0.035,
        method="materialized",
    )

    mean, std = gp.predict(X_test, return_std=True)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))
    params = gp.get_learned_params()

    fig_pred, forecast_ax = plt.subplots(figsize=(10, 4))
    forecast_ax.plot(years[:split], y[:split], label="train", color="tab:blue")
    forecast_ax.plot(years[split:], y_test, label="held-out truth", color="tab:red")
    forecast_ax.plot(X_test[:, 0], mean, label="prediction", color="tab:green")
    forecast_ax.fill_between(
        X_test[:, 0], mean - 2 * std, mean + 2 * std, color="tab:green", alpha=0.2
    )
    forecast_ax.set_title("Composite kernel forecast")
    forecast_ax.legend()
    fig_pred.tight_layout()
    mo.vstack(
        [
            mo.mpl.interactive(fig_pred),
            mo.md(
                f"""
                ## Fit Summary

                | Quantity | Value |
                |---|---:|
                | RMSE on held-out tail | {rmse:.4f} |
                | Final NLL | {train_result.nll:.4f} |
                | Learned periodic period | {params["right_periodic_period"]:.4f} |
                | Learned noise | {params["noise"]:.4f} |
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
