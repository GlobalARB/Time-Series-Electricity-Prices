import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

from data import load_data


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects subsets of feature columns inside a sklearn pipeline."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df):
        return df[self.columns]


def build_model() -> BaseEstimator:
    """Set up the model.

    Depending on the type of model you build, it may be convenient
    to encapsulate it in a suitable object with .fit() and .predict() methods.
    """

    return make_pipeline(
        ColumnSelector(["TotalLoadValue", "Solar", "Wind Offshore", "Wind Onshore"]),
        LinearRegression(),
    )


def evaluate_model(model: BaseEstimator) -> None:
    # The dataset contains data for 12 months of 2024
    # This is an example of a possible simple train-test split:
    df_train = load_data(from_month=1, to_month=6)
    df_test = load_data(from_month=7, to_month=8)

    # define target
    target_col = "Price[Currency/MWh]"
    X_train, y_train = df_train.drop(columns=target_col), df_train[target_col]
    X_test, y_test = df_test.drop(columns=target_col), df_test[target_col]

    # fit model
    print(
        f"Training on {len(X_train)} rows ({X_train['DateTime'].min()}..{X_train['DateTime'].max()})"
    )
    model.fit(X_train, y_train)

    # This is an example of some metrics you may consider:
    print(
        f"Testing on {len(X_test)} rows ({X_test['DateTime'].min()}..{X_test['DateTime'].max()})"
    )
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    print(f"overall RMSE train: {rmse_train:.1f}, test: {rmse_test:.1f}")

    # We may also want to look at the prediction errors in different ways:
    area_codes = ["10Y1001A1001A82H", "10YFR-RTE------C"]
    fig, axes = plt.subplots(
        ncols=len(area_codes), sharex=True, sharey=True, figsize=(10, 4)
    )
    fig.canvas.manager.set_window_title("Example of error plot")
    for ax, area_code in zip(axes, area_codes, strict=True):
        selection = X_test["AreaCode"] == area_code
        rmse_selected = root_mean_squared_error(
            y_test[selection], y_test_pred[selection]
        )

        ax.scatter(y_test[selection], y_test_pred[selection], s=1)
        ax.set_title(f"{area_code} RMSE={rmse_selected:.1f}")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    print("Close plot window to exit...")
    plt.show()


if __name__ == "__main__":
    model = build_model()
    evaluate_model(model)
