import os

import pandas as pd


def _data_path(key: str, year: int, month: int) -> str:
    return os.path.join(
        os.path.dirname(__file__), "data", f"{year}_{month:02d}_{key}.csv.gz"
    )


def _read_day_ahead_prices(year: int, month: int) -> pd.DataFrame:
    df = pd.read_csv(
        _data_path("EnergyPrices_12.1.D_r3", year=year, month=month),
        sep="\t",
        parse_dates=["DateTime(UTC)"],
    )

    # BZN means bidding zone. The value is always BZN, but we make sure here.
    df = df[df["AreaTypeCode"] == "BZN"]

    # The value is always Day-ahead, but we make sure here.
    df = df[df["ContractType"] == "Day-ahead"]

    # We assume that the other DateTime columns are all UTC as well, so we rename this column to match the others.
    df.rename(columns={"DateTime(UTC)": "DateTime"}, inplace=True)

    # We drop the columns that we aren't going to use.
    df = df[["DateTime", "ResolutionCode", "AreaCode", "Price[Currency/MWh]"]]

    # We convert a few columns from string to categorical (reducing memory usage).
    df["ResolutionCode"] = df["ResolutionCode"].astype("category")
    df["AreaCode"] = df["AreaCode"].astype("category")

    return df


def _read_load_forecasts(year: int, month: int) -> pd.DataFrame:
    df = pd.read_csv(
        _data_path("DayAheadTotalLoadForecast_6.1.B", year=year, month=month),
        sep="\t",
        parse_dates=["DateTime"],
    )

    # Similar comments apply here as those in read_day_ahead_prices

    df = df[df["AreaTypeCode"] == "BZN"]

    df = df[["DateTime", "ResolutionCode", "AreaCode", "TotalLoadValue"]]

    df["ResolutionCode"] = df["ResolutionCode"].astype("category")
    df["AreaCode"] = df["AreaCode"].astype("category")

    return df


def _read_renewables_forecasts(year: int, month: int) -> pd.DataFrame:
    df = pd.read_csv(
        _data_path(
            "DayAheadGenerationForecastForWindAndSolar_14.1.D", year=year, month=month
        ),
        sep="\t",
        parse_dates=["DateTime"],
    )

    # Similar comments apply here as those in read_day_ahead_prices

    df = df[df["AreaTypeCode"] == "BZN"]

    df = df[
        [
            "DateTime",
            "ResolutionCode",
            "AreaCode",
            "ProductionType",
            "AggregatedGenerationForecast",
        ]
    ]

    df["ResolutionCode"] = df["ResolutionCode"].astype("category")
    df["AreaCode"] = df["AreaCode"].astype("category")
    df["ProductionType"] = df["ProductionType"].astype("category")

    # The dataframe is tall - there's a column called ProductionType, and it can take one of three values:
    # Solar, Wind Onshore, Wind Offshore
    # We want to convert it to a wide dataframe - i.e. one with columns called Solar, Onshore Wind, Offshore Wind.
    df.set_index(["DateTime", "ResolutionCode", "AreaCode"], inplace=True)
    df.sort_index(inplace=True)
    df = df.pivot(columns="ProductionType", values="AggregatedGenerationForecast")
    df.reset_index(inplace=True)

    for column in ["Solar", "Wind Offshore", "Wind Onshore"]:
        df[column] = df[column].fillna(0)

    return df


def _load_one_month(year: int, month: int) -> pd.DataFrame:
    price_df = _read_day_ahead_prices(year, month)
    load_forecasts_df = _read_load_forecasts(year, month)
    renewables_forecasts_df = _read_renewables_forecasts(year, month)

    df = pd.merge(
        price_df,
        load_forecasts_df,
        on=["DateTime", "ResolutionCode", "AreaCode"],
        how="inner",
    )
    df = pd.merge(
        df,
        renewables_forecasts_df,
        on=["DateTime", "ResolutionCode", "AreaCode"],
        how="inner",
    )
    return df


def _months(from_year: int, from_month: int, to_year: int, to_month: int):
    year, month = from_year, from_month
    while (year, month) < (to_year, to_month):
        assert 1 <= month <= 12
        yield (year, month)
        month += 1
        if month > 12:
            year, month = year + 1, 1


def load_data(
    from_year: int = 2024, from_month: int = 1, to_year: int = 2024, to_month: int = 12
) -> pd.DataFrame:
    assert (from_year, from_month) <= (to_year, to_month)
    return pd.concat(
        _load_one_month(year, month)
        for year, month in _months(
            from_year=from_year,
            from_month=from_month,
            to_year=to_year,
            to_month=to_month,
        )
    )


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(df)
