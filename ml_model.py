import pandas as pd
from prophet import Prophet

def train_model_1(data):
    """
    Trains a Prophet model using time series data from a transformed dataset, and
    generates a forecast for future values.

    Args:
        - data: A Pandas DataFrame containing time series data with a 'Date' column
        and a 'Close apple' column.

    Returns:
        - A Pandas DataFrame containing the forecasted values for the time series,
        along with actual values.

    Example Usage:
        transformed_data = pd.read_csv(transformed_dataset_dir, index_col=0)
        forecast = train_model_1(transformed_data)

    """

    data.set_index("Date", inplace=True)
    # Create a new DataFrame with the closing price of Apple for the training data
    training_data = data.loc["2017-04-01":"2022-04-30", :]

    training_data = pd.DataFrame(
        {"ds": training_data.index, "y": training_data.loc[:,"Close apple"]}
    )

    # create model
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    # train model
    model.fit(training_data)

    # future data
    future_data = pd.DataFrame({"ds": data.index})

    # predict
    forecast = model.predict(future_data)

    # add actual values
    forecast["actual"] = data.loc[:, "Close apple"].reset_index(drop=True)
    print(forecast.head())
    model.plot_components(forecast)
    plt.savefig(os.path.join(figures_dir, "model_1_components.png"))
    return forecast