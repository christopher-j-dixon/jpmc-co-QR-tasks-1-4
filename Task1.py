import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.stattools import adfuller as adf
import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data file
df = pd.read_csv('Nat_Gas.csv')

# Convert 'Dates' feature to datetime
df['Dates'] = pd.to_datetime(df['Dates'])

# Check the data file
print(df.isnull().sum())
print(df.dtypes.unique())
print(df.duplicated().sum())

# Inspect the data file
print(df.head())
print(df.tail())
print(df.describe())

# Time series plotting function
def time_series_plot(dates, prices, label):
    """
    Time series plotting function.

    Args:
        dates (pandas.core.series.Series): Dates.
        prices (pandas.core.series.Series): Prices.

    Returns:
        Plot : Plot of given prices over dates.
    """

    # Configure the plot style
    sns.set(style='whitegrid')

    # Plot the input data
    plt.plot(dates, prices, label=label)

    # Configure the axis & labels 
    plt.xlabel("Date (YYYY-MM)")
    plt.ylabel("Price ($)")

    # Configure the x-axis format
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(md.MonthLocator(interval=3)) # interval=3 to showcase seasonality
    plt.gcf().autofmt_xdate()

    # Configure legend and gird
    plt.grid(True)
    plt.legend()

# Plot of original data file
plt.figure(figsize=(12,6))
plt.title('Natural Gas Prices (2020-10-31) - (2024-09-30)')
time_series_plot(df['Dates'], df['Prices'], 'Natural gas prices')

# Seasonal time series deocomposition of the data file & plot
decomposed_df = sd(df['Prices'], period=12)
decomposed_df.plot()

# 0.05 p-value ADF test for stationary points 
ADFtest = adf(df['Prices'])
print(f'ADF Test p-value: {ADFtest[1]}.')

# Chronological split data into 80% training and 20% testing
split = int(0.8 * len(df))
train_df = df.iloc[:split]
test_df = df.iloc[split:]

# Fit the SARIMA model using training data (lowest AIC)
train_model = pm.auto_arima(train_df['Prices'], seasonal=True, m=12)

# Generate the SARIMA model predictions over test dates
train_pred = train_model.predict(n_periods=len(test_df))

# Plot the model predictions vs actual prices over test dates
plt.figure(figsize=(12,6))
plt.title('Trained vs Actual Prices (2023-12-31) -  (2024-09-30)')
time_series_plot(test_df['Dates'], test_df['Prices'], 'Actual')
time_series_plot(test_df['Dates'], train_pred, 'Predicted')

# Summerise the test model
print(train_model.summary())

# Train data SARIMA model evaluation
print('Model Evaluation on Test Data:')
print('MAE:', mean_absolute_error(test_df['Prices'], train_pred))
print('RMSE:', np.sqrt(mean_squared_error(test_df['Prices'], train_pred)))

# Fit the SARIMA model using all the data
model = pm.auto_arima(df['Prices'], seasonal=True, m=12)

# Generate SARIMA model predictions over the next year 
predict = model.predict(n_periods=13)

# Prediction dates 
predict_dates = pd.date_range(start=df['Dates'].iloc[-1], periods=13, freq='M')

# Plot the predicted data and the actual data
plt.figure(figsize=(12,6))
plt.title('Forcasted prices')
time_series_plot(df['Dates'], df['Prices'], 'Actual')
time_series_plot(predict_dates, predict, 'Forcasted')

def price_forcast(date):

    """
    Predicts the Gas price for a given date in range (2024-10-31) - (2025-09-30).
    
    Parameters:
    - date (str): The date in 'YYYY-MM-DD' format
    
    Returns:
    - Gas price prediction (float): predicted gas price on given date
    """

    # Converts string date to datetime format
    input_date = pd.to_datetime(date)

    # Converts predict to numpy array for indexing
    prediction_array = predict.values

    # Check for date region and return accordingly
    if predict_dates.min() <= input_date <= predict_dates.max():

        # Diffrence in month for indexing
        months_difference = (input_date.year - predict_dates[0].year) * 12 + input_date.month - predict_dates[0].month

        return prediction_array[months_difference]
    
    elif input_date < predict_dates.min():
        return "Before prediction period. Look to historic price data."
    else:
        return "After prediction period."

# Testing
print(price_forcast("2024-09-30"))  # Start of the prediction range
print(price_forcast("2025-05-15"))  # Middle of the prediction range
print(price_forcast("2025-09-30"))  # End of the prediction range
print(price_forcast("2024-09-15"))  # A date before the prediction range
print(price_forcast("2025-10-01"))  # A date after the prediction range

# Show all above plots
plt.show()
