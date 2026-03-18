import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.arima.model import ARIMA
from database import read_table, store_projections

def forecast_portfolio_value(last_value, forecast_df):
    mean_cum = forecast_df["forecast"].cumsum()
    lower_cum = forecast_df["lower_ci"].cumsum()
    upper_cum = forecast_df["upper_ci"].cumsum()

    forecast_df["mean_value"] = last_value * np.exp(mean_cum)
    forecast_df["lower_value"] = last_value * np.exp(lower_cum)
    forecast_df["upper_value"] = last_value * np.exp(upper_cum)

    return forecast_df

def returns_distribution(eq_weights, optimized_weights, trading_days=252):
    df = pd.read_csv("data/historical_data.csv")
    df = df[['symbol','open','high','low','close','volume','timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    df['ret_daily'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x / x.shift(1)))
    benchmark = df[df['symbol'] == 'NIFTY']
    assets = df[df['symbol'] != 'NIFTY']
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    eq_portfolio_returns = asset_returns.dot(eq_weights)
    optimized_portfolio_returns = asset_returns.dot(optimized_weights)
    eq_portfolio_returns, benchmark_returns = eq_portfolio_returns.align(benchmark_returns, join='inner')
    optimized_portfolio_returns, benchmark_returns = optimized_portfolio_returns.align(benchmark_returns, join='inner')
    return eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns

def portfolio_value(returns, initial_value=10000):
    return initial_value * (1 + returns).cumprod()

def forecast_portfolio_returns(returns, periods=90):
    returns.index = pd.DatetimeIndex(returns.index)
    returns = returns.asfreq('B')
    returns = returns.fillna(method='ffill')
    model = ARIMA(returns, order=(1, 0, 1))
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=periods)
    mean_fcast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.05)
    future_index = pd.bdate_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({
        "forecast": mean_fcast,
        "lower_ci": confidence_intervals.iloc[:, 0],
        "upper_ci": confidence_intervals.iloc[:, 1]
    }, index = future_index)
    store_projections(forecast_df)

    return forecast_df

if __name__ == "__main__":
    optimizer_results = read_table('optimizer_results')
    eq_weights = pd.Series(optimizer_results['equal_weight'].values, index=optimizer_results['symbol'])
    optimized_weights = pd.Series(optimizer_results['optimal_weight'].values, index=optimizer_results['symbol'])
    eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns = returns_distribution(eq_weights, optimized_weights)
    opt_forecast = forecast_portfolio_returns(optimized_portfolio_returns)
    last_value = portfolio_value(optimized_portfolio_returns).iloc[-1]
    fc_value = (1 + opt_forecast['forecast']).cumprod().iloc[-1] * last_value
    low_value = (1 + opt_forecast['lower_ci']).cumprod().iloc[-1] * last_value
    up_value = (1 + opt_forecast['upper_ci']).cumprod().iloc[-1] * last_value
    print("fc_value:", fc_value)
    print("low_value:", low_value)
    print("up_value:", up_value)

    final_value = 10000 * (1 + optimized_portfolio_returns).cumprod().iloc[-1]
    print("final_value:", final_value)
    # print("optimized_portfolio_returns: ", optimized_portfolio_returns)
    # print(opt_forecast)
