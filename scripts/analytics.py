#------------------------------------------------
#Computes all required risk/performance metrics for both
#individual assets and the portfolio.
#------------------------------------------------
from scipy import stats
import pandas as pd
import numpy as np
from .database import store_asset_metrics, store_benchmark_metrics, store_portfolio_metrics, store_corr_matrix

def summary_statistics(returns, benchmark_returns):

    benchmark_returns,_ = benchmark_returns.align(returns, join='inner')
    stats = pd.DataFrame()

    for i in returns.columns:
        stats[i] = pd.Series({
        "Mean Daily Return":  returns[i].mean()*100,
        "Std Dev (Daily)":    returns[i].std()*100,
        "Annualised Return":  returns[i].mean() * 252* 100,
        "Annualised Vol":     returns[i].std() * np.sqrt(252)* 100,
        "Skewness":           returns[i].skew(),
        "Kurtosis":           returns[i].kurt(),
        "Min":                returns[i].min()*100,
        "Max":                returns[i].max()*100,
        "Sharpe Ratio":       returns[i].mean() / returns[i].std() * np.sqrt(252),
        "Sortino Ratio":      returns[i].mean() / returns[i][returns[i]<0].std() * np.sqrt(252),
        "Beta":               np.cov(returns[i], benchmark_returns)[0][1] / np.var(benchmark_returns, ddof=1) # calculates benchmark_re - 1 in denimninator
    })
    return stats

def calculate_returns(returns):
    # 1D, 5D, 1month, 3month, 6month, 1year CAGR, 3year CAGR, 5year CAGR
    returns['ret_daily'] = returns.groupby('symbol')['close'].pct_change()
    returns['5D'] = returns.groupby('symbol')['close'].pct_change(5)
    returns['1month'] = returns.groupby('symbol')['close'].pct_change(21)
    returns['3month'] = returns.groupby('symbol')['close'].pct_change(63)
    returns['6month'] = returns.groupby('symbol')['close'].pct_change(126)
    returns['1Y_CAGR'] = returns.groupby('symbol')['close'].transform(lambda x: ((x / x.shift(252))**(1/1) - 1)*100)
    returns['3Y_CAGR'] = returns.groupby('symbol')['close'].transform(lambda x: ((x / x.shift(756))**(1/3) - 1)*100)
    returns['5Y_CAGR'] = returns.groupby('symbol')['close'].transform(lambda x: ((x / x.shift(1260))**(1/5) - 1)*100)
    benchmark = returns[returns['symbol'] == 'NIFTY'].reset_index(drop = True)
    assets = returns[returns['symbol'] != 'NIFTY'].reset_index(drop = True)

    return benchmark, assets

# volatility and tracking
def calculate_vol_tracking(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(
        index='timestamp',
        columns='symbol',
        values='ret_daily'
    ).dropna()

    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']

    asset_returns, benchmark_returns = asset_returns.align(benchmark_returns, join='inner', axis=0)

    portfolio_returns = asset_returns.dot(weights)

    # Portfolio variance
    port_vol_annual = portfolio_returns.std() * np.sqrt(trading_days)
    
    rolling_vol_annual = portfolio_returns.rolling(window=21).std() * np.sqrt(trading_days)
    rolling_vol_annual = rolling_vol_annual.iloc[-1]
    # Benchmark volatility
    bench_vol_annual = benchmark_returns.std() * np.sqrt(trading_days)

    # Tracking error
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(trading_days)

    return port_vol_annual.round(2), bench_vol_annual.round(2), tracking_error.round(2), rolling_vol_annual.round(2)



def calculate_risk_metrics(assets, benchmark, weights, trading_days=252):

    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()

    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']

    portfolio_returns = asset_returns.dot(weights)

    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    # Tracking Error
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(trading_days)

    # Information Ratio
    information_ratio = (excess_returns.mean() * trading_days)/tracking_error

    # Portfolio Sharpe
    portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std())*np.sqrt(trading_days)

    # Portfolio Sortino
    downside = portfolio_returns[portfolio_returns < 0].std()
    portfolio_sortino = (portfolio_returns.mean() / downside)*np.sqrt(trading_days)

    # Benchmark Sharpe
    bench_sharpe = (benchmark_returns.mean() / benchmark_returns.std())*np.sqrt(trading_days)

    bench_downside = benchmark_returns[benchmark_returns < 0].std()
    bench_sortino = (benchmark_returns.mean() / bench_downside) * np.sqrt(trading_days)

    return (
        portfolio_sharpe.round(2),
        portfolio_sortino.round(2),
        bench_sharpe.round(2),
        bench_sortino.round(2),
        information_ratio.round(2)
    )

# Drawdown:

def calculate_drawdown(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    # Portfolio Drawdown
    cum = (1 + portfolio_returns).cumprod()
    max_port = cum.cummax()
    drawdown = (cum - max_port) / max_port
    max_drawdown_port = drawdown.min()

    # Benchmark Drawdown
    cum_bench = (1 + benchmark_returns).cumprod()
    max_bench = cum_bench.cummax()
    drawdown_bench = (cum_bench - max_bench) / max_bench
    max_drawdown_bench = drawdown_bench.min()

    return max_drawdown_port.round(2), max_drawdown_bench.round(2)

# Risk Sensitivity:

def calculate_beta(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')
    beta_portfolio = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
    return beta_portfolio.round(2)


# Value at Risk:
'''
1D VaR (95%) — on a normal bad day (worst 5% of days), this is your expected maximum loss. If VaR is -2%, you should expect to lose more than 2% about 12-13 days per year.
1D CVaR (95%) — also called Expected Shortfall. The average loss on those worst 5% of days. Always worse than VaR. More honest about tail risk because VaR just tells you the threshold, CVaR tells you what happens beyond it.
Annualised VaR / CVaR — daily figures scaled up using √252. Gives a yearly risk estimate. Rough approximation — assumes returns are independent day to day which isn't perfectly true.
'''
def calculate_var(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')
    # Portfolio VAR
    sorted_port_ret = np.sort(portfolio_returns)
    portfolio_var = sorted_port_ret[int(0.05 * len(sorted_port_ret))]
    portfolio_cvar = sorted_port_ret[:int(0.05 * len(sorted_port_ret))].mean()
    portfolio_var_annual = portfolio_var * np.sqrt(trading_days)
    portfolio_cvar_annual = portfolio_cvar * np.sqrt(trading_days)

    return portfolio_var.round(2), portfolio_cvar.round(2), portfolio_var_annual.round(2), portfolio_cvar_annual.round(2)

# Alpha and Related Stats:

def calculate_alpha_metrics(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp', columns='symbol', values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

    # Beta
    beta = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)

    # Jensen's Alpha— excess return over CAPM expected return
    daily_rf = 0.0
    alpha_series = portfolio_returns - (daily_rf + beta * (benchmark_returns - daily_rf))

    jensens_alpha_daily = alpha_series.mean()

    # R-squared
    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0][1]
    r_squared = correlation ** 2

    # Alpha Skewness and Kurtosis
    alpha_skewness = alpha_series.skew()
    alpha_kurtosis = alpha_series.kurt()

    # Mean Alpha on Stress Days (bottom 5% benchmark returns)
    stress_threshold = benchmark_returns.quantile(0.05)
    stress_days = benchmark_returns[benchmark_returns <= stress_threshold].index
    mean_alpha_stress = alpha_series.loc[alpha_series.index.isin(stress_days)].mean()

    return (
        round(jensens_alpha_daily, 8),
        round(r_squared, 2),
        round(alpha_skewness, 2),
        round(alpha_kurtosis, 2),
        round(mean_alpha_stress, 8)
    )


# Distribution Metrics
def calculate_distribution_metrics(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    portfolio_returns = asset_returns.dot(weights)
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')
    port_folio_skew = portfolio_returns.skew()
    port_folio_kurt = portfolio_returns.kurt()
    bench_skew = benchmark_returns.skew()
    bench_kurt = benchmark_returns.kurt()
    return port_folio_skew.round(2), port_folio_kurt.round(2), bench_skew.round(2), bench_kurt.round(2)

# Diversification:
def calculate_corr_metrics(assets, benchmark, weights, trading_days=252):
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    corr_matrix = asset_returns.corr()
    return corr_matrix.round(2)


def main():
    df = pd.read_csv("data/historical_data.csv")
    df = df[['symbol','open','high','low','close','volume','timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    benchmark, assets = calculate_returns(df)


    #general statistics
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='ret_daily').dropna()
    benchmark_returns = benchmark.set_index('timestamp')['ret_daily']
    asset_metrics = summary_statistics(asset_returns, benchmark_returns)
    print(f"General Statistics: \n {stats}")


    weights = np.array([0.2,0.2,0.2,0.2,0.2])
    portfolio_vol_annual, benchmark_vol_annual, tracking_error, rolling_vol_annual = calculate_vol_tracking(assets, benchmark, weights)
    portfolio_sharpe, portfolio_sortino, bench_sharpe, bench_sortino, information_ratio = calculate_risk_metrics(assets, benchmark, weights)
    max_drawdown_port, max_drawdown_bench= calculate_drawdown(assets, benchmark, weights)
    beta_port = calculate_beta(assets, benchmark, weights)
    portfolio_var, portfolio_cvar, portfolio_var_annual, portfolio_cvar_annual = calculate_var(assets, benchmark, weights)
    port_folio_skew, port_folio_kurt, bench_skew, bench_kurt = calculate_distribution_metrics(assets, benchmark, weights)
    corr_matrix = calculate_corr_metrics(assets, benchmark, weights)
    jensens_alpha, r_squared, alpha_skew, alpha_kurt, mean_alpha_stress = calculate_alpha_metrics(assets, benchmark, weights)
    
    portfolio_metrics = {
        "sharpe":             float(portfolio_sharpe),
        "sortino":            float(portfolio_sortino),
        "information_ratio":  float(information_ratio),
        "max_drawdown_port":  float(max_drawdown_port),
        "beta":               float(beta_port),
        "var_1d":             float(portfolio_var),
        "cvar_1d":            float(portfolio_cvar),
        "var_annual":         float(portfolio_var_annual),
        "cvar_annual":        float(portfolio_cvar_annual),
        "tracking_error":     float(tracking_error),
        "jensens_alpha":      float(jensens_alpha),
        "r_squared":          float(r_squared),
        "mean_alpha_stress":  float(mean_alpha_stress),
        "alpha_skew":         float(alpha_skew),
        "alpha_kurt":         float(alpha_kurt)
    }

    benchmark_metrics = {
        "symbol":             "NIFTY",
        "sharpe":             float(bench_sharpe),
        "sortino":            float(bench_sortino),
        "max_drawdown_bench":  float(max_drawdown_bench),
        "bench_skew":         float(bench_skew),
        "bench_kurt":         float(bench_kurt)
    }

    print("\nVolatility metrics: ")
    print(f"Portfolio Vol:{portfolio_vol_annual*100:.2f}" )
    print(f"Benchmark Vol:{benchmark_vol_annual*100:.2f}" )
    print(f"Rolling Vol:{rolling_vol_annual*100:.2f}" )
    print(f"Tracking Error:{tracking_error*100:.2f}" )

    print("\nRisk metrics: ")
    print("Portfolio Sharpe:", portfolio_sharpe)
    print("Portfolio Sortino:", portfolio_sortino)
    print("Benchmark Sharpe:", bench_sharpe)
    print("Benchmark Sortino:", bench_sortino)
    print("Information Ratio:", information_ratio)

    print("\nDrawdown metrics:")
    print("Portfolio Max Drawdown:", max_drawdown_port * 100)
    print("Benchmark Max Drawdown:", max_drawdown_bench * 100)

    print("\nRisk Sensitivity metrics:")
    print("Portfolio Beta:", beta_port)

    print("\nValue at Risk metrics:")
    print("Portfolio VAR:", portfolio_var * 100)
    print("Portfolio CVAR:", portfolio_cvar * 100)
    print("Portfolio VAR Annual:", portfolio_var_annual * 100)
    print("Portfolio CVAR Annual:", portfolio_cvar_annual * 100)

    print("\nDistribution metrics:")
    print("Portfolio Skew:", port_folio_skew)
    print("Portfolio Kurt:", port_folio_kurt)
    print("Benchmark Skew:", bench_skew)
    print("Benchmark Kurt:", bench_kurt)

    print("\nCorrelation metrics:")
    print(corr_matrix)

    print("\nAlpha metrics:")
    print("Jensen's Alpha (Daily):", jensens_alpha)
    print("R-squared:", r_squared)
    print("Alpha Skewness:", alpha_skew)
    print("Alpha Kurtosis:", alpha_kurt)
    print("Mean Alpha on Stress Days:", mean_alpha_stress)

    return asset_metrics, benchmark_metrics, portfolio_metrics, corr_matrix

if __name__ == "__main__":
    asset_metrics, benchmark_metrics, portfolio_metrics, corr_matrix = main()
    store_asset_metrics(asset_metrics)
    store_benchmark_metrics(benchmark_metrics)
    store_portfolio_metrics(portfolio_metrics)
    store_corr_matrix(corr_matrix)
