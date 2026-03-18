
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import numpy as np
from .database import read_table
from .forecasting import forecast_portfolio_returns, returns_distribution


def plot_corr(corr_matrix):
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',ax=ax)
    ax.set_title('Correlation Matrix — Pairwise Return Correlations', fontsize=13)
    ax.set_xlabel("Asset", fontsize=12)
    ax.set_ylabel("Asset", fontsize=12)
    plt.tight_layout()
    plt.savefig('graphs/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_cumulative_returns(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns):
    initial_value = 10000
    fig, ax = plt.subplots(figsize=(13, 6))
    (initial_value * (1 + eq_portfolio_returns).cumprod()).plot(ax=ax, label="Equal Weight Portfolio", color='blue', linewidth=1.5)
    (initial_value * (1 + optimized_portfolio_returns).cumprod()).plot(ax=ax, label="Optimized Portfolio", color='green', linewidth=1.5, linestyle="-.")
    (initial_value * (1 + benchmark_returns).cumprod()).plot(ax=ax, label="Benchmark", color='red', linewidth=1.5, linestyle="--")
    ax.set_title("Cumulative Returns — Portfolio vs Benchmark", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_drawdown(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns):  
    def compute_drawdown(returns):
        cum = (1 + returns).cumprod()
        return (cum - cum.cummax()) / cum.cummax()

    dd_eq = compute_drawdown(eq_portfolio_returns)
    dd_opt = compute_drawdown(optimized_portfolio_returns)
    dd_bench = compute_drawdown(benchmark_returns)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    def get_plot(ax, dd, label, color):
        dd.plot(ax=ax, color=color, linewidth=1.2)
        ax.fill_between(dd.index, dd, 0, alpha=0.3, color=color)

        max_dd = dd.min()
        max_dd_date = dd.idxmin()

        ax.axhline(max_dd, color=color, linewidth=0.8, linestyle=":")
        ax.annotate(f"Max Drawdown: {max_dd:.1%}",
                    xy=(max_dd_date, max_dd),
                    xytext=(10, -15),
                    textcoords="offset points",
                    fontsize=9, color=color)

        ax.set_ylabel("Drawdown", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend([label], fontsize=10, loc="lower left")
        ax.set_ylim(bottom=max_dd * 1.3)

    get_plot(ax1, dd_eq, "Equal Weight Portfolio", "blue")
    get_plot(ax2, dd_opt, "Optimised Portfolio", "green")
    get_plot(ax3, dd_bench, "Benchmark (NIFTY)", "red")

    fig.suptitle("Drawdown Over Time", fontsize=14, fontweight='bold', y=1.01)
    ax3.set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.savefig('graphs/drawdown.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_rolling_volatility(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns):
    window = 21
    eq_roll    = eq_portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    opt_roll   = optimized_portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    bench_roll = benchmark_returns.rolling(window).std() * np.sqrt(252) * 100
    
    fig, ax = plt.subplots(figsize=(13, 6))
    eq_roll.plot(ax=ax, label="Equal Weight Portfolio", color='blue', linewidth=1.5)
    opt_roll.plot(ax=ax, label="Optimised Portfolio", color='green', linewidth=1.5, linestyle="-.")
    bench_roll.plot(ax=ax, label="Benchmark (NIFTY)", color='red', linewidth=1.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Day Annualised Volatility", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Annualised Volatility (%)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/rolling_volatility.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_var(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns):
    
    fig, ax = plt.subplots(figsize=(13, 6))
    eq_returns_pct = eq_portfolio_returns * 100
    opt_returns_pct = optimized_portfolio_returns * 100
    bench_returns_pct = benchmark_returns * 100
    sns.kdeplot(eq_returns_pct, ax=ax, label="Equal Weight Portfolio", color='blue', fill=False, linewidth=1.5, alpha=0.5)
    sns.kdeplot(opt_returns_pct, ax=ax, label="Optimised Portfolio", color='green', fill=False, linewidth=1.5, alpha=0.5, linestyle="-.")
    sns.kdeplot(bench_returns_pct, ax=ax, label="Benchmark (NIFTY)", color='red', fill=False, linewidth=1.5, alpha=0.5, linestyle="--")
    ax.set_title("Return Distribution", fontsize=13)
    ax.set_xlabel("Daily Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/return_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_weights(eq_weights, optimized_weights):
    df = pd.DataFrame({
        'Symbol': eq_weights.index,
        'Equal Weight': eq_weights.values,
        'Optimized Weight': optimized_weights.values
    }).melt(id_vars='Symbol', var_name='Portfolio', value_name='Weight')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Symbol', y='Weight', hue='Portfolio', ax=ax)
    ax.set_title("Portfolio Weights Comparison", fontsize=13)
    ax.set_xlabel("Asset Symbol", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.legend(title="")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graphs/portfolio_weights.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rolling_sharpe(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns, risk_free_rate=0):
    window = 21
    eq_sharpe = (eq_portfolio_returns.rolling(window).mean() * 252 - risk_free_rate) / (eq_portfolio_returns.rolling(window).std() * np.sqrt(252))
    opt_sharpe = (optimized_portfolio_returns.rolling(window).mean() * 252 - risk_free_rate) / (optimized_portfolio_returns.rolling(window).std() * np.sqrt(252))
    bench_sharpe = (benchmark_returns.rolling(window).mean() * 252 - risk_free_rate) / (benchmark_returns.rolling(window).std() * np.sqrt(252))

    fig, ax = plt.subplots(figsize=(13, 6))
    eq_sharpe.plot(ax=ax, label="Equal Weight Portfolio", color='blue', linewidth=1.5)
    opt_sharpe.plot(ax=ax, label="Optimised Portfolio", color='green', linewidth=1.5, linestyle="-.")
    bench_sharpe.plot(ax=ax, label="Benchmark (NIFTY)", color='red', linewidth=1.5, linestyle="--")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/rolling_sharpe.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_forecast(portfolio_returns, forecast_df, initial_capital=10000):
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    last_value = portfolio_value.iloc[-1]
    last_date = portfolio_value.index[-1]
    forecast_value = last_value * (1 + forecast_df['forecast']).cumprod()
    lower_value = last_value * (1 + forecast_df['lower_ci']).cumprod()
    upper_value = last_value * (1 + forecast_df['upper_ci']).cumprod()

    fig, ax = plt.subplots(figsize=(13, 6))
    portfolio_value.plot(ax=ax, label="Historical Portfolio Value", color='blue', linewidth=1.5)
    forecast_value.plot(ax=ax, label="Forecasted Portfolio Value", color='green', linewidth=1.5, linestyle="--")
    ax.fill_between(forecast_df.index, lower_value, upper_value, color='green', alpha=0.2, label="95% Confidence Interval")
    ax.axvline(last_date, color='gray', linestyle=":", linewidth=1)
    ax.annotate(f"Rs {forecast_value.iloc[-1]:,.0f}", xy=(forecast_df.index[-1], forecast_value.iloc[-1]), xytext=(10, 10), textcoords="offset points", fontsize=9, color='gray')
    ax.set_title("Portfolio Value Forecast for 90 days", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (Rs)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('graphs/portfolio_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    #correlation matrix
    #corr_matrix = read_table('corr_matrix')
    #corr_matrix = corr_matrix.pivot(index='symbol1', columns='symbol2', values='correlation')
    #print(corr_matrix)
    #plot_corr(corr_matrix)
    
    #cumulative returns
    optimizer_results = read_table('optimizer_results')
    eq_weights = pd.Series(optimizer_results['equal_weight'].values, index=optimizer_results['symbol'])
    optimized_weights = pd.Series(optimizer_results['optimal_weight'].values, index=optimizer_results['symbol'])
    eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns = returns_distribution(eq_weights, optimized_weights)
    plot_cumulative_returns(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns)
    plot_rolling_volatility(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns)
    plot_drawdown(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns)
    plot_var(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns)
    plot_weights(eq_weights, optimized_weights)
    plot_rolling_sharpe(eq_portfolio_returns, optimized_portfolio_returns, benchmark_returns)
    forecast_df = forecast_portfolio_returns(optimized_portfolio_returns)
    plot_forecast(optimized_portfolio_returns, forecast_df)


    
