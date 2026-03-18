# optimized portfolio to get best returns
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .database import store_optimizer_results

'''
We've seen that the equal weights we took gave a sharpe of 0.6 where as NIFTY is 0.97
which means for every 1 re risk we get only 0.56 re returns.
So there is need to optimize this
'''

def mean_variance_optimizer(returns):

    mean_returns = returns.mean().values * 252
    cov_matrix   = returns.cov().values * 252
    n = len(mean_returns)

    def neg_sharpe(weights):
        weights = np.array(weights)
        port_return = np.dot(weights, mean_returns)
        port_vol    = np.sqrt(weights @ cov_matrix @ weights)
        return -(port_return / port_vol) # -ve to keep max sharpe ratio

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} # sum of weights = 1  
    bounds = tuple((0.0, 1.0) for _ in range(n)) # each weights are between 0 - 1
    init_weights = np.array([1.0 / n] * n) # initial weights

    result = minimize(
        neg_sharpe,
        init_weights,
        method='SLSQP', # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000} # stops when function value changes by less than 1e-12
    )

    if not result.success:
        raise ValueError(f"Optimisation failed: {result.message}")

    optimal_weights = result.x
    opt_return = float(np.dot(optimal_weights, mean_returns))
    opt_vol    = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
    opt_sharpe = opt_return / opt_vol

    weights_series = pd.Series(optimal_weights, index=returns.columns)
    return weights_series, opt_return, opt_vol, opt_sharpe


def plot_efficient_frontier(returns, optimal_weights, opt_return, opt_vol):
    mean_returns = returns.mean().values * 252
    cov_matrix   = returns.cov().values * 252
    tickers = returns.columns.tolist()
    n = len(mean_returns)

    # Monte Carlo generates 5000 random portfolios
    mc_vols, mc_rets, mc_sharpes = [], [], []
    np.random.seed(42)

    for _ in range(5000):
        w = np.random.dirichlet(np.ones(n)) #random weights summing to one
        r = w @ mean_returns # portfolio return
        v = np.sqrt(w @ cov_matrix @ w) # portfolio volatility
        mc_vols.append(v)
        mc_rets.append(r)
        mc_sharpes.append(r / v) # portfolio sharpe ratio

    # Efficient Frontier curve : minimize variance weights sums to 1 for each 200 target returns connecting all point traces blue line
    target_returns = np.linspace(min(mc_rets), max(mc_rets), 200)
    frontier_vols, frontier_rets = [], []

    for target in target_returns:
        def port_variance(w):
            return w @ cov_matrix @ w

        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: w @ mean_returns - t}
        ]
        res = minimize(port_variance,
                       np.ones(n) / n,
                       method='SLSQP',
                       bounds=tuple((0, 1) for _ in range(n)),
                       constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 1000})
        if res.success:
            frontier_vols.append(np.sqrt(res.fun))
            frontier_rets.append(target)

    # Equal weight --> 0.2 each
    eq_w   = np.ones(n) / n
    eq_ret = eq_w @ mean_returns
    eq_vol = np.sqrt(eq_w @ cov_matrix @ eq_w)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Random portfolios --> coloured by Sharpe
    sc = ax.scatter(mc_vols, mc_rets,
                    c=mc_sharpes, cmap='viridis',
                    alpha=0.3, s=8, label='Random Portfolios')
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    # Efficient Frontier line
    ax.plot(frontier_vols, frontier_rets,
            'b-', linewidth=2.5, label='Efficient Frontier')

    # Max Sharpe portfolio --> red star
    ax.scatter(opt_vol, opt_return,
               marker='*', color='red', s=400,
               zorder=5, label=f'Max Sharpe ({opt_return/opt_vol:.2f})')

    # Equal weight portfolio --> orange diamond
    ax.scatter(eq_vol, eq_ret,
               marker='D', color='orange', s=120,
               zorder=5, label=f'Equal Weight (Sharpe: {eq_ret/eq_vol:.2f})')

    # Individual stocks --> labelled
    for i, ticker in enumerate(tickers):
        sv = np.sqrt(cov_matrix[i, i])
        sr = mean_returns[i]
        ax.scatter(sv, sr, s=60, zorder=5, color='grey')
        ax.annotate(ticker, (sv, sr),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.set_xlabel('Annualised Volatility', fontsize=12)
    ax.set_ylabel('Annualised Return', fontsize=12)
    ax.set_title('Efficient Frontier --> Mean Variance Optimisation', fontsize=14)
    ax.legend(fontsize=10)


    fig.savefig("graphs/efficient_frontier.png", bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: efficient_frontier.png")

if __name__ == "__main__":
    df = pd.read_csv("data\\historical_data.csv")
    df = df[['symbol','open','high','low','close','volume','timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df = df.dropna().reset_index(drop=True)
    assets = df[df['symbol'] != 'NIFTY'].reset_index(drop=True)
    benchmark = df[df['symbol'] == 'NIFTY'].reset_index(drop=True)
    asset_returns = assets.pivot(index='timestamp',columns='symbol',values='returns').dropna()
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=asset_returns.columns)
    eq_weight_port_returns = asset_returns.dot(weights)
    eq_weight_annual_returns = eq_weight_port_returns.mean() * 252
    eq_weight_annual_volatility = eq_weight_port_returns.std() * np.sqrt(252)
    eq_weight_sharpe = eq_weight_annual_returns/eq_weight_annual_volatility

    
    weights, ret, vol, sharpe = mean_variance_optimizer(asset_returns)
    print("\nOptimal Weights:")
    print(weights.round(4)*100)
    print("\nBefore Optimisation: ")
    print(f"\nAnnualised Return: {eq_weight_annual_returns*100:.2f}%")
    print(f"Annualised Vol: {eq_weight_annual_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {eq_weight_sharpe:.4f}")
    print("\nAfter Optimisation: ")
    print(f"\nAnnualised Return: {ret*100:.2f}%")
    print(f"Annualised Vol: {vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")

    store_optimizer_results(weights, ret, vol, sharpe,
                        eq_weight_annual_returns,
                        eq_weight_annual_volatility,
                        eq_weight_sharpe)

    plot_efficient_frontier(asset_returns, weights, ret, vol)
