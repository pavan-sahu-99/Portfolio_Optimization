# database.py
import sqlite3
import pandas as pd
import os

db_path = "data/analytics.db"

# Create tables

def create_asset_metrics_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS asset_metrics (
            symbol         TEXT PRIMARY KEY,
            mean_daily_ret  REAL,
            std_dev_daily   REAL,
            annual_return   REAL,
            annual_vol      REAL,
            skewness        REAL,
            kurtosis        REAL,
            min_return      REAL,
            max_return      REAL,
            sharpe          REAL,
            sortino         REAL,
            beta            REAL
        );
    """)
    conn.commit()
    conn.close()

def create_portfolio_metrics_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_metrics (
            sharpe              REAL,
            sortino             REAL,
            information_ratio   REAL,
            max_drawdown_port   REAL,
            beta                REAL,
            var_1d              REAL,
            cvar_1d             REAL,
            var_annual          REAL,
            cvar_annual         REAL,
            tracking_error      REAL,
            jensens_alpha       REAL,
            r_squared           REAL,
            mean_alpha_stress   REAL,
            alpha_skew          REAL,
            alpha_kurt          REAL
        );
    """)
    conn.commit()
    conn.close()

def create_benchmark_metrics_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_metrics (
            symbol         TEXT PRIMARY KEY,
            sharpe              REAL,
            sortino             REAL,
            max_drawdown_bench  REAL,
            bench_skew          REAL,
            bench_kurt          REAL
        );
    """)
    conn.commit()
    conn.close()

def create_optimizer_results_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimizer_results (
            symbol          TEXT PRIMARY KEY,
            optimal_weight  REAL,
            equal_weight    REAL,
            opt_port_return      REAL,
            opt_port_vol         REAL,
            opt_port_sharpe      REAL,
            eq_port_return       REAL,
            eq_port_vol          REAL,
            eq_port_sharpe       REAL
        );
    """)
    conn.commit()
    conn.close()

def create_projections_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projections (
            date        TEXT PRIMARY KEY UNIQUE,
            forecast    REAL,
            lower_ci    REAL,
            upper_ci    REAL
        );
    """)
    conn.commit()
    conn.close()

def create_corr_matrix_table(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corr_matrix (
            symbol1     TEXT NOT NULL,
            symbol2     TEXT NOT NULL,
            correlation REAL,
            UNIQUE(symbol1, symbol2)
        );
    """)
    conn.commit()
    conn.close()

# Insert data into tables

def store_asset_metrics(stats_df):
    df = stats_df.T.reset_index()
    df.columns = [
        "symbol", "mean_daily_ret", "std_dev_daily",
        "annual_return", "annual_vol", "skewness",
        "kurtosis", "min_return", "max_return",
        "sharpe", "sortino", "beta"
    ]
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM asset_metrics")
    df.to_sql("asset_metrics", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("Stored: asset_metrics")

def store_portfolio_metrics(metrics):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM portfolio_metrics")
    pd.DataFrame([metrics]).to_sql("portfolio_metrics", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("Stored: portfolio_metrics")

def store_benchmark_metrics(metrics):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM benchmark_metrics")
    pd.DataFrame([metrics]).to_sql("benchmark_metrics", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("Stored: benchmark_metrics")

def store_optimizer_results(weights_series, opt_return, opt_vol, opt_sharpe,
                             eq_return, eq_vol, eq_sharpe):
    df = pd.DataFrame({
        "symbol":          weights_series.index,
        "optimal_weight":  weights_series.values,
        "equal_weight":    [1 / len(weights_series)] * len(weights_series),
        "opt_port_return": opt_return,    # ← match table schema exactly
        "opt_port_vol":    opt_vol,
        "opt_port_sharpe": opt_sharpe,
        "eq_port_return":  eq_return,
        "eq_port_vol":     eq_vol,
        "eq_port_sharpe":  eq_sharpe
    })
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM optimizer_results")
    df.to_sql("optimizer_results", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("Stored: optimizer_results")

def store_projections(forecast_df):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM projections")
    forecast_df.reset_index().rename(columns={'index': 'date'}).to_sql("projections", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

def store_corr_matrix(corr_matrix):
    rows = []
    for symbol1 in corr_matrix.index:
        for symbol2 in corr_matrix.columns:
            rows.append({
                "symbol1":     symbol1,
                "symbol2":     symbol2,
                "correlation": float(corr_matrix.loc[symbol1, symbol2])
            })
    df = pd.DataFrame(rows)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM corr_matrix")
    df.to_sql("corr_matrix", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print("Stored: corr_matrix")

# Read data from tables

def read_table(table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def create_all_tables(db_path):
    create_benchmark_metrics_table(db_path)
    create_asset_metrics_table(db_path)
    create_portfolio_metrics_table(db_path)
    create_optimizer_results_table(db_path)
    create_projections_table(db_path)
    create_corr_matrix_table(db_path)

if __name__ == "__main__":
    create_all_tables(db_path)
    for table in ["asset_metrics","benchmark_metrics", "portfolio_metrics",
                  "optimizer_results", "projections", "corr_matrix"]:
        try:
            df = read_table(table)
            print(f"\n{table} ({len(df)} rows):")
            print(df.head())
        except Exception as e:
            print(f"{table} error: {e}")


