import numpy as np
import pandas as pd
import yfinance as yf
import logging
from itertools import product
from tqdm import tqdm
from datetime import timedelta

# Configure logging for verbosity
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level
    )

# Placeholder for the Crank-Michelson PDE solver for option pricing
def price_option_crank_michelson(S, K, T, r, q, sigma, option_type='call'):
    """
    Price European option using Crank-Michelson (Crank-Nicolson) PDE method.
    Falls back to Black–Scholes until PDE implementation.
    """
    from scipy.stats import norm
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

class CoveredCallBacktester:
    def __init__(
        self, start_date, end_date,
        capital=100000, dividend_yield=0.018,
        risk_free_rate=0.03, volatility=0.15,
        verbose=False
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = capital
        self.dividend_yield = dividend_yield
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.verbose = verbose
        self.data = self._fetch_data()

    def _fetch_data(self):
        logging.info(f"Fetching SPY OHLC data from {self.start_date} to {self.end_date}")
        df = yf.download(
            'SPY', start=self.start_date, end=self.end_date,
            progress=False, auto_adjust=False
        )
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise ValueError(
                "Expected OHLC columns in fetched data. "
                "Ensure yfinance.download() returns Open, High, Low, Close."
            )
        df_ohlc = df[['Open', 'High', 'Low', 'Close', 'Adj Close']].copy()
        df_ohlc.columns = ['open', 'high', 'low', 'close', 'adj_close']
        df_ohlc.index = pd.to_datetime(df_ohlc.index)
        logging.info(f"Retrieved {len(df_ohlc)} trading days with OHLC data")
        return df_ohlc

    def run(self, delta_prot_call, delta_prot_put, prot_expiration_days, delta_0dte):
        if self.verbose:
            logging.info(
                f"Starting run: prot_call={delta_prot_call:.2f}, prot_put={delta_prot_put:.2f}, "
                f"prot_days={prot_expiration_days}, delta_0dte={delta_0dte:.2f}"
            )

        cash = self.initial_capital
        shares = 0
        total_premiums = 0
        prot_expires = None
        prot_positions = {'call': None, 'put': None}
        ledger = []
        action_ledger = []  # Detailed action log

        for i, (date, row) in enumerate(self.data.iterrows(), start=1):
            open_price = row['open']
            high_price = row['high']
            close_price = row['close']
            price = close_price  # For portfolio value, use close
            lot_qty = shares // 100 if shares > 0 else int(cash // (100 * open_price))
            # Roll protective positions at expiration
            if prot_expires and date >= prot_expires:
                prot_expires = None
                prot_positions = {'call': None, 'put': None}
                action_ledger.append({
                    'date': date, 'price': price, 'transaction': 'Protective Options Expire', 'lots': lot_qty,
                    'option_price': None, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                    'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                })

            # Buy underlying if not held
            if shares == 0:
                lot_qty = int(cash // (100 * open_price))
                buy_shares = lot_qty * 100
                if buy_shares > 0:
                    cost = buy_shares * open_price
                    cash -= cost
                    shares = buy_shares
                    action_ledger.append({
                        'date': date, 'price': open_price, 'transaction': 'Purchase Underlying', 'lots': lot_qty,
                        'option_price': None, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                        'P/L': -cost, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                    })

            # Set up protective options if none and shares exist
            if shares > 0 and prot_positions['call'] is None:
                T = prot_expiration_days / 252
                K_call = price * np.exp(self.volatility * np.sqrt(T) * np.sign(delta_prot_call - 0.5))
                K_put = price * np.exp(-self.volatility * np.sqrt(T) * np.sign(delta_prot_put - 0.5))
                call_price = price_option_crank_michelson(price, K_call, T, self.risk_free_rate,
                                                       self.dividend_yield, self.volatility, 'call')
                put_price = price_option_crank_michelson(price, K_put, T, self.risk_free_rate,
                                                     self.dividend_yield, self.volatility, 'put')
                lot_qty = shares // 100
                call_premium = call_price * 100 * lot_qty
                put_premium = put_price * 100 * lot_qty
                cost = call_premium + put_premium
                cash -= cost
                prot_positions = {
                    'call': {'strike': K_call, 'price': call_price},
                    'put': {'strike': K_put, 'price': put_price}
                }
                prot_expires = date + timedelta(days=int(prot_expiration_days))
                action_ledger.append({
                    'date': date, 'price': price, 'transaction': 'Buy Protective Call', 'lots': lot_qty,
                    'option_price': call_price, 'premium_received': 0, 'premium_paid': call_premium, 'total_premiums': total_premiums - call_premium,
                    'P/L': -call_premium, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums - call_premium
                })
                total_premiums -= call_premium
                action_ledger.append({
                    'date': date, 'price': price, 'transaction': 'Buy Protective Put', 'lots': lot_qty,
                    'option_price': put_price, 'premium_received': 0, 'premium_paid': put_premium, 'total_premiums': total_premiums - put_premium,
                    'P/L': -put_premium, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums - put_premium
                })
                total_premiums -= put_premium

            # Write 0DTE call if holding shares
            if shares > 0:
                T0 = 1 / 252
                K0 = price * np.exp(self.volatility * np.sqrt(T0) * np.sign(delta_0dte - 0.5))
                c_price = price_option_crank_michelson(price, K0, T0, self.risk_free_rate,
                                                      self.dividend_yield, self.volatility, 'call')
                lot_qty = shares // 100
                callwrite_premium = c_price * 100 * lot_qty
                cash += callwrite_premium
                total_premiums += callwrite_premium
                action_ledger.append({
                    'date': date, 'price': price, 'transaction': 'Call Writing', 'lots': lot_qty,
                    'option_price': c_price, 'premium_received': callwrite_premium, 'premium_paid': 0, 'total_premiums': total_premiums,
                    'P/L': callwrite_premium, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                })
                # Assignment check
                if high_price > K0:  # Assignment only if high exceeds strike
                    proceeds = shares * K0  # Assignment at strike price
                    cash += proceeds
                    action_ledger.append({
                        'date': date, 'price': K0, 'transaction': 'Assignment', 'lots': lot_qty,
                        'option_price': K0, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                        'P/L': proceeds, 'cash': cash, 'shares': 0, 'total_value': cash + total_premiums
                    })
                    shares = 0

            total_value = cash + shares * price + total_premiums
            ledger.append({'date': date, 'cash': cash, 'shares': shares,
                           'total_premiums': total_premiums, 'total_value': total_value})

        return pd.DataFrame(ledger), pd.DataFrame(action_ledger)

def main(verbose=False):
    setup_logging(logging.INFO if verbose else logging.WARNING)
    backtester = CoveredCallBacktester('2020-05-01', '2025-05-01', verbose=verbose)
    # Use larger steps to reduce the number of simulations
    deltas = np.arange(0.1, 1.01, 0.2)  # Step size 0.2 instead of 0.1
    prot_days = np.arange(30, 181, 60)   # Step size 60 instead of 30
    param_grid = list(product(deltas, deltas, prot_days, deltas))
    results = []
    action_ledgers = []
    logging.info(f"Running {len(param_grid)} simulations...")
    for idx, (d_pc, d_pp, pdays, d0) in enumerate(tqdm(param_grid), start=1):
        df_ledger, df_action_ledger = backtester.run(d_pc, d_pp, pdays, d0)
        results.append({
            'delta_prot_call': d_pc,
            'delta_prot_put': d_pp,
            'prot_expiration_days': pdays,
            'delta_0dte': d0,
            'final_value': df_ledger['total_value'].iloc[-1],
            'volatility': df_ledger['total_value'].pct_change().std(),
            'action_ledger': df_action_ledger
        })
        if idx % 100 == 0:
            logging.info(f"{idx}/{len(param_grid)} completed")
    df_results = pd.DataFrame(results)
    top5 = df_results.sort_values(['final_value', 'volatility'], ascending=[False, True]).head(5)
    print(top5[[col for col in top5.columns if col != 'action_ledger']])
    # Export top 5 action ledgers to XLSX
    for i, row in enumerate(top5.itertuples(), 1):
        df_action = row.action_ledger.copy()
        df_action = df_action.sort_values('date')
        # Create a full date range for the simulation
        full_dates = pd.date_range(backtester.start_date, backtester.end_date, freq='B')
        # Fetch SPY prices for the full date range and ensure columns are ['date', 'price']
        spy_prices = backtester.data.reindex(full_dates).copy().reset_index()
        spy_prices.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close']
        # Merge action ledger with full date range and SPY prices
        merged = pd.DataFrame({'date': full_dates})
        merged = merged.merge(spy_prices[['date', 'open', 'high', 'low', 'close']], on='date', how='left')
        merged = merged.merge(df_action, on='date', how='left')
        # Ensure 'open' column exists
        if 'open' not in merged.columns:
            merged['open'] = np.nan
        # Only include columns that exist
        base_cols = ['date', 'open', 'high', 'low', 'close']
        extra_cols = [c for c in ['transaction', 'lots', 'option_price', 'premium_received', 'premium_paid', 'total_premiums', 'P/L', 'cash', 'shares', 'total_value'] if c in merged.columns]
        other_cols = [c for c in merged.columns if c not in base_cols + extra_cols]
        cols = base_cols + extra_cols + other_cols
        merged = merged[cols]
        fname = f"simulation_{i}_final_{row.final_value:.2f}.xlsx"
        merged.to_excel(fname, index=False)
        print(f"Exported {fname}")

# Minimal test case
def test_run():
    dates = pd.date_range('2021-01-01', periods=5)
    df_test = pd.DataFrame({
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'adj_close': 100.5
    }, index=dates)
    tester = CoveredCallBacktester('2021-01-01', '2021-01-05', verbose=False)
    tester.data = df_test
    ledger, action_ledger = tester.run(0.5, 0.5, 30, 0.5)
    assert len(ledger) == 5
    print("test_run OK")

if __name__ == '__main__':
    test_run()
    main(verbose=True)
