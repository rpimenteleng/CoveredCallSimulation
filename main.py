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
        logging.info(f"Fetching SPY data from {self.start_date} to {self.end_date}")
        df = yf.download(
            'SPY', start=self.start_date, end=self.end_date,
            progress=False, auto_adjust=False
        )
        if 'Adj Close' not in df.columns:
            raise ValueError(
                "Expected 'Adj Close' column in fetched data. "
                "Ensure auto_adjust=False in yfinance.download()."
            )
        df_price = df[['Adj Close']].copy()
        df_price.columns = ['price']
        df_price.index = pd.to_datetime(df_price.index)
        logging.info(f"Retrieved {len(df_price)} trading days")
        return df_price

    def run(self, delta_prot_call, delta_prot_put, prot_expiration_days, delta_0dte):
        if self.verbose:
            logging.info(
                f"Starting run: prot_call={delta_prot_call:.2f}, prot_put={delta_prot_put:.2f}, "
                f"prot_days={prot_expiration_days}, delta_0dte={delta_0dte:.2f}"
            )

        cash = self.initial_capital
        shares = 0
        premiums = 0
        prot_expires = None
        prot_positions = {'call': None, 'put': None}
        ledger = []

        for i, (date, row) in enumerate(self.data.iterrows(), start=1):
            price = row['price']

            # Roll protective positions at expiration
            if prot_expires and date >= prot_expires:
                if self.verbose:
                    logging.info(f"Protective options expired on {date.date()}, rolling over")
                prot_expires = None
                prot_positions = {'call': None, 'put': None}

            # Buy underlying if not held
            if shares == 0:
                lot_qty = int(cash // (100 * price))
                buy_shares = lot_qty * 100
                if buy_shares > 0:
                    cash -= buy_shares * price
                    shares = buy_shares
                    if self.verbose:
                        logging.info(f"Bought {shares} shares @ {price:.2f}, cash {cash:.2f}")

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
                cash -= (call_price + put_price) * lot_qty
                prot_positions = {
                    'call': {'strike': K_call, 'price': call_price},
                    'put': {'strike': K_put, 'price': put_price}
                }
                prot_expires = date + timedelta(days=int(prot_expiration_days))
                if self.verbose:
                    logging.info(f"Hedge: call@(K={K_call:.2f},p={call_price:.2f}), put@(K={K_put:.2f},p={put_price:.2f}), exp={prot_expires.date()}")

            # Write 0DTE call if holding shares
            if shares > 0:
                T0 = 1 / 252
                K0 = price * np.exp(self.volatility * np.sqrt(T0) * np.sign(delta_0dte - 0.5))
                c_price = price_option_crank_michelson(price, K0, T0, self.risk_free_rate,
                                                      self.dividend_yield, self.volatility, 'call')
                lot_qty = shares // 100
                cash += c_price * lot_qty
                premiums += c_price * lot_qty
                if self.verbose and i % 20 == 0:
                    logging.info(f"Day {i}: wrote 0DTE call@(K={K0:.2f},p={c_price:.2f}), portf {cash + shares*price + premiums:.2f}")
                # Assignment check

                if price > K0:
                    cash += shares * K0
                    shares = 0
                    if self.verbose:
                        logging.info(f"Assigned @ K0={K0:.2f}, cash {cash:.2f}")

            total_value = cash + shares * price + premiums
            ledger.append({'date': date, 'cash': cash, 'shares': shares,
                           'premiums': premiums, 'total_value': total_value})

        return pd.DataFrame(ledger)


def main(verbose=False):
    setup_logging(logging.INFO if verbose else logging.WARNING)
    backtester = CoveredCallBacktester('2020-05-01', '2025-05-01', verbose=verbose)
    deltas = np.arange(0.1, 1.01, 0.1)
    prot_days = np.arange(30, 181, 30)
    param_grid = list(product(deltas, deltas, prot_days, deltas))
    results = []
    logging.info(f"Running {len(param_grid)} simulations...")
    for idx, (d_pc, d_pp, pdays, d0) in enumerate(tqdm(param_grid), start=1):
        df_ledger = backtester.run(d_pc, d_pp, pdays, d0)
        results.append({
            'delta_prot_call': d_pc,
            'delta_prot_put': d_pp,
            'prot_expiration_days': pdays,
            'delta_0dte': d0,
            'final_value': df_ledger['total_value'].iloc[-1],
            'volatility': df_ledger['total_value'].pct_change().std()
        })
        if idx % 500 == 0:
            logging.info(f"{idx}/{len(param_grid)} completed")
    df_results = pd.DataFrame(results)
    top5 = df_results.sort_values(['final_value', 'volatility'], ascending=[False, True]).head(5)
    print(top5)

# Minimal test case
def test_run():
    dates = pd.date_range('2021-01-01', periods=5)
    df_test = pd.DataFrame({'price': 100.0}, index=dates)
    tester = CoveredCallBacktester('2021-01-01', '2021-01-05', verbose=False)
    tester.data = df_test
    ledger = tester.run(0.5, 0.5, 30, 0.5)
    assert len(ledger) == 5
    print("test_run OK")

if __name__ == '__main__':
    test_run()
    main(verbose=True)
