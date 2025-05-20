import numpy as np
import pandas as pd
import yfinance as yf
import logging
from itertools import product
from tqdm import tqdm
from datetime import timedelta
import math

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
        verbose=False, stock='SPY'
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = capital
        self.dividend_yield = dividend_yield
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.verbose = verbose
        self.stock = stock
        self.data = self._fetch_data()

    def _fetch_data(self):
        logging.info(f"Fetching {self.stock} OHLC data from {self.start_date} to {self.end_date}")
        df = yf.download(
            self.stock, start=self.start_date, end=self.end_date,
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

    def run(self, delta_prot_call, delta_prot_put, prot_expiration_days, delta_0dte, asd=0.0):
        if self.verbose:
            logging.info(
                f"Starting run: prot_call={delta_prot_call:.2f}, prot_put={delta_prot_put:.2f}, "
                f"prot_days={prot_expiration_days}, delta_0dte={delta_0dte:.2f}, asd={asd:.2f}"
            )

        cash = self.initial_capital
        shares = 0
        total_premiums = 0
        prot_expires = None
        prot_positions = {'call': None, 'put': None}
        ledger = []
        action_ledger = []  # Detailed action log
        cost_basis_per_share = None
        lookback = 30  # Number of days for quantile calculation
        event_second = 0  # Used to increment event index for same-day events
        assignment_count = 0
        call_write_count = 0
        def add_action_entry(entry):
            nonlocal event_second
            entry['event_index'] = event_second
            event_second += 1
            action_ledger.append(entry)

        for i, (date, row) in enumerate(self.data.iterrows(), start=1):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            price = close_price  # For portfolio value, use close
            lot_qty = shares // 100 if shares > 0 else int(cash // (100 * open_price))
            # Helper for quantile-based strike
            def round_up_50(x):
                return math.ceil(x * 2) / 2
            # Get lookback window
            if i > lookback:
                ohlc_window = self.data.iloc[i - lookback:i]
                open_to_high = ohlc_window['high'] - ohlc_window['open']
                open_to_low = ohlc_window['open'] - ohlc_window['low']
                # For calls: strike = open + quantile(Open-to-High, delta)
                call_strike_quantile = lambda delta: round_up_50(open_price + np.quantile(open_to_high, delta))
                # For puts: strike = open - quantile(Open-to-Low, delta)
                put_strike_quantile = lambda delta: round_up_50(open_price - np.quantile(open_to_low, delta))
            else:
                # Not enough data, fallback to previous logic
                call_strike_quantile = lambda delta: round_up_50(open_price)
                put_strike_quantile = lambda delta: round_up_50(open_price)

            # Roll protective positions at expiration
            if prot_expires and date >= prot_expires:
                # Realize value for ITM protective options
                itm_call_value = 0
                itm_put_value = 0
                if prot_positions['call'] is not None:
                    call_strike = prot_positions['call']['strike']
                    if price > call_strike:
                        itm_call_value = (price - call_strike) * shares
                        cash += itm_call_value
                        add_action_entry({
                            'date': date, 'price': price, 'transaction': 'Protective Call ITM Expiry', 'lots': lot_qty,
                            'option_price': call_strike, 'Strike Price': call_strike, 'premium_received': itm_call_value, 'premium_paid': 0, 'total_premiums': total_premiums,
                            'Inflow/Outflow': itm_call_value, 'P/L': itm_call_value, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                        })
                if prot_positions['put'] is not None:
                    put_strike = prot_positions['put']['strike']
                    if price < put_strike:
                        itm_put_value = (put_strike - price) * shares
                        cash += itm_put_value
                        add_action_entry({
                            'date': date, 'price': price, 'transaction': 'Protective Put ITM Expiry', 'lots': lot_qty,
                            'option_price': put_strike, 'Strike Price': put_strike, 'premium_received': itm_put_value, 'premium_paid': 0, 'total_premiums': total_premiums,
                            'Inflow/Outflow': itm_put_value, 'P/L': itm_put_value, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                        })
                prot_expires = None
                prot_positions = {'call': None, 'put': None}
                add_action_entry({
                    'date': date, 'price': price, 'transaction': 'Protective Options Expire', 'lots': lot_qty,
                    'option_price': None, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                    'Inflow/Outflow': 0, 'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                })

            # Buy underlying if not held
            if shares == 0:
                lot_qty = int(cash // (100 * open_price))
                buy_shares = lot_qty * 100
                if buy_shares > 0:
                    cost = buy_shares * open_price
                    cash -= cost
                    shares = buy_shares
                    cost_basis_per_share = open_price  # Track cost basis for assignment P/L
                    add_action_entry({
                        'date': date, 'price': open_price, 'transaction': 'Purchase Underlying', 'lots': lot_qty,
                        'option_price': None, 'Strike Price': None, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                        'Inflow/Outflow': -cost, 'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                    })

            # Set up protective options if none and shares exist
            if shares > 0 and prot_positions['call'] is None:
                T = prot_expiration_days / 252
                K_call = call_strike_quantile(delta_prot_call)
                K_put = put_strike_quantile(delta_prot_put)
                # Ensure protective call is OTM (strike > price) and put is OTM (strike < price)
                if K_call <= price:
                    K_call = round_up_50(price + 0.5)
                if K_put >= price:
                    K_put = round_up_50(price - 0.5)
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
                add_action_entry({
                    'date': date, 'price': price, 'transaction': 'Buy Protective Call', 'lots': lot_qty,
                    'option_price': call_price, 'Strike Price': K_call, 'premium_received': 0, 'premium_paid': call_premium, 'total_premiums': total_premiums - call_premium,
                    'Inflow/Outflow': -call_premium, 'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums - call_premium
                })
                total_premiums -= call_premium
                add_action_entry({
                    'date': date, 'price': price, 'transaction': 'Buy Protective Put', 'lots': lot_qty,
                    'option_price': put_price, 'Strike Price': K_put, 'premium_received': 0, 'premium_paid': put_premium, 'total_premiums': total_premiums - put_premium,
                    'Inflow/Outflow': -put_premium, 'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums - put_premium
                })
                total_premiums -= put_premium

            # Write 0DTE call if holding shares
            if shares > 0:
                call_write_count += 1
                T0 = 1 / 252
                K0 = call_strike_quantile(delta_0dte)
                if cost_basis_per_share is not None and K0 < cost_basis_per_share:
                    K0 = cost_basis_per_share
                K0 = round_up_50(K0)
                c_price = price_option_crank_michelson(price, K0, T0, self.risk_free_rate,
                                                      self.dividend_yield, self.volatility, 'call')
                lot_qty = shares // 100
                callwrite_premium = c_price * 100 * lot_qty
                cash += callwrite_premium
                total_premiums += callwrite_premium
                add_action_entry({
                    'date': date, 'price': price, 'transaction': 'Call Writing', 'lots': lot_qty,
                    'option_price': c_price, 'Strike Price': K0, 'premium_received': callwrite_premium, 'premium_paid': 0, 'total_premiums': total_premiums,
                    'Inflow/Outflow': callwrite_premium, 'P/L': 0, 'cash': cash, 'shares': shares, 'total_value': cash + shares * price + total_premiums
                })
                # Assignment check
                if high_price > K0 + asd:  # Assignment only if high exceeds strike + asd
                    assignment_count += 1
                    proceeds = shares * K0  # Assignment at strike price
                    cash += proceeds
                    pl_assignment = proceeds - (shares * cost_basis_per_share if cost_basis_per_share is not None else 0)
                    add_action_entry({
                        'date': date, 'price': K0, 'transaction': 'Assignment', 'lots': lot_qty,
                        'option_price': K0, 'Strike Price': K0, 'premium_received': 0, 'premium_paid': 0, 'total_premiums': total_premiums,
                        'Inflow/Outflow': proceeds, 'P/L': pl_assignment, 'cash': cash, 'shares': 0, 'total_value': cash + total_premiums
                    })
                    shares = 0
                    cost_basis_per_share = None

            total_value = cash + shares * price + total_premiums
            ledger.append({'date': date, 'cash': cash, 'shares': shares,
                           'total_premiums': total_premiums, 'total_value': total_value})

        assignment_rate = assignment_count / call_write_count if call_write_count > 0 else 0
        return pd.DataFrame(ledger), pd.DataFrame(action_ledger), assignment_rate

def main(verbose=False):
    setup_logging(logging.INFO if verbose else logging.WARNING)
    stock = 'SPY'  # You can change this to any stock symbol
    deltas = np.arange(0.1, 1.01, 0.2)  # Step size 0.2 instead of 0.1
    prot_days = np.arange(30, 181, 60)   # Step size 60 instead of 30
    asd = 0.7  # Set your desired assignment sensitivity delta here
    print(f"\nStarting Covered Call Simulation for stock: {stock}")
    print(f"Delta range: {deltas}")
    print(f"Protective expiration days: {prot_days}")
    print(f"Assignment sensitivity delta (asd): {asd}")
    print(f"Date range: 2020-05-01 to 2025-05-01\n")
    backtester = CoveredCallBacktester('2020-05-01', '2025-05-01', verbose=verbose, stock=stock)
    param_grid = list(product(deltas, deltas, prot_days, deltas))
    results = []
    action_ledgers = []
    logging.info(f"Running {len(param_grid)} simulations...")
    for idx, (d_pc, d_pp, pdays, d0) in enumerate(tqdm(param_grid), start=1):
        df_ledger, df_action_ledger, assignment_rate = backtester.run(d_pc, d_pp, pdays, d0, asd=asd)
        # --- BUY & HOLD BENCHMARK ---
        # Buy as many shares as possible at first open, hold to end
        bh_open = backtester.data.iloc[0]['open']
        bh_shares = int(backtester.initial_capital // bh_open)
        bh_cash = backtester.initial_capital - (bh_shares * bh_open)
        bh_values = []
        for date, row in backtester.data.iterrows():
            bh_value = bh_shares * row['close'] + bh_cash
            bh_values.append({'date': date, 'buy_and_hold_value': bh_value})
        df_bh = pd.DataFrame(bh_values)
        # Merge buy & hold value into ledger for Excel export
        df_action_ledger = df_action_ledger.copy()
        df_action_ledger = pd.merge(df_action_ledger, df_bh, on='date', how='left')
        # Also merge for full ledger if needed
        df_ledger = pd.merge(df_ledger, df_bh, on='date', how='left')
        results.append({
            'delta_prot_call': d_pc,
            'delta_prot_put': d_pp,
            'prot_expiration_days': pdays,
            'delta_0dte': d0,
            'final_value': df_ledger['total_value'].iloc[-1],
            'buy_and_hold_final': df_ledger['buy_and_hold_value'].iloc[-1],
            'volatility': df_ledger['total_value'].pct_change().std(),
            'assignment_rate': assignment_rate,
            'action_ledger': df_action_ledger
        })
        if idx % 100 == 0:
            logging.info(f"{idx}/{len(param_grid)} completed")
    df_results = pd.DataFrame(results)
    top5 = df_results.sort_values(['final_value', 'volatility'], ascending=[False, True]).head(5)
    print(f"\nTop 5 Results for stock: {stock} (asd={asd})")
    print(top5[[col for col in top5.columns if col not in ['action_ledger']]])
    # Export top 5 action ledgers to XLSX
    for i, row in enumerate(top5.itertuples(), 1):
        df_action = row.action_ledger.copy()
        # Sort by event_index if present, otherwise by date
        if 'event_index' in df_action.columns:
            df_action = df_action.sort_values(['date', 'event_index'])
        else:
            df_action = df_action.sort_values('date')
        full_dates = pd.date_range(backtester.start_date, backtester.end_date, freq='B')
        spy_prices = backtester.data.reindex(full_dates).copy().reset_index()
        spy_prices.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close']
        merged = pd.DataFrame({'date': full_dates})
        merged = merged.merge(spy_prices[['date', 'open', 'high', 'low', 'close']], on='date', how='left')
        merged = merged.merge(df_action, on='date', how='left')
        # Add buy & hold value for all dates
        bh_open = backtester.data.iloc[0]['open']
        bh_shares = int(backtester.initial_capital // bh_open)
        bh_cash = backtester.initial_capital - (bh_shares * bh_open)
        merged['BUY_&_HOLD'] = bh_shares * merged['close'] + bh_cash
        if 'open' not in merged.columns:
            merged['open'] = np.nan
        merged = merged.sort_values(['date', 'event_index'] if 'event_index' in merged.columns else ['date'])
        base_cols = ['event_index', 'date', 'open', 'high', 'low', 'close']
        extra_cols = [c for c in ['transaction', 'lots', 'option_price', 'Strike Price', 'premium_received', 'premium_paid', 'total_premiums', 'Inflow/Outflow', 'P/L', 'cash', 'shares', 'total_value', 'buy_and_hold_value', 'BUY_&_HOLD'] if c in merged.columns]
        other_cols = [c for c in merged.columns if c not in base_cols + extra_cols]
        cols = base_cols + extra_cols + other_cols
        merged = merged[cols]
        fname = f"{stock}_asd{asd}_simulation_{i}_final_{row.final_value:.2f}.xlsx"
        merged.to_excel(fname, index=False)
        print(f"Exported {fname} for stock: {stock} (asd={asd})")

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
    ledger, action_ledger, _ = tester.run(0.5, 0.5, 30, 0.5)
    assert len(ledger) == 5
    print("test_run OK")

if __name__ == '__main__':
    test_run()
    main(verbose=True)
