import datetime
import pandas as pd
import yfinance as yf
import ta

# 1. DEFINE TARGET F&O STOCKS FROM SCREENER
FO_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TCS.NS"
]

def calculate_timeframe_signals(df_raw, timeframe_label):
    if len(df_raw) < 200:
        return "Insufficient Data"
        
    df = df_raw.copy()
    # Correct handling of multi-level or single-level column headers from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        
    current_price = close.iloc[-1]
    
    buy_votes = 0
    sell_votes = 0
    total_indicators = 0
    
    # --- SECTION A: 12 MOVING AVERAGES (SMA & EMA: 5, 10, 20, 50, 100, 200) ---
    periods = [5, 10, 20, 50, 100, 200]
    for p in periods:
        sma = ta.trend.sma_indicator(close, window=p).iloc[-1]
        if current_price > sma: buy_votes += 1
        else: sell_votes += 1
        
        ema = ta.trend.ema_indicator(close, window=p).iloc[-1]
        if current_price > ema: buy_votes += 1
        else: sell_votes += 1
        
        total_indicators += 2

    # --- SECTION B: TECHNICAL OSCILLATORS ---
    rsi = ta.momentum.rsi(close, window=14).iloc[-1]
    total_indicators += 1
    if rsi > 50: buy_votes += 1
    elif rsi < 50: sell_votes += 1

    stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3).iloc[-1]
    stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3).iloc[-1]
    total_indicators += 1
    if stoch_k > stoch_d: buy_votes += 1
    else: sell_votes += 1

    macd_line = ta.trend.macd(close).iloc[-1]
    macd_signal = ta.trend.macd_signal(close).iloc[-1]
    total_indicators += 1
    if macd_line > macd_signal: buy_votes += 1
    else: sell_votes += 1

    adx_pos = ta.trend.adx_pos(high, low, close).iloc[-1]
    adx_neg = ta.trend.adx_neg(high, low, close).iloc[-1]
    total_indicators += 1
    if adx_pos > adx_neg: buy_votes += 1
    else: sell_votes += 1

    will_r = ta.momentum.williams_r(high, low, close, lbp=14).iloc[-1]
    total_indicators += 1
    if will_r > -50: buy_votes += 1
    else: sell_votes += 1

    cci = ta.trend.cci(high, low, close, window=20).iloc[-1]
    total_indicators += 1
    if cci > 0: buy_votes += 1
    else: sell_votes += 1

    # --- SECTION C: THE VOTING MATRIX TRANSLATION ---
    buy_ratio = buy_votes / total_indicators
    
    if buy_ratio >= 0.70: return "Strong Buy"
    elif buy_ratio >= 0.55: return "Buy"
    elif buy_ratio <= 0.30: return "Strong Sell"
    elif buy_ratio <= 0.45: return "Sell"
    else: return "Neutral"

# 2. MAIN FETCH AND PROCESSING LOOP
results_book = []
start_date = (datetime.date.today() - datetime.timedelta(days=365 * 20)).strftime('%Y-%m-%d')

print("⏳ Scanning Timeframes & Compiling Multi-Timeframe Scoring Matrix...")

for ticker in FO_STOCKS:
    try:
        stock_data = yf.download(ticker, start=start_date, progress=False)
        if stock_data.empty: continue
        
        df_daily = stock_data
        
        # Safe multi-level resampling aggregation mapping
        resample_dict = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Flatten or handle structural aggregation safely
            df_flat = stock_data.copy()
            df_flat.columns = [c[0] for c in df_flat.columns]
            df_weekly = df_flat.resample('W').agg(resample_dict)
            df_monthly = df_flat.resample('M').agg(resample_dict)
            last_trade_price = round(float(df_flat['Close'].iloc[-1]), 2)
        else:
            df_weekly = stock_data.resample('W').agg(resample_dict)
            df_monthly = stock_data.resample('M').agg(resample_dict)
            last_trade_price = round(float(stock_data['Close'].iloc[-1]), 2)
        
        daily_bias = calculate_timeframe_signals(df_daily, "Daily")
        weekly_bias = calculate_timeframe_signals(df_weekly, "Weekly")
        monthly_bias = calculate_timeframe_signals(df_monthly, "Monthly")
        
        results_book.append({
            "Company Ticker": ticker.replace(".NS", ""),
            "Daily Signal": daily_bias,
            "Weekly Signal": weekly_bias,
            "Monthly Signal": monthly_bias,
            "Last Close (₹)": last_trade_price
        })
    except Exception as error:
        print(f"⚠️ Error compiling data for {ticker}: {error}")

df_matrix = pd.DataFrame(results_book)
print("\n" + "="*70)
print("              MULTI-TIMEFRAME ALGORITHMIC SCREENER OUTPUT       ")
print("="*70)
print(df_matrix.to_string(index=False))
print("="*70)
