"""PaperTrader - Olymp-style paper trading web app with a self-learning AI. Run: streamlit run app.py"""
import os, time
import pandas as pd
import streamlit as st
import trading_bot as tb

st.set_page_config(page_title="PaperTrader", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
[data-testid="stMetric"]{background:#1e2030;padding:10px 14px;border-radius:10px}
.st-key-btn_up button,.st-key-btn_down button{width:100%;height:3.4rem;font-size:1.15rem;font-weight:700;color:#fff;border:0}
.st-key-btn_up button{background:#27c498}.st-key-btn_down button{background:#f0524f}
.st-key-btn_up button:disabled,.st-key-btn_down button:disabled{opacity:.35}
.sig{background:#1e2030;border-left:5px solid;border-radius:8px;padding:10px 14px;margin-top:10px}
</style>""", unsafe_allow_html=True)

ASSETS = ["GBP/CAD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/INR",
          "BTC/USDT", "ETH/USDT", "AAPL", "^NSEI"]
TF = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
EXPIRY = {"1 min": 60, "2 min": 120, "5 min": 300, "15 min": 900}

@st.cache_resource
def core():                       # one demo wallet + one brain per market, shared and persistent
    return tb.Store("demo_account.db"), {}
store, agents = core()

@st.cache_data(ttl=5, show_spinner=False)
def get(sym, tf): return tb.fetch(sym, tf, 400)

def fmt(x): return f"{x:,.5f}" if x < 50 else f"{x:,.2f}"

with st.sidebar:
    st.title("📈 PaperTrader")
    sym = st.selectbox("Asset", ASSETS, key="asset")
    tf = st.selectbox("Candle size", list(TF), key="tf")
    refresh = st.slider("Live refresh (seconds)", 5, 60, 5, key="refresh")
    payout = st.slider("Payout %", 60, 95, 80, key="payout") / 100
    st.divider()
    auto = st.toggle("🤖 Let the AI trade by itself", key="auto")
    watch = st.multiselect("Extra markets for the AI", [a for a in ASSETS if a != sym], key="watch")
    st.caption("The AI always learns from live candles; it only places demo trades when the toggle is on.")
    if st.button("♻️ Reset demo account to ₹10,000", key="reset"):
        try: store.db.close(); os.remove("demo_account.db")
        except OSError: pass
        st.cache_resource.clear(); st.rerun()

@st.fragment(run_every=refresh)
def panel():
    now, watched = time.time(), list(dict.fromkeys([sym] + watch))
    open_syms = [r[0] for r in store.db.execute("SELECT DISTINCT symbol FROM manual")]
    data, prices = {}, {}
    for s in dict.fromkeys(watched + open_syms):
        try: data[s] = get(s, tf); prices[s] = float(data[s].close.iloc[-1])
        except Exception as e: st.warning(f"{s}: data unavailable ({str(e)[:90]})")
    for s in watched:
        if s in data:
            if s not in agents: agents[s] = tb.Agent(s, store)
            tb.advance(agents[s], data[s], trade=auto)
    for s, side, pnl in tb.settle_manual(store, prices):
        st.toast(f"{'✅ WIN' if pnl > 0 else '➖ DRAW' if pnl == 0 else '❌ LOSS'} {s} {'UP' if side == 'BUY' else 'DOWN'}: ₹{pnl:+,.2f}")
    if sym not in data: st.error("Waiting for market data…"); return
    df, a = data[sym], agents[sym]
    cash = store.cash()
    eq = cash + store.staked() + sum(x.unreal() for x in agents.values())
    store.record_equity(now, eq)
    sig, p = a.sig
    price = prices[sym]
    stale = now - float(df.ts.iloc[-1]) > max(3 * TF[tf], 600)

    m = st.columns(4)
    m[0].metric("Demo balance", f"₹{cash:,.2f}")
    m[1].metric("Equity (P&L)", f"₹{eq:,.2f}", f"{eq - tb.START_INR:+,.2f}")
    m[2].metric(sym, fmt(price), f"{price / float(df.close.iloc[-60]) - 1:+.3%}")
    m[3].metric("AI signal", sig, f"P(up) {p:.0%}", delta_color="off")
    if stale: st.warning("Market looks closed (no fresh candles). Trading is paused until it reopens.")

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(tb.build_fig(store, sym, df, f"{sym} · {fmt(price)}", height=640), key="chart")
    with right:
        amount = st.number_input("Amount (₹)", 10.0, 100000.0, 500.0, 50.0, key="amount")
        exp = st.selectbox("Expiry", list(EXPIRY), key="expiry")
        st.caption(f"Win +₹{amount * payout:,.2f} · break-even needs {1 / (1 + payout):.1%} accuracy")
        up = st.button("⬆ UP", key="btn_up", disabled=stale)
        down = st.button("⬇ DOWN", key="btn_down", disabled=stale)
        for clicked, side in ((up, "BUY"), (down, "SELL")):
            if clicked:
                if tb.open_manual(store, sym, side, amount, price, EXPIRY[exp], payout):
                    st.toast(f"Trade opened: {sym} {'UP' if side == 'BUY' else 'DOWN'} ₹{amount:,.0f} @ {fmt(price)}")
                    st.rerun(scope="fragment")
                else: st.error("Not enough demo balance")
        col = "#27c498" if sig == "BUY" else "#f0524f" if sig == "SELL" else "#8a8fa3"
        st.markdown(f"<div class='sig' style='border-color:{col}'><b style='color:{col}'>"
                    f"{'BUY ⬆' if sig == 'BUY' else 'SELL ⬇' if sig == 'SELL' else 'HOLD ⏸'}</b><br>"
                    f"AI says market likely {'UP' if p > .5 else 'DOWN'}<br>"
                    f"accuracy {a.brain.accuracy():.0%}</div>", unsafe_allow_html=True)
        if a.brain.n < tb.MIN_SAMPLES: st.caption(f"🧠 learning: {a.brain.n}/{tb.MIN_SAMPLES} candles")
        if auto: st.caption("🤖 AI trading ON: " + ", ".join(watched))

    t1, t2, t3 = st.tabs(["Open trades", "History", "AI brain & balance"])
    with t1:
        o = pd.read_sql("SELECT symbol,side,amount,entry,expiry_ts FROM manual", store.db)
        o["closes in"] = (o.expiry_ts - now).clip(lower=0).round().astype(int).astype(str) + " s"
        bots = [dict(symbol=k, side=v.pos["side"], amount=round(v.pos["notional"], 2), entry=v.pos["entry"],
                     **{"closes in": "AI"}) for k, v in agents.items() if v.pos]
        st.dataframe(pd.concat([o.drop(columns="expiry_ts"), pd.DataFrame(bots)]), hide_index=True)
    with t2:
        h = pd.read_sql("SELECT close_ts,symbol,side,entry,exit,pnl,why FROM trades ORDER BY id DESC LIMIT 50", store.db)
        h["time"] = tb.T(h.close_ts).dt.strftime("%d %b %H:%M")
        st.dataframe(h[["time", "symbol", "side", "entry", "exit", "pnl", "why"]], hide_index=True)
    with t3:
        st.dataframe(pd.DataFrame([dict(market=k, candles_learned=v.brain.n, accuracy=f"{v.brain.accuracy():.1%}")
                                   for k, v in agents.items()]), hide_index=True)
        e = pd.read_sql("SELECT ts,value FROM equity ORDER BY ts", store.db)
        if len(e) > 1: st.line_chart(e.assign(t=tb.T(e.ts)).set_index("t")["value"])

panel()
st.caption("Paper trading with virtual money only. Not financial advice; no strategy is guaranteed to profit.")
