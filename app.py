import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import baostock as bs
import os
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="A股全景量化工作站 Pro Max", layout="wide", initial_sidebar_state="expanded")

# --- 全局路径 ---
STORE_DIR = "baostock_local_warehouse"
if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)

# --- 辅助算法：技术指标与120M合成 ---
def synthesize_120m(df_60m):
    if len(df_60m) == 0: return pd.DataFrame()
    df = df_60m.copy()
    df['block'] = np.arange(len(df)) // 2
    agg_dict = {'date': 'last', 'time': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'amount': 'sum'}
    return df.groupby('block').agg(agg_dict).reset_index(drop=True)

def compute_indicators(df, short=12, long=26, signal=9):
    for c in ['open', 'high', 'low', 'close', 'volume', 'turn', 'pctChg', 'amount']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    if len(df) == 0: return df
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    ema_short = df['close'].ewm(span=short, adjust=False).mean()
    ema_long = df['close'].ewm(span=long, adjust=False).mean()
    df['dif'] = ema_short - ema_long
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    return df

# --- 获取最新交易日 ---
def get_latest_trading_day():
    today = datetime.now()
    if today.hour < 15: today = today - timedelta(days=1)
    while today.weekday() > 4: today = today - timedelta(days=1)
    return today.strftime("%Y-%m-%d")

# --- 顶级导航栏 ---
tab_market, tab_strategy, tab_ai = st.tabs([
    "📈 模块一：分段极速下载与东财看板", 
    "🔍 模块二：多因子策略选股漏斗", 
    "🤖 模块三：AI 智能自定义选股"
])

# ==========================================
# 📈 模块一：分段下载与东财级看板
# ==========================================
with tab_market:
    st.header("🗄️ 全市场底座数据中心 (分段断点续传版)")
    
    col_dl1, col_dl2 = st.columns([1, 3])
    
    with col_dl1:
        st.subheader("📡 分段下载控制器")
        fetch_days = st.number_input("拉取历史长度(天)", min_value=30, max_value=720, value=365, step=30)
        
        target_pool = st.selectbox("🎯 选择本次下载的数据分块：", [
            "1. 沪深300成分股 (核心资产)",
            "2. 中证500成分股 (中盘成长)",
            "3. 上证50成分股 (大盘蓝筹)",
            "4. 全市场剩余 A股 (智能去重补全)",
            "0. 演示精选股 (仅供测速)"
        ])
        btn_download = st.button("📥 启动安全分段下载", type="primary", use_container_width=True)

    with col_dl2:
        st.subheader("同步进度监控")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    if btn_download:
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"BaoStock登录失败: {lg.error_msg}")
        else:
            status_text.info("正在向交易所网关请求成分股清单...")
            stock_list = []
            
            if "演示精选股" in target_pool:
                stock_list = ["sh.600519", "sz.000001", "sz.300750", "sh.600036", "sz.002594", "sh.601012"]
            elif "沪深300" in target_pool:
                rs = bs.query_hs300_stocks()
                while (rs.error_code == '0') and rs.next(): stock_list.append(rs.get_row_data()[1])
            elif "中证500" in target_pool:
                rs = bs.query_zz500_stocks()
                while (rs.error_code == '0') and rs.next(): stock_list.append(rs.get_row_data()[1])
            elif "上证50" in target_pool:
                rs = bs.query_sz50_stocks()
                while (rs.error_code == '0') and rs.next(): stock_list.append(rs.get_row_data()[1])
            elif "全市场剩余" in target_pool:
                rs = bs.query_all_stock(day=get_latest_trading_day())
                all_stocks = []
                while (rs.error_code == '0') and rs.next(): all_stocks.append(rs.get_row_data()[0])
                existing_codes = set([f.split("_")[0] for f in os.listdir(STORE_DIR) if f.endswith("_daily.csv")])
                for s in all_stocks:
                    if (s.startswith('sh.') or s.startswith('sz.')) and s.split('.')[-1] not in existing_codes:
                        stock_list.append(s)

            total_stocks = len(stock_list)
            if total_stocks == 0:
                status_text.success("该分块下的股票已全部在本地最新状态，无需重复下载！")
            else:
                status_text.info(f"目标锁定 {total_stocks} 只标的，准备拉取历史数据...")
                
                start_date = (datetime.now() - timedelta(days=int(fetch_days))).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")
                saved_count, skip_count = 0, 0
                
                for idx, bs_code in enumerate(stock_list):
                    rc = bs_code.split('.')[-1]
                    daily_file = os.path.join(STORE_DIR, f"{rc}_daily.csv")
                    
                    if os.path.exists(daily_file):
                        mtime = datetime.fromtimestamp(os.path.getmtime(daily_file)).strftime('%Y-%m-%d')
                        if mtime == datetime.now().strftime('%Y-%m-%d'):
                            skip_count += 1
                            progress_bar.progress((idx + 1) / total_stocks)
                            continue

                    status_text.text(f"📡 正在拉取: {bs_code} ({idx+1}/{total_stocks})")
                    
                    rs_d = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume,turn,pctChg", start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
                    rs_w = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume,turn,pctChg", start_date=start_date, end_date=end_date, frequency="w", adjustflag="2")
                    rs_30m = bs.query_history_k_data_plus(bs_code, "date,time,open,high,low,close,volume,amount", start_date=start_date, end_date=end_date, frequency="30", adjustflag="2")
                    rs_60m = bs.query_history_k_data_plus(bs_code, "date,time,open,high,low,close,volume,amount", start_date=start_date, end_date=end_date, frequency="60", adjustflag="2")
                    
                    def fetch_to_df(rs):
                        res_list = []
                        while (rs.error_code == '0') and rs.next(): res_list.append(rs.get_row_data())
                        return pd.DataFrame(res_list, columns=rs.fields) if res_list else pd.DataFrame()
                    
                    df_d = fetch_to_df(rs_d)
                    df_60m = fetch_to_df(rs_60m)
                    
                    if not df_d.empty and not df_60m.empty:
                        df_w = compute_indicators(fetch_to_df(rs_w))
                        df_d = compute_indicators(df_d)
                        df_30m = compute_indicators(fetch_to_df(rs_30m))
                        df_60m = compute_indicators(df_60m)
                        df_120m = compute_indicators(synthesize_120m(df_60m))
                        
                        df_w.to_csv(os.path.join(STORE_DIR, f"{rc}_weekly.csv"), index=False)
                        df_d.to_csv(daily_file, index=False)
                        df_30m.to_csv(os.path.join(STORE_DIR, f"{rc}_30m.csv"), index=False)
                        df_60m.to_csv(os.path.join(STORE_DIR, f"{rc}_60m.csv"), index=False)
                        df_120m.to_csv(os.path.join(STORE_DIR, f"{rc}_120m.csv"), index=False)
                        saved_count += 1
                    
                    progress_bar.progress((idx + 1) / total_stocks)
                    
                status_text.success(f"✨ 任务完成！本次成功拉取 {saved_count} 只，智能跳过 {skip_count} 只。")
        bs.logout()

    st.markdown("---")
    cached_all_files = [f for f in os.listdir(STORE_DIR) if f.endswith("_daily.csv")]
    if cached_all_files:
        st.subheader("📈 东财级交互式全景看板 (支持底部拖拽与框选)")
        
        stock_options = sorted(list(set([f.split("_")[0] for f in cached_all_files])))
        c_view1, c_view2 = st.columns([1, 4])
        with c_view1:
            selected_stock = st.selectbox("🎯 搜索本地标的：", stock_options)
            selected_period = st.selectbox("⏱️ 看盘周期：", ["日线", "周线", "120分钟线", "60分钟线", "30分钟线"])
            
        period_map = {"日线": "daily", "周线": "weekly", "120分钟线": "120m", "60分钟线": "60m", "30分钟线": "30m"}
        
        with c_view2:
            target_path = os.path.join(STORE_DIR, f"{selected_stock}_{period_map[selected_period]}.csv")
            if os.path.exists(target_path):
                df_v = pd.read_csv(target_path)
                if not df_v.empty:
                    df_v['display_time'] = df_v['time'].astype(str).str[:14] if 'time' in df_v.columns else df_v['date']
                    
                    COLOR_UP = '#FF3232'    
                    COLOR_DOWN = '#00CC00'  
                    COLOR_MA5 = '#FFFFFF'   
                    COLOR_MA10 = '#FFFF00'  
                    COLOR_MA20 = '#FF00FF'  
                    
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6, 0.15, 0.25])
                    
                    fig.add_trace(go.Candlestick(x=df_v['display_time'], open=df_v['open'], high=df_v['high'], low=df_v['low'], close=df_v['close'],
                        name="K线", increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN), row=1, col=1)
                    
                    for ma, color in [('MA5', COLOR_MA5), ('MA10', COLOR_MA10), ('MA20', COLOR_MA20)]:
                        if ma in df_v.columns: fig.add_trace(go.Scatter(x=df_v['display_time'], y=df_v[ma], name=ma, line=dict(width=1.2, color=color)), row=1, col=1)
                    
                    vol_colors = [COLOR_UP if row['close'] > row['open'] else COLOR_DOWN for _, row in df_v.iterrows()]
                    fig.add_trace(go.Bar(x=df_v['display_time'], y=df_v['volume'], name="成交量", marker_color=vol_colors), row=2, col=1)
                    
                    macd_colors = [COLOR_UP if val >= 0 else COLOR_DOWN for val in df_v['macd']]
                    fig.add_trace(go.Bar(x=df_v['display_time'], y=df_v['macd'], name="MACD", marker_color=macd_colors), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df_v['display_time'], y=df_v['dif'], name="DIF", line=dict(color=COLOR_MA5, width=1.2)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df_v['display_time'], y=df_v['dea'], name="DEA", line=dict(color=COLOR_MA10, width=1.2)), row=3, col=1)
                    
                    fig.update_layout(
                        height=700, 
                        template="plotly_dark", 
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False,
                        xaxis_rangeslider_visible=False  
                    )
                    
                    fig.update_xaxes(type='category', row=1, col=1)
                    fig.update_xaxes(type='category', row=2, col=1)
                    fig.update_xaxes(
                        type='category', 
                        rangeslider=dict(visible=True, thickness=0.08, bgcolor='#222222'), 
                        row=3, col=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🔍 模块二：多因子策略选股漏斗 (原样保留)
# ==========================================
with tab_strategy:
    st.header("🎛️ 本地数据筛查漏斗 (抽屉模块化)")
    st.markdown("💡 **操作指南**：初始状态下工具**不实施任何拦截**。展开你需要研究的维度，勾选激活，然后执行筛选。")
    
    cached_files = [f for f in os.listdir(STORE_DIR) if f.endswith("_daily.csv")]
    if not cached_files:
        st.warning("⚠️ 数据库为空，请先前往第一页下载分段全市场数据。")
    else:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            with st.expander("💸 维度 1：异动放量特征过滤", expanded=False):
                enable_vol = st.checkbox("激活：天量换手筛选")
                eval_window = st.number_input("考核追溯期(天)", value=30, step=5)
                vol_ratio = st.slider("当日换手 / 前日换手 倍数下限", 1.5, 5.0, 2.5)
                vol_count = st.number_input("上述暴涨发生最少次数", min_value=1, value=1)
                
            with st.expander("📈 维度 2：均线与趋势形态", expanded=False):
                enable_ma = st.checkbox("激活：日线均线排列筛选")
                ma_strategy = st.radio("均线规则", ["收盘价站上 MA20 (短线转强)", "MA5 > MA10 > MA20 (标准多头排列)"])
                
        with col_f2:
            with st.expander("⏱️ 维度 3：次级别 (60M) MACD 过滤", expanded=False):
                enable_macd = st.checkbox("激活：60分钟 MACD 过滤")
                macd_mode = st.radio("副图 MACD 形态要求", ["DIF与DEA均在零轴上方 (水上强势)", "DIF 刚上穿 DEA (水下金叉/水上金叉)"])

            with st.expander("🛑 维度 4：极端涨跌空间防守", expanded=False):
                enable_range = st.checkbox("激活：风险与僵尸股拦截")
                min_gain = st.number_input("区间最高波段脉冲空间不低于(%)", value=10.0)
                max_drawdown = st.number_input("最高点产生后最大回撤不超过(%)", value=-15.0)

        if st.button("🚀 组合选股启动 (从本地 5000 只标的中淘金)", type="primary"):
            passed_stocks = []
            scan_progress = st.progress(0)
            scan_text = st.empty()
            
            for i, f in enumerate(cached_files):
                stock_id = f.split("_")[0]
                scan_progress.progress((i + 1) / len(cached_files))
                
                df_d = pd.read_csv(os.path.join(STORE_DIR, f))
                df_60m = pd.read_csv(os.path.join(STORE_DIR, f"{stock_id}_60m.csv"))
                
                if len(df_d) < eval_window or len(df_60m) < 5: continue
                df_d_win = df_d.tail(eval_window).copy()
                keep = True
                
                if enable_vol and keep:
                    df_d_win['prev_turn'] = df_d_win['turn'].shift(1)
                    hits = ((df_d_win['turn'] >= df_d_win['prev_turn'] * vol_ratio) & (df_d_win['close'] > df_d_win['open'])).sum()
                    if hits < vol_count: keep = False
                
                if enable_ma and keep:
                    last_c, last_m5, last_m10, last_m20 = df_d['close'].iloc[-1], df_d['MA5'].iloc[-1], df_d['MA10'].iloc[-1], df_d['MA20'].iloc[-1]
                    if ma_strategy == "收盘价站上 MA20 (短线转强)":
                        if last_c <= last_m20: keep = False
                    else:
                        if not (last_m5 > last_m10 > last_m20): keep = False
                
                if enable_macd and keep:
                    last_dif, last_dea = df_60m['dif'].iloc[-1], df_60m['dea'].iloc[-1]
                    last_macd, prev_macd = df_60m['macd'].iloc[-1], df_60m['macd'].iloc[-2]
                    if macd_mode == "DIF与DEA均在零轴上方 (水上强势)":
                        if not (last_dif > 0 and last_dea > 0): keep = False
                    else:
                        if not (prev_macd <= 0 and last_macd > 0): keep = False
                        
                if enable_range and keep:
                    base_p = df_d_win['close'].iloc[0]
                    high_p = df_d_win['high'].max()
                    calc_gain = (high_p - base_p) / base_p * 100
                    
                    idx_h = df_d_win['high'].idxmax()
                    low_after_h = df_d_win.loc[df_d_win.index >= idx_h, 'low'].min()
                    calc_dd = (low_after_h - high_p) / high_p * 100 if not pd.isna(low_after_h) else 0.0
                    
                    if calc_gain < min_gain or calc_dd < max_drawdown: keep = False

                if keep:
                    last_row = df_d.iloc[-1]
                    passed_stocks.append({
                        "代码": stock_id,
                        "最新价": f"{last_row['close']}",
                        "涨跌幅": f"{last_row['pctChg']}%",
                        "换手率": f"{last_row['turn']}%"
                    })
            
            scan_text.text("扫描完毕！")
            st.subheader(f"✅ 入围标的 (共 {len(passed_stocks)} 只)")
            if passed_stocks:
                st.dataframe(pd.DataFrame(passed_stocks), use_container_width=True)
            else:
                st.info("条件过严，无符合标准的股票，请调低参数再次尝试。")

# ==========================================
# 🤖 模块三：AI 智能自定义选股 (全量升级版)
# ==========================================
with tab_ai:
    import json
    import requests
    import re

    st.header("🤖 AI 智能语义选股引擎")
    st.markdown("💡 **操作指南**：直接使用日常自然语言描述你的条件，AI 会自动为你生成底层参数并直接调用本地量化库。")

    # ---- 1. 本地硬核因子函数映射表 ----
    def check_volume_surge(df_d, df_60m, params):
        """放量因子: 当日换手率 / 前一日换手率 >= min_ratio"""
        if len(df_d) < 2: return False
        min_ratio = params.get('min_ratio', 2.0)
        prev_turn = df_d['turn'].iloc[-2]
        if pd.isna(prev_turn) or prev_turn == 0: return False
        return df_d['turn'].iloc[-1] >= (prev_turn * min_ratio)

    def check_macd_status(df_d, df_60m, params):
        """MACD状态因子: 支持日线/60分钟轴的零轴上(above_zero)或金叉(cross_up)"""
        timeframe = params.get('timeframe', 'daily')
        df = df_60m if timeframe == '60m' else df_d
        if len(df) < 2: return False
        
        status = params.get('status', 'above_zero')
        last_macd, prev_macd = df['macd'].iloc[-1], df['macd'].iloc[-2]
        last_dif, last_dea = df['dif'].iloc[-1], df['dea'].iloc[-1]
        
        if status == 'above_zero':
            return last_dif > 0 and last_dea > 0
        elif status == 'cross_up':
            return prev_macd <= 0 and last_macd > 0
        return True

    def check_ma_trend(df_d, df_60m, params):
        """均线排列因子: 多头排列(long) 或 空头排列(short)"""
        if len(df_d) < 1: return False
        trend = params.get('trend', 'long')
        if 'MA5' not in df_d.columns or 'MA10' not in df_d.columns or 'MA20' not in df_d.columns:
            return False
        last_m5, last_m10, last_m20 = df_d['MA5'].iloc[-1], df_d['MA10'].iloc[-1], df_d['MA20'].iloc[-1]
        if trend == 'long':
            return last_m5 > last_m10 > last_m20
        elif trend == 'short':
            return last_m5 < last_m10 < last_m20
        return True
        
    def check_price_change(df_d, df_60m, params):
        """当日涨跌幅区间因子"""
        if len(df_d) < 1: return False
        min_pct = params.get('min_pct', -10.0)
        max_pct = params.get('max_pct', 10.0)
        return min_pct <= df_d['pctChg'].iloc[-1] <= max_pct

    # 因子路由映射表
    FACTOR_REGISTRY = {
        "volume_surge": check_volume_surge,
        "macd_status": check_macd_status,
        "ma_trend": check_ma_trend,
        "price_change": check_price_change
    }

    # ---- 2. 完备的系统级提示词 (System Prompt) ----
    FACTOR_DOCS = """
    可用量化因子库 (必须且只能使用以下列表中的 name 和相应的 params 参数)：
    1. {"name": "volume_surge", "params": {"min_ratio": 浮点数}} -> 含义: 今日换手率是昨天的 min_ratio 倍以上。
    2. {"name": "macd_status", "params": {"timeframe": "daily"或"60m", "status": "above_zero"或"cross_up"}} -> 含义: MACD在零轴上方或刚好金叉。
    3. {"name": "ma_trend", "params": {"trend": "long"或"short"}} -> 含义: 均线系统呈多头排列(MA5>MA10>MA20)或空头排列。
    4. {"name": "price_change", "params": {"min_pct": 浮点数, "max_pct": 浮点数}} -> 含义: 当日涨跌幅介于 min_pct% 和 max_pct% 之间。
    """

    sys_prompt = f"""你是一个高级股票策略解析器。请将用户的自然语言选股诉求转换为严格、规范的纯 JSON 配置对象。
    {FACTOR_DOCS}
    
    【输出控制硬性规范】
    - 必须输出合法的纯 JSON 字符串，绝不允许包含任何 Markdown 语法格式标记（不要带 ```json 或 ``` 符号）。
    - 严格根据用户指令映射参数，不要胡乱捏造未定义的因子。
    - 结构必须保持为: {{"filters": [{{"name": "因子名", "params": {{...}}}}]}}
    - 如果用户提示的信息未能匹配到任何因子，返回空的 filters 列表。
    """

    # ---- 3. 双驱动交互引擎 UI 切换 ----
    engine_mode = st.radio("⚙️ 驱动引擎切换", ["🤖 API 直连全自动编译", "📋 网页大模型协同 (免 API Key 复制粘贴)"], horizontal=True)
    user_query = st.text_area("🗣️ 请输入您的自然语言选股条件描述:", placeholder="例如：帮我寻找今日换手率放大3倍以上、均线多头排列，并且日线MACD在零轴上方的股票。", height=100)
    
    parsed_json_str = ""

    if engine_mode == "🤖 API 直连全自动编译":
        with st.expander("🔑 开放式 API 鉴权配置", expanded=False):
            api_key = st.text_input("API Key", type="password", help="支持任何符合 OpenAI 标准接口规范的 API (如 DeepSeek, Kimi, 通义等)")
            api_base = st.text_input("Base URL", value="https://api.deepseek.com/v1")
            model_name = st.text_input("Model Name", value="deepseek-chat")
            
        if st.button("🧠 唤醒 AI 自动编译并选股", type="primary", use_container_width=True):
            if not api_key:
                st.error("请先展开上方配置项并输入您的大模型 API Key！")
            elif not user_query.strip():
                st.error("指令不能为空！")
            else:
                with st.spinner("AI 正在深度解析您的语义意图..."):
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        "temperature": 0.0  # 压低采样随机性，确保结构输出极度稳定
                    }
                    try:
                        res = requests.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=15)
                        res.raise_for_status()
                        raw_content = res.json()['choices'][0]['message']['content'].strip()
                        # 清洗可能存在的 Markdown 干扰符
                        parsed_json_str = re.sub(r'```json|```', '', raw_content).strip()
                    except Exception as e:
                        st.error(f"API 节点请求异常: {e}")
                        
    else:
        st.info("💡 **高灵活性操作说明**：复制下方自动组装的完整提示词投喂给网页端的 DeepSeek/ChatGPT，然后将它回复的 JSON 复制贴回下方即可。")
        composed_prompt = f"{sys_prompt}\n\n当前用户的实际选股要求是：\n{user_query}"
        st.code(composed_prompt, language="text")
        parsed_json_str = st.text_area("📥 请粘贴网页端 AI 回传的纯 JSON 代码串:", height=120, placeholder='{"filters": [...]}')

    # ---- 4. 纯 Python 高性能策略过滤执行引擎 ----
    if parsed_json_str:
        # 手动粘贴模式下需要加一个触发按钮避免自动频繁刷新
        if engine_mode == "📋 网页大模型协同 (免 API Key 复制粘贴)" and not st.button("🚀 确认提炼并执行本地过滤拦截", type="primary", use_container_width=True):
            st.stop()
            
        try:
            # 双重保险清洗
            cleaned_json = re.sub(r'```json|```', '', parsed_json_str).strip()
            strategy_config = json.loads(cleaned_json)
            filters = strategy_config.get("filters", [])
            
            if not filters:
                st.warning("⚠️ 未能从中识别到任何支持的激活因子。请确认您的描述是否落在了底座函数的可控范围内。")
            else:
                st.success(f"⚡ 策略链编译成功！当前激活拦截器：{[f['name'] for f in filters]}")
                with st.expander("🔍 结构化因子控制链参数查看", expanded=False):
                    st.json(strategy_config)
                    
                cached_files = [f for f in os.listdir(STORE_DIR) if f.endswith("_daily.csv")]
                if not cached_files:
                    st.error("⚠️ 本地冷库无数据文件，请先前往第一页批量下载。")
                else:
                    ai_passed_stocks = []
                    ai_progress = st.progress(0)
                    ai_status = st.empty()
                    
                    for i, f in enumerate(cached_files):
                        stock_id = f.split("_")[0]
                        ai_progress.progress((i + 1) / len(cached_files))
                        
                        try:
                            df_d = pd.read_csv(os.path.join(STORE_DIR, f))
                            p_60m = os.path.join(STORE_DIR, f"{stock_id}_60m.csv")
                            df_60m = pd.read_csv(p_60m) if os.path.exists(p_60m) else pd.DataFrame()
                            
                            keep = True
                            # 依次通过 AI 配置的因子过滤器链路
                            for filt in filters:
                                func_name = filt.get("name")
                                params = filt.get("params", {})
                                if func_name in FACTOR_REGISTRY:
                                    if not FACTOR_REGISTRY[func_name](df_d, df_60m, params):
                                        keep = False
                                        break  # 只要有一个因子拦截，立即判负退出
                                        
                            if keep and len(df_d) > 0:
                                last_row = df_d.iloc[-1]
                                ai_passed_stocks.append({
                                    "代码": stock_id,
                                    "最新价": f"{last_row['close']}",
                                    "涨跌幅": f"{last_row['pctChg']}%",
                                    "换手率": f"{last_row['turn']}%"
                                })
                        except Exception:
                            continue  # 自动跳过异常或空文件
                            
                    ai_status.success(f"🎯 扫描全市场完毕！当前符合 AI 意图形态的标的共计: {len(ai_passed_stocks)} 只")
                    if ai_passed_stocks:
                        st.dataframe(pd.DataFrame(ai_passed_stocks), use_container_width=True)
                    else:
                        st.info("当前全市场冷库中无标的能够同时满足该 AI 因子链组合。")
                        
        except json.JSONDecodeError:
            st.error("❌ JSON 编译失败！请确保框体内输入或粘贴的是标准、合法的 JSON 对象。")