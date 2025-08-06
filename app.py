import streamlit as st
import yfinance as yf
import pandas as pd
from mftool import Mftool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from stock_list import indian_stocks
from bs4 import BeautifulSoup
import json
from urllib.parse import quote

# For Gemini AI
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Financial Asset Inspector 📈",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with enhanced styling
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Main components dark styling */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a5c;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4a4a5c;
    }
    
    /* Main header with gradient */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a4a5c;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #fafafa;
    }
    
    /* AI Sidebar with beautiful gradient background */
    .ai-sidebar {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #4a5568;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.4);
        margin-bottom: 1rem;
    }
    
    .ai-sidebar h2, .ai-sidebar h3 {
        color: #64ffda !important;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    .ai-sidebar .stSuccess {
        background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%);
        color: #000;
        border-radius: 8px;
        padding: 0.5rem;
        font-weight: bold;
    }
    
    .ai-sidebar .stWarning {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #000;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .ai-sidebar .stInfo {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #000;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Enhanced chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border-left: 4px solid #00c9ff;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        color: #fafafa;
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.4);
        border-left: 4px solid #64ffda;
    }
    
    /* Trading signals with neon effect */
    .trading-signal {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .trading-signal::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        color: #fafafa;
        border-radius: 8px;
        border: 1px solid #4a4a5c;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric value styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #4a4a5c;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="metric-container"] > div {
        color: #fafafa;
    }
    
    /* Gemini chat container */
    .gemini-chat {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #4a5568;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.4);
        color: #fafafa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(145deg, #0e1117 0%, #1a1a2e 100%);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Custom signal colors */
    .buy-signal {
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        font-weight: bold;
    }
    
    .sell-signal {
        color: #ff4757;
        text-shadow: 0 0 10px rgba(255, 71, 87, 0.5);
        font-weight: bold;
    }
    
    .hold-signal {
        color: #ffa726;
        text-shadow: 0 0 10px rgba(255, 167, 38, 0.5);
        font-weight: bold;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #4a4a5c;
        text-align: center;
        color: #b0b0b0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility function to determine currency symbol ---
def get_currency_symbol(ticker_symbol, market_type):
    """Get appropriate currency symbol based on market type."""
    if market_type == "Indian Stocks (NSE)":
        return "₹"
    else:
        return "$"

# --- Gemini AI Functions ---
def initialize_gemini(api_key):
    """Initialize Gemini AI with API key."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {e}")
        return None

def get_stock_analysis_prompt(stock_info, technical_data, price_data, currency_symbol):
    """Generate a comprehensive prompt for stock analysis."""
    current_price = stock_info.get('currentPrice', 'N/A')
    market_cap = stock_info.get('marketCap', 'N/A')
    
    prompt = f"""
    As a financial AI assistant, provide a comprehensive analysis of the following stock:
    
    Company: {stock_info.get('longName', 'N/A')}
    Ticker: {stock_info.get('symbol', 'N/A')}
    Current Price: {currency_symbol}{current_price}
    Market Cap: {currency_symbol}{market_cap}
    P/E Ratio: {stock_info.get('trailingPE', 'N/A')}
    P/B Ratio: {stock_info.get('priceToBook', 'N/A')}
    Dividend Yield: {stock_info.get('dividendYield', 'N/A')}
    Beta: {stock_info.get('beta', 'N/A')}
    52-Week High: {currency_symbol}{stock_info.get('fiftyTwoWeekHigh', 'N/A')}
    52-Week Low: {currency_symbol}{stock_info.get('fiftyTwoWeekLow', 'N/A')}
    
    Technical Indicators:
    RSI: {technical_data.get('RSI', 'N/A')}
    MACD: {technical_data.get('MACD', 'N/A')}
    Trading Signal: {technical_data.get('signal', 'N/A')}
    Technical Score: {technical_data.get('score', 'N/A')}/100
    
    Please provide:
    1. A brief company overview and business analysis
    2. Financial health assessment based on the metrics
    3. Technical analysis interpretation
    4. Investment recommendation (Buy/Hold/Sell) with reasoning
    5. Key risks to consider
    6. Price targets or expectations
    7. Suitable investor profile for this stock
    
    Keep the analysis concise but comprehensive, suitable for both beginners and experienced investors.
    """
    return prompt

def chat_with_gemini(model, message, context=None):
    """Chat with Gemini AI."""
    try:
        if context:
            full_message = f"Context: {context}\n\nUser Question: {message}"
        else:
            full_message = message
        
        response = model.generate_content(full_message)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- Store API key in session state ---
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Enhanced AI Sidebar ---
with st.sidebar:
    st.markdown('<div class="ai-sidebar">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🤖 AI Financial Assistant")
    st.markdown("*Powered by Google Gemini*")
    
    # API Key Input with enhanced styling
    api_key = st.text_input(
        "🔑 Enter your Gemini API key:",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Get your free API key from https://makersuite.google.com/app/apikey"
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        if api_key:
            st.session_state.gemini_model = initialize_gemini(api_key)
    
    if st.session_state.gemini_model:
        st.success("✅ Gemini AI Connected & Ready!")
        
        # Enhanced chat interface
        user_question = st.text_area(
            "💭 Ask me anything about finance:",
            placeholder="E.g., What is a good P/E ratio? How to analyze stocks? Market trends?",
            height=120
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Ask AI", use_container_width=True):
                if user_question.strip():
                    with st.spinner("🤖 AI is thinking..."):
                        response = chat_with_gemini(st.session_state.gemini_model, user_question)
                        st.session_state.chat_history.append(("user", user_question))
                        st.session_state.chat_history.append(("ai", response))
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Enhanced chat history display
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Conversations")
            # Show last 4 messages (2 exchanges)
            for i, (sender, message) in enumerate(reversed(st.session_state.chat_history[-4:])):
                if sender == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>👤 You:</strong><br>{message[:150]}{'...' if len(message) > 150 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-message">
                        <strong>🤖 Gemini:</strong><br>{message[:200]}{'...' if len(message) > 200 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("🔑 Please enter your Gemini API key to unlock AI features")
        st.info("💡 Get your free key from Google AI Studio")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Stock data functions (keeping existing logic) ---
@st.cache_data(ttl=300)
def get_stock_data(ticker_symbol: str, period: str = "1y") -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Fetch comprehensive stock data."""
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        hist = ticker_data.history(period=period)
        return info, hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

def calculate_advanced_indicators(hist_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators."""
    if hist_data is None or hist_data.empty:
        return hist_data
    
    df = hist_data.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    return df

def generate_trading_signal(df: pd.DataFrame) -> Tuple[str, float, Dict]:
    """Generate buy/sell/hold signal based on multiple indicators."""
    if df is None or df.empty or len(df) < 50:
        return "HOLD", 50, {"reason": "Insufficient data"}
    
    latest = df.iloc[-1]
    score = 50
    signals = []
    
    # RSI analysis
    if latest['RSI'] < 30:
        score += 15
        signals.append("RSI oversold")
    elif latest['RSI'] > 70:
        score -= 15
        signals.append("RSI overbought")
    
    # MACD analysis
    if latest['MACD'] > latest['MACD_Signal']:
        score += 10
        signals.append("MACD bullish")
    else:
        score -= 10
        signals.append("MACD bearish")
    
    # Moving average analysis
    if latest['Close'] > latest['SMA_20']:
        score += 5
        if latest['Close'] > latest['SMA_50']:
            score += 10
            signals.append("Price above major SMAs")
    else:
        score -= 5
        if latest['Close'] < latest['SMA_50']:
            score -= 10
            signals.append("Price below major SMAs")
    
    # Determine signal
    if score >= 70:
        signal = "STRONG BUY"
    elif score >= 60:
        signal = "BUY"
    elif score <= 30:
        signal = "STRONG SELL"
    elif score <= 40:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    technical_data = {
        "signal": signal,
        "score": score,
        "RSI": latest['RSI'],
        "MACD": latest['MACD'],
        "signals": signals
    }
    
    return signal, score, technical_data

# --- Main Application ---
def main():
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Financial Asset Inspector</h1>
        <p>✨ Advanced Stock Analysis • Real-time Data • Smart AI Insights • Dark Theme Experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation with enhanced styling
    tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "🎯 AI Stock Insights", "🌍 Market Overview"])
    
    with tab1:
        st.subheader("📈 Stock Analysis Dashboard")
        
        # Market selection
        market_type = st.selectbox("🌏 Select Market:", ["Indian Stocks (NSE)", "Global Stocks"])
        
        if market_type == "Indian Stocks (NSE)":
            selected_stock = st.selectbox("📊 Select Indian Stock:", indian_stocks)
            if selected_stock:
                ticker_symbol = selected_stock.split(" - ")[0]
        else:
            global_stocks = [
                "AAPL - Apple Inc",
                "MSFT - Microsoft Corporation", 
                "GOOGL - Alphabet Inc",
                "AMZN - Amazon.com Inc",
                "TSLA - Tesla Inc",
                "META - Meta Platforms Inc",
                "NVDA - NVIDIA Corporation",
                "JPM - JPMorgan Chase & Co",
                "JNJ - Johnson & Johnson",
                "V - Visa Inc"
            ]
            selected_stock = st.selectbox("📊 Select Global Stock:", global_stocks)
            if selected_stock:
                ticker_symbol = selected_stock.split(" - ")[0]
        
        if selected_stock:
            period = st.selectbox("⏰ Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"])
            
            # Get currency symbol based on market type
            currency_symbol = get_currency_symbol(ticker_symbol, market_type)
            
            with st.spinner("📊 Fetching stock data..."):
                info, hist_data = get_stock_data(ticker_symbol, period)
            
            if info and hist_data is not None and not hist_data.empty:
                # Enhanced metrics display with correct currency
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = info.get('currentPrice', hist_data['Close'].iloc[-1])
                    prev_close = info.get('previousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price)
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0
                    
                    st.metric(
                        "💰 Current Price", 
                        f"{currency_symbol}{current_price:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                
                with col2:
                    market_cap = info.get('marketCap', 0)
                    if market_cap > 1e12:
                        if currency_symbol == "₹":
                            mc_display = f"₹{market_cap/1e12:.2f}T"
                        else:
                            mc_display = f"${market_cap/1e12:.2f}T"
                    elif market_cap > 1e9:
                        if currency_symbol == "₹":
                            mc_display = f"₹{market_cap/1e9:.2f}B"
                        else:
                            mc_display = f"${market_cap/1e9:.2f}B"
                    else:
                        mc_display = "N/A"
                    st.metric("🏢 Market Cap", mc_display)
                
                with col3:
                    pe_ratio = info.get('trailingPE', 0)
                    st.metric("📊 P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                
                with col4:
                    volume = info.get('volume', 0)
                    if volume > 1e6:
                        vol_display = f"{volume/1e6:.1f}M"
                    elif volume > 1e3:
                        vol_display = f"{volume/1e3:.1f}K"
                    else:
                        vol_display = str(volume) if volume else "N/A"
                    st.metric("📈 Volume", vol_display)
                
                # Enhanced Technical Analysis
                st.subheader("🔍 Technical Analysis")
                
                df_with_indicators = calculate_advanced_indicators(hist_data)
                signal, score, technical_data = generate_trading_signal(df_with_indicators)
                
                # Enhanced trading signal display
                signal_class = "buy-signal" if "BUY" in signal else "sell-signal" if "SELL" in signal else "hold-signal"
                st.markdown(f"""
                <div class="trading-signal">
                    <h2>🎯 Trading Signal: <span class="{signal_class}">{signal}</span></h2>
                    <h3>📊 Technical Score: <span class="{signal_class}">{score}/100</span></h3>
                    <p>Based on RSI, MACD, and Moving Average analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced technical indicators
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rsi = technical_data.get('RSI', 0)
                    rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📈 RSI (14)</h4>
                        <h2>{rsi:.1f}</h2>
                        <p>{rsi_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    macd = technical_data.get('MACD', 0)
                    macd_trend = "🟢 Bullish" if macd > 0 else "🔴 Bearish"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 MACD</h4>
                        <h2>{macd:.4f}</h2>
                        <p>{macd_trend} Momentum</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if technical_data.get('signals'):
                        signals_display = '<br>'.join([f"• {signal}" for signal in technical_data['signals'][:3]])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎯 Key Signals</h4>
                            <p>{signals_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Enhanced price chart
                st.subheader("📈 Interactive Price Chart")
                fig = go.Figure()
                
                # Main price line with gradient
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00d4ff', width=3),
                    fill='tonexty',
                    fillcolor='rgba(0, 212, 255, 0.1)'
                ))
                
                if not df_with_indicators.empty:
                    fig.add_trace(go.Scatter(
                        x=df_with_indicators.index,
                        y=df_with_indicators['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#ffa726', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_with_indicators.index,
                        y=df_with_indicators['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#ff5722', width=2)
                    ))
                
                fig.update_layout(
                    title=f"{info.get('longName', selected_stock)} - Price Chart",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency_symbol})",
                    height=600,
                    hovermode='x unified',
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 AI-Powered Stock Insights")
        
        if not st.session_state.gemini_model:
            st.markdown("""
            <div class="gemini-chat">
                <h3>🔑 AI Features Locked</h3>
                <p>Please enter your Gemini API key in the sidebar to unlock powerful AI insights!</p>
                <p>💡 Get your free API key from: <a href="https://makersuite.google.com/app/apikey" target="_blank" style="color: #64ffda;">Google AI Studio</a></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if 'selected_stock' in locals() and selected_stock and info and hist_data is not None:
                st.markdown(f"""
                <div class="gemini-chat">
                    <h3>🤖 AI Analysis Ready for: {info.get('longName', selected_stock)}</h3>
                    <p>Current Price: {currency_symbol}{info.get('currentPrice', 'N/A')} | Market Cap: {currency_symbol}{info.get('marketCap', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Analysis section
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("🧠 Get Full AI Analysis", use_container_width=True, key="full_analysis"):
                        with st.spinner("🤖 AI is deeply analyzing the stock..."):
                            # Prepare comprehensive data for AI analysis
                            df_with_indicators = calculate_advanced_indicators(hist_data)
                            signal, score, technical_data = generate_trading_signal(df_with_indicators)
                            
                            analysis_prompt = get_stock_analysis_prompt(info, technical_data, hist_data, currency_symbol)
                            ai_analysis = chat_with_gemini(st.session_state.gemini_model, analysis_prompt)
                            
                            st.session_state[f'ai_analysis_{ticker_symbol}'] = ai_analysis
                
                # Display comprehensive AI analysis if available
                analysis_key = f'ai_analysis_{ticker_symbol}'
                if analysis_key in st.session_state:
                    st.markdown("### 📊 Comprehensive AI Analysis Report")
                    st.markdown(f"""
                    <div class="gemini-chat">
                        <h4>🤖 Gemini AI Deep Analysis</h4>
                        {st.session_state[analysis_key]}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick AI Insights section
                st.markdown("### ⚡ Quick AI Insights")
                
                quick_questions = [
                    "Is this stock suitable for long-term investment?",
                    "What are the main risks of investing in this company?",
                    "How does this stock compare to its industry peers?",
                    "What should I watch out for in upcoming earnings?",
                    "What is the fair value estimate for this stock?",
                    "Should I buy, hold, or sell this stock right now?"
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_question = st.selectbox("🎯 Choose a quick question:", ["Select a question..."] + quick_questions)
                
                with col2:
                    if selected_question != "Select a question...":
                        if st.button("🚀 Ask AI", key="quick_question", use_container_width=True):
                            with st.spinner("🤖 Getting AI insights..."):
                                context = f"Stock: {info.get('longName')} ({ticker_symbol}), Current Price: {currency_symbol}{info.get('currentPrice', 'N/A')}, P/E: {info.get('trailingPE', 'N/A')}, Market: {market_type}"
                                response = chat_with_gemini(st.session_state.gemini_model, selected_question, context)
                                
                                st.markdown("### 🤖 AI Response")
                                st.markdown(f"""
                                <div class="ai-message">
                                    <strong>❓ Question:</strong> {selected_question}<br><br>
                                    <strong>🤖 AI Answer:</strong><br>{response}
                                </div>
                                """, unsafe_allow_html=True)
                
                # Custom question section
                st.markdown("### 💭 Ask Your Own Question")
                custom_question = st.text_area(
                    "Ask anything about this stock:",
                    placeholder="e.g., What are the growth prospects? Is the valuation reasonable?",
                    height=80,
                    key="custom_stock_question"
                )
                
                if st.button("🔍 Get Custom Analysis", key="custom_analysis"):
                    if custom_question.strip():
                        with st.spinner("🤖 AI is analyzing your question..."):
                            context = f"Stock Analysis Context: {info.get('longName')} ({ticker_symbol}), Price: {currency_symbol}{info.get('currentPrice', 'N/A')}, Market Cap: {currency_symbol}{info.get('marketCap', 'N/A')}, P/E: {info.get('trailingPE', 'N/A')}, Beta: {info.get('beta', 'N/A')}"
                            response = chat_with_gemini(st.session_state.gemini_model, custom_question, context)
                            
                            st.markdown(f"""
                            <div class="ai-message">
                                <strong>💭 Your Question:</strong> {custom_question}<br><br>
                                <strong>🤖 AI Analysis:</strong><br>{response}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="gemini-chat">
                    <h3>📈 Select a Stock First</h3>
                    <p>Please go to the <strong>Stock Analysis</strong> tab and select a stock to get AI-powered insights!</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🌍 Global Market Overview")
        
        # Enhanced market indices with correct currency
        st.markdown("### 📊 Major Market Indices")
        
        indices_data = {
            "Global Markets": {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC", 
                "Dow Jones": "^DJI",
                "FTSE 100": "^FTSE"
            },
            "Indian Markets": {
                "NIFTY 50": "^NSEI",
                "SENSEX": "^BSESN",
                "NIFTY Bank": "^NSEBANK",
                "NIFTY IT": "^CNXIT"
            }
        }
        
        for market_name, indices in indices_data.items():
            st.markdown(f"#### {market_name}")
            cols = st.columns(4)
            
            for i, (name, symbol) in enumerate(indices.items()):
                with cols[i % 4]:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="2d")
                        if not hist.empty and len(hist) >= 2:
                            current = hist['Close'].iloc[-1]
                            previous = hist['Close'].iloc[-2]
                            change = current - previous
                            change_pct = (change / previous) * 100
                            
                            # Use appropriate currency symbol for Indian markets
                            currency = "₹" if "NIFTY" in name or "SENSEX" in name else "$" if name not in ["FTSE 100"] else "£"
                            
                            st.metric(
                                name,
                                f"{current:.2f}",
                                f"{change:+.2f} ({change_pct:+.2f}%)"
                            )
                        else:
                            st.metric(name, "N/A", "N/A")
                    except:
                        st.metric(name, "Loading...", "N/A")
        
        # AI Market Insights section
        if st.session_state.gemini_model:
            st.markdown("### 🤖 AI Market Intelligence")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Global Market Analysis", use_container_width=True):
                    with st.spinner("🤖 Analyzing global market conditions..."):
                        market_prompt = """
                        Provide a comprehensive analysis of current global financial market conditions:
                        1. Overall market sentiment and trends
                        2. Key economic indicators and their impact
                        3. Sector-wise performance outlook
                        4. Geopolitical factors affecting markets
                        5. Investment strategies for current environment
                        6. Risk factors to monitor
                        7. Opportunities in different asset classes
                        
                        Keep it actionable for retail investors while being comprehensive.
                        """
                        
                        market_analysis = chat_with_gemini(st.session_state.gemini_model, market_prompt)
                        
                        st.markdown(f"""
                        <div class="gemini-chat">
                            <h4>🌍 Global Market Analysis</h4>
                            {market_analysis}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🇮🇳 Indian Market Focus", use_container_width=True):
                    with st.spinner("🤖 Analyzing Indian market specifics..."):
                        indian_market_prompt = """
                        Provide a focused analysis of the Indian stock market:
                        1. NIFTY and SENSEX outlook
                        2. Key sectors driving Indian markets
                        3. Impact of government policies on markets
                        4. FII/DII flows and their significance
                        5. Currency impact (INR vs USD)
                        6. Best investment strategies for Indian markets
                        7. Regulatory changes and their market impact
                        
                        Focus on insights relevant to Indian retail investors.
                        """
                        
                        indian_analysis = chat_with_gemini(st.session_state.gemini_model, indian_market_prompt)
                        
                        st.markdown(f"""
                        <div class="gemini-chat">
                            <h4>🇮🇳 Indian Market Analysis</h4>
                            {indian_analysis}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col3:
                if st.button("📈 Sector Recommendations", use_container_width=True):
                    with st.spinner("🤖 Analyzing sector opportunities..."):
                        sector_prompt = """
                        Provide sector-wise investment recommendations:
                        1. Top performing sectors currently
                        2. Undervalued sectors with potential
                        3. Technology sector outlook
                        4. Financial services analysis
                        5. Healthcare and pharma prospects
                        6. Consumer goods and retail trends
                        7. Energy and infrastructure outlook
                        8. Emerging sectors to watch
                        
                        Include both global and Indian market perspectives.
                        """
                        
                        sector_analysis = chat_with_gemini(st.session_state.gemini_model, sector_prompt)
                        
                        st.markdown(f"""
                        <div class="gemini-chat">
                            <h4>📊 Sector Analysis & Recommendations</h4>
                            {sector_analysis}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Market news and insights
            st.markdown("### 📰 Ask About Market News & Trends")
            market_question = st.text_area(
                "Ask about specific market topics:",
                placeholder="e.g., What's the impact of interest rate changes? How are tech stocks performing?",
                height=80,
                key="market_question"
            )
            
            if st.button("📡 Get Market Insights", key="market_insights"):
                if market_question.strip():
                    with st.spinner("🤖 Analyzing market question..."):
                        response = chat_with_gemini(st.session_state.gemini_model, market_question)
                        
                        st.markdown(f"""
                        <div class="ai-message">
                            <strong>❓ Your Question:</strong> {market_question}<br><br>
                            <strong>🤖 Market Analysis:</strong><br>{response}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="gemini-chat">
                <h3>🔑 Unlock AI Market Intelligence</h3>
                <p>Enter your Gemini API key in the sidebar to access powerful market analysis and insights!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div class="footer">
        <h3>🤖 AI-Powered Financial Asset Inspector</h3>
        <p>✨ <strong>Enhanced with Google Gemini AI</strong> • Dark Theme • Real-time Data • Smart Analysis</p>
        <br>
        <p>⚠️ <strong>Important Disclaimer:</strong> This application provides AI-generated analysis for educational and informational purposes only. 
        AI insights should not be considered as financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.</p>
        <br>
        <p>🔗 <strong>Resources:</strong></p>
        <p>🆓 <a href="https://makersuite.google.com/app/apikey" target="_blank" style="color: #64ffda;">Get Free Gemini API Key</a> | 
        📚 <a href="https://developers.google.com/ai/gemini-api" target="_blank" style="color: #64ffda;">Gemini API Documentation</a></p>
        <br>
        <p style="color: #888;">💡 <strong>Tip:</strong> Use the AI assistant in the sidebar for general financial questions and specific stock analysis in the dedicated tabs!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()