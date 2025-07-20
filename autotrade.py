import os
import pyupbit
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import json
import pandas as pd
import ta
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

def get_news_headlines(api_key, query="bitcoin OR cryptocurrency", gl="us", hl="en"):
    """SerpAPI를 통해 최신 뉴스 헤드라인을 가져옵니다."""
    params = {
        "engine": "google_news",
        "q": query,
        "gl": gl,
        "hl": hl,
        "api_key": api_key,
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()
        # 'news_results' 키가 없을 경우를 대비하여 get 사용, 상위 10개만 반환
        return response.json().get("news_results", [])[:10]
    except requests.exceptions.RequestException as e:
        print(f"### SerpAPI News Fetch Error: {e} ###")
        return None
    except json.JSONDecodeError:
        print(f"### SerpAPI JSON Decode Error ###")
        return None

def get_fear_and_greed_index(limit=30):
    """alternative.me API를 통해 공포 탐욕 지수를 가져옵니다."""
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"### Fear & Greed Index API Error: {e} ###")
        return None

def add_technical_indicators(df):
    """주어진 데이터프레임에 주요 보조지표를 추가합니다."""
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df.dropna(inplace=True)
    return df

def generate(upbit_client):
    """필요한 모든 데이터를 수집하고 AI에게 투자 결정을 요청하는 함수"""
    try:
        # 1. 내 자산 정보
        all_balances = upbit_client.get_balances()
        my_balances = [b for b in all_balances if b['currency'] in ['KRW', 'BTC']]

        # 2. 오더북
        orderbook = pyupbit.get_orderbook('KRW-BTC')

        # 3. 차트 데이터 (raw)
        df_day_raw = pyupbit.get_ohlcv('KRW-BTC', count=60, interval="day")
        df_hour_raw = pyupbit.get_ohlcv('KRW-BTC', count=60, interval="minute60")

        # 4. 보조지표 추가
        df_day = add_technical_indicators(df_day_raw.copy())
        df_hour = add_technical_indicators(df_hour_raw.copy())

        # 5. 공포 탐욕 지수
        fng_data = get_fear_and_greed_index(limit=30)
        if fng_data is None: fng_data = []

        # 6. 최신 뉴스 헤드라인 (SerpAPI)
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        news_headlines = []
        if serpapi_key:
            news_headlines = get_news_headlines(serpapi_key)
            if news_headlines is None: news_headlines = []
        else:
            print("### SERPAPI_API_KEY not found in .env. Skipping news fetch. ###")

        # 7. AI에게 보낼 데이터 종합
        prompt_data = f"""
        As a top-tier crypto analyst, synthesize all of the following data points to make a single, decisive investment call (buy/sell/hold) for KRW-BTC.

        ### 1. My Current Investment Status (KRW & BTC only)
        {json.dumps(my_balances, indent=2)}

        ### 2. Current Order Book (Hoga)
        {json.dumps(orderbook, indent=2)}

        ### 3. Chart Data with Indicators (Daily)
        {df_day.to_json(orient='split')}

        ### 4. Chart Data with Indicators (Hourly)
        {df_hour.to_json(orient='split')}
        
        ### 5. Fear & Greed Index (Sentiment, Last 30 Days)
        {json.dumps(fng_data, indent=2)}

        ### 6. Latest News Headlines (Qualitative Factor)
        {json.dumps(news_headlines, indent=2)}
        """
    except Exception as e:
        print(f"### Data Fetching Error: {e} ###")
        return None

    # AI 클라이언트 설정 및 판단 요청
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-1.5-pro"

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt_data)]),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Comprehensive Analysis: Integrating News, Sentiment, and Technicals**

I will now conduct a multi-layered analysis, starting with qualitative factors and then confirming with quantitative data.

**1. Qualitative News Analysis:** First, I'll scan the news headlines for any market-moving events. Positive news (e.g., a major company announcing Bitcoin adoption, favorable ETF news) is a strong bullish factor. Negative news (e.g., a major exchange hack, new restrictive regulations) is a strong bearish factor. The absence of major news suggests technicals will dominate.

**2. Market Sentiment Analysis (Fear & Greed Index):** Next, I'll gauge the overall market emotion. 'Extreme Fear' often signals a bottom is near, making it a contrarian buy indicator. 'Extreme Greed' signals a potential top, suggesting caution or profit-taking.

**3. Technical Analysis (Charts & Indicators):** Finally, I'll confirm my qualitative and sentiment-based hypothesis with technical data. I'm looking for confluence. For example, if there's major positive news and the F&G index shows 'Fear' (indicating disbelief or an early-stage rally), I will look for a bullish MACD cross and rising RSI on the daily chart to confirm a 'buy' signal.

**4. Synthesized Thesis & Decision:**
The news is reporting a significant new regulatory approval for Bitcoin ETFs in a major economy. This is a strong bullish qualitative factor. The F&G Index is at '55' (Greed), indicating growing optimism but not yet 'Extreme Greed'. The daily chart's MACD is crossing upwards, and the RSI is climbing at 60. This confluence of bullish news, rising optimism, and confirming technicals presents a clear 'buy' signal.
"""),
                types.Part.from_text(text="""{\"decision\":\"buy\",\"reason\":\"Strong bullish signal from major positive regulatory news, confirmed by rising market optimism (F&G Index at 55) and bullish technicals (MACD crossover, RSI at 60) on the daily chart.\"}"""),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""You are a world-class crypto analyst. Your primary goal is to synthesize qualitative data (news) with quantitative data (charts, indicators, sentiment index) to make a single, actionable investment decision (buy, sell, or hold). Your reasoning must demonstrate how these different data types support each other (or conflict). Prioritize major news events as they can override technicals. Respond in JSON format.

Response Example:
{\"decision\":\"sell\",\"reason\":\"Major negative news about an exchange hack overrides bullish technicals, creating high uncertainty. F&G index is dropping sharply, confirming market panic.\"}
"""),
        ],
    )

    try:
        response = client.generate_content(
            model=model,
            contents=contents,
            generation_config=generate_content_config,
        )
        return response.text
    except Exception as e:
        print(f"### AI Response Error: {e} ###")
        return None

def transaction(ai_decision_json, upbit_client):
    """AI의 결정을 받아 실제 매매를 실행하는 함수"""
    if not ai_decision_json:
        print("### AI decision is missing. Skipping transaction. ###")
        return

    try:
        ai_decision = json.loads(ai_decision_json)
    except json.JSONDecodeError:
        print(f"### Failed to decode AI's JSON response: {ai_decision_json} ###")
        return

    print("### AI Decision: ", ai_decision.get("decision", "N/A").upper(), "###")
    print(f"### Reason: {ai_decision.get('reason', 'No reason provided.')} ###")

    decision = ai_decision.get("decision")
    if decision == "buy":
        krw_balance = upbit_client.get_balance("KRW")
        if krw_balance > 5000:
            buy_result = upbit_client.buy_market_order("KRW-BTC", krw_balance * 0.9995)
            print("### Buy Order Executed! ###")
            print(buy_result)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif decision == "sell":
        btc_balance = upbit_client.get_balance("KRW-BTC")
        if btc_balance > 0:
            current_price = pyupbit.get_current_price("KRW-BTC")
            if current_price is not None and btc_balance * current_price > 5000:
                sell_result = upbit_client.sell_market_order("KRW-BTC", btc_balance)
                print("### Sell Order Executed! ###")
                print(sell_result)
            else:
                print("### Sell Order Failed: Insufficient BTC value (less than 5000 KRW) or price fetch failed ###")
        else:
            print("### Sell Order Failed: No BTC to sell ###")
    elif decision == "hold":
        print("### Hold Position ###")

if __name__ == "__main__":
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        raise ValueError("UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY must be set in .env file")

    upbit = pyupbit.Upbit(access, secret)
    print("### AI Trading Bot Started with News & Sentiment Analysis ###")

    while True:
        ai_response = generate(upbit)
        transaction(ai_response, upbit)
        print("\n--- Waiting for next cycle (10s) ---\n")
        time.sleep(10)