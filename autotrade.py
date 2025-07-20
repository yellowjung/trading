import os
import time
import json
import pandas as pd
import ta
import requests
import pyupbit
from dotenv import load_dotenv
from google import genai
from google.genai import types
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# .env 파일에서 환경 변수 로드
load_dotenv()

# ==============================================================================
# STEP 1: 차트 스크린샷 캡처 기능 (test.py에서 가져와 통합)
# ==============================================================================
def capture_chart_screenshot(url, save_path):
    """
    주어진 URL의 차트를 설정(1시간봉, 볼린저밴드)하고 캡처하여 저장합니다.
    자동화를 위해 헤드리스 모드로 실행됩니다.
    """
    print("📈 차트 스크린샷 캡처를 시작합니다...")
    try:
        chrome_options = Options()
        # 자동 매매 환경에서는 브라우저 창이 보이지 않도록 headless 옵션을 사용합니다.
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--lang=ko_KR")
        # 일부 시스템에서 발생하는 오류 방지를 위한 옵션
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")


        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        wait = WebDriverWait(driver, 20) # 네트워크 환경을 고려해 대기시간을 20초로 설정

        # 1. '1시간' 봉으로 변경
        time_menu_button_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]"
        time_menu_button = wait.until(EC.element_to_be_clickable((By.XPATH, time_menu_button_xpath)))
        time_menu_button.click()

        one_hour_option_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]"
        one_hour_option = wait.until(EC.element_to_be_clickable((By.XPATH, one_hour_option_xpath)))
        one_hour_option.click()
        time.sleep(1)

        # 2. '볼린저 밴드' 지표 추가
        indicator_menu_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]"
        indicator_menu = wait.until(EC.element_to_be_clickable((By.XPATH, indicator_menu_xpath)))
        indicator_menu.click()

        bollinger_bands_option_xpath = "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[14]"
        bollinger_bands_option = wait.until(EC.element_to_be_clickable((By.XPATH, bollinger_bands_option_xpath)))
        bollinger_bands_option.click()
        time.sleep(3)

        # 3. 전체 화면 캡처
        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        driver.set_window_size(1920, total_height)
        time.sleep(2)
        driver.save_screenshot(save_path)
        print(f"✅ 스크린샷이 '{save_path}' 경로에 성공적으로 저장되었습니다.")
        return True

    except Exception as e:
        print(f"❌ 차트 캡처 중 오류가 발생했습니다: {e}")
        return False
    finally:
        if 'driver' in locals() and driver:
            driver.quit()

# --- 기존 autotrade.py 함수들 ---
def get_news_headlines(api_key, query="bitcoin OR cryptocurrency", gl="us", hl="en"):
    """SerpAPI를 통해 최신 뉴스 헤드라인(title, date만)을 가져옵니다."""
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
        news_results = response.json().get("news_results", [])
        filtered_news = [{"title": item.get("title"), "date": item.get("date")} for item in news_results]
        return filtered_news[:10]
    except Exception as e:
        print(f"### SerpAPI News Fetch Error: {e} ###")
        return None

def get_fear_and_greed_index(limit=30):
    """alternative.me API를 통해 공포 탐욕 지수를 가져옵니다."""
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
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

# ==============================================================================
# STEP 2 & 3: AI 분석 요청 함수 수정 (이미지 데이터 및 프롬프트 수정)
# ==============================================================================
def generate(upbit_client):
    """필요한 모든 데이터를 수집하고 AI에게 투자 결정을 요청하는 함수"""
    # 현재 스크립트 파일이 위치한 디렉토리 경로를 가져옵니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 위 경로와 파일명을 합쳐서 전체 저장 경로를 생성합니다.
    screenshot_filename = "upbit_chart_for_ai.png"
    screenshot_path = os.path.join(script_dir, screenshot_filename)
    chart_image_part = None

    try:
        # 0. 차트 이미지 캡처
        chart_url = "https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC"
        if capture_chart_screenshot(chart_url, screenshot_path):
            # 성공적으로 캡처한 이미지를 AI에게 보낼 Part 객체로 변환
            with open(screenshot_path, 'rb') as f:
                image_bytes = f.read()
            chart_image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png'
            )
        else:
            print("### 차트 이미지를 분석에서 제외합니다. ###")

        # 1. 내 자산 정보
        my_balances = [b for b in upbit_client.get_balances() if b['currency'] in ['KRW', 'BTC']]

        # 2. 오더북
        orderbook = pyupbit.get_orderbook('KRW-BTC')

        # 3. 차트 데이터 (raw) 및 보조지표 추가
        df_day = add_technical_indicators(pyupbit.get_ohlcv('KRW-BTC', count=60, interval="day").copy())
        df_hour = add_technical_indicators(pyupbit.get_ohlcv('KRW-BTC', count=60, interval="minute60").copy())

        # 4. 공포 탐욕 지수
        fng_data = get_fear_and_greed_index(limit=30) or []

        # 5. 최신 뉴스 헤드라인
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        news_headlines = get_news_headlines(serpapi_key) if serpapi_key else []

        # 6. AI에게 보낼 프롬프트 데이터 (텍스트 부분) - 이미지 분석 요청 문구 추가
        prompt_data = f"""
        As a top-tier crypto analyst, synthesize all of the following data points to make a single, decisive investment call (buy/sell/hold) for KRW-BTC.

        **CRITICAL: You MUST visually analyze the attached chart image.** Look for candlestick patterns (e.g., doji, hammer), chart patterns (e.g., head and shoulders, flags), and indicator shapes (e.g., Bollinger Band squeezes/expansions, RSI divergence) that are not present in the raw data below. The visual analysis is a primary factor in your decision.

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
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path) # 에러 발생 시 임시 이미지 파일 삭제
        return None

    # ==============================================================================
    # STEP 4: AI API 호출 부분 수정 (이미지 Part 추가)
    # ==============================================================================
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # AI에게 전달할 콘텐츠 리스트 구성
    user_parts = []
    if chart_image_part:
        user_parts.append(chart_image_part) # 이미지 Part 추가
    user_parts.append(types.Part.from_text(text=prompt_data)) # 텍스트 Part 추가

    contents = [
        types.Content(role="user", parts=user_parts),
        # --- 기존의 few-shot 예제는 그대로 유지 ---
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""{\"decision\":\"buy\",\"reason\":\"Visually confirmed a Bollinger Band squeeze on the attached hourly chart, suggesting imminent volatility. This is coupled with a rising RSI and positive news sentiment, creating a strong buy signal before a potential breakout.\"}"""),
            ],
        ),
    ]

    # ... (기존 시스템 지침 및 설정은 동일하게 유지) ...
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""You are a world-class crypto analyst. Your primary goal is to synthesize qualitative data (news), quantitative data (charts, indicators), and **visual chart analysis (the attached image)** to make a single, actionable investment decision (buy, sell, or hold). Your reasoning must demonstrate how these different data types support each other. Prioritize major news events and clear visual patterns on the chart. Respond in JSON format.

Response Example:
{\"decision\":\"sell\",\"reason\":\"The attached chart clearly shows a head and shoulders pattern, a strong bearish visual indicator. This is confirmed by the F&G index dropping sharply, indicating market panic.\"}
"""),
        ],
    )

    try:
        response = client.generate_content(
            model="gemini-1.5-flash", # 이미지 분석을 위해 1.5 버전 이상 모델 권장
            contents=contents,
            generation_config=generate_content_config,
        )
        return response.text
    except Exception as e:
        print(f"### AI Response Error: {e} ###")
        return None
    finally:
        # 분석이 끝난 임시 이미지 파일 삭제
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)
            print(f"🗑️ 임시 스크린샷 파일 '{screenshot_path}'을(를) 삭제했습니다.")


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
            print("### Buy Order Executed! ###"); print(buy_result)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif decision == "sell":
        btc_balance = upbit_client.get_balance("KRW-BTC")
        if btc_balance and btc_balance > 0:
            current_price = pyupbit.get_current_price("KRW-BTC")
            if current_price and (btc_balance * current_price > 5000):
                sell_result = upbit_client.sell_market_order("KRW-BTC", btc_balance)
                print("### Sell Order Executed! ###"); print(sell_result)
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
    print("### 👁️ AI Trading Bot Started with Visual Chart Analysis ###")

    while True:
        ai_response = generate(upbit)
        transaction(ai_response, upbit)
        # 업비트 API는 요청 제한이 있으므로 실제 운영 시에는 더 긴 대기 시간을 권장합니다.
        print("\n--- Waiting for next cycle (60s) ---\n")
        time.sleep(60)