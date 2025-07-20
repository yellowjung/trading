import os
import pyupbit
from dotenv import load_dotenv
from google import genai
from google.genai import types
import base64
import time
import json

load_dotenv()

# 1. 업비트 차트 데이터 가져오기 (30일)

df = pyupbit.get_ohlcv('KRW-BTC', count = 30, interval="day")
# 2. AI에게 데이터 제공하고 판단 받기
def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=df.to_json()),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Assessing the Bitcoin Data**

I've begun dissecting the Bitcoin chart data, noting its structure. The JSON object containing the open, high, low, close, volume, and value data points for each timestamp has been thoroughly examined. I am now concentrating on the initial price trends and volume patterns. I'm seeing several key points I need to examine to move forward.


**Investigating Price Movements**

I'm now diving deeper into the close price trends. I've pinpointed key price levels and identified the main trend as an upward trajectory with considerable consolidation. My analysis of the volume during the significant upward move supports this trend. I'm noting significant volume spikes. I will now integrate this data to construct a trade thesis.


**Examining Volume and Trends**

My focus has shifted to the volume patterns in relation to price action. I've noted a strong uptrend followed by a consolidation period with declining volume. This decrease in volume during consolidation after a significant rise suggests a potential continuation pattern. I'm now comparing the volume data to previous patterns to formulate a buy/sell/hold decision.


**Synthesizing a Trade Thesis**

The recent data analysis has yielded a bullish outlook. I've noted a strong uptrend followed by a consolidation phase. This consolidation, marked by declining volume, reinforces my belief in an impending continuation of the upward trend. Buying now seems prudent, with the consolidation phase presenting a solid entry opportunity before a potential breakout to new highs.


**Evaluating a Buy Decision**

I've carefully analyzed the price movements, finding a significant uptrend followed by a consolidation phase. Decreasing volume during this consolidation solidifies my belief in a bullish continuation. The confluence of these indicators suggests a strategic buy opportunity. Buying now, before a potential breakout, seems like a viable plan.


"""),
                types.Part.from_text(text="""{\"decision\":\"buy\",\"reason\":\"The chart shows a strong uptrend followed by a consolidation phase on declining volume. This pattern is often a bullish continuation signal, suggesting a high probability of the price breaking out to new highs.\"}"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""You are an expert in Bitcoin investing. Tell me whether to buy, sell, or hold at the moment based on the chart data provided. response in json format

Response Example:
{\"decision\":\"buy\",\"reason\":\"some technical reason\"}
{\"decision\":\"sell\",\"reason\":\"some technical reason\"}
{\"decision\":\"hold\",\"reason\":\"some technical reason\"}"""),
        ],
    )
    response_parts = []
    for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
    ):
        response_parts.append(chunk.text)

    return "".join(response_parts)

def transaction(ai_decision):
    print("### AI Decision: ", ai_decision["decision"].upper(), "###")
    print(f"### Reason: {ai_decision['reason']} ###")
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access=access, secret=secret)

    if ai_decision["decision"] == "buy":
        my_krw = upbit.get_balance("KRW")
        if my_krw * 0.9995 > 5000:
            print("### Buy Order Executed! ###")
            print(upbit.buy_market_order("KRW-BTC", upbit.get_balance(my_krw * 0.9995)))
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif ai_decision["decision"] == "sell":
        my_btc = upbit.get_balance("KRW-BTC")
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")["orderbook_units"][0]["ask_price"]
        if my_btc * current_price > 5000:
            print("### Sell Order Executed! ###")
            print(upbit.sell_market_order("KRW-BTC", upbit.get_balance("KRW-BTC")))
        else:
            print("### Sell Order Failed: Insufficient BTC (less than 5000 BTC) ###")
    elif ai_decision["decision"] == "hold":
        print("### Hold Position ###")

if __name__ == "__main__":
    while True:
        time.sleep(10)
        transaction(json.loads(generate()))


