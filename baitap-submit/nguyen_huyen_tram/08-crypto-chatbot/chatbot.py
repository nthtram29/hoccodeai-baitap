import requests
import yfinance as yf
from pprint import pprint
import inspect
from pydantic import TypeAdapter
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def get_symbol(company: str) -> str:
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company, "country": "United States"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    
    res = requests.get(url, params=params, headers=headers)
    data = res.json()

    if not data.get("quotes"):
        return "Không tìm thấy mã chứng khoán cho công ty này."

    return data["quotes"][0]["symbol"]

def get_stock_price(symbol: str):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d", interval="1m")

    if hist.empty:
        return f"Không tìm thấy dữ liệu giá cổ phiếu cho mã {symbol}."

    latest = hist.iloc[-1]
    return {
        "timestamp": str(latest.name),
        "open": latest["Open"],
        "high": latest["High"],
        "low": latest["Low"],
        "close": latest["Close"],
        "volume": latest["Volume"]
    }
    
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": inspect.getdoc(get_symbol),
            "parameters": TypeAdapter(get_symbol).json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": inspect.getdoc(get_stock_price),
            "parameters": TypeAdapter(get_stock_price).json_schema(),
        },
    }
]

client = OpenAI(
    base_url = "https://api.together.xyz/v1",
    api_key = os.getenv("TOGETHER_API_KEY"),
)

def get_completion(messages):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages,
        tools=tools,
        temperature=0
    )
    return response

FUNCTION_MAP = {
        "get_symbol": get_symbol,
        "get_stock_price": get_stock_price
    }

messages = [
       {
            "role": "system",
            "content": "You are an extremely friendly, witty, and slightly mischievous assistant who loves to chat with users like an old friend. Your job is not just to help, but to make every conversation feel fun, engaging, and full of positive energy! Feel free to use humor, playful banter, and a lighthearted tone to make interactions enjoyable. If needed, use the provided tools to fetch accurate information. However, if a tool provides a result, you MUST use that result to answer the user. Do NOT ignore valid data from tools. Always provide a complete and entertaining answer based on the tool output. Most importantly, when delivering the final response to the user, always reply in Vietnamese, keeping the tone natural and friendly!"
        }
    ]

while True:
    question = input("\nNhập câu hỏi của bạn (hoặc nhập 'exit' để thoát): ").strip()
    
    if question.lower() == "exit":
        print("Thoát chương trình.")
        break
    
    messages.append(
        {"role": "user", "content": question}
    )
    
    response = get_completion(messages)
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    while finish_reason != "stop":
        
        tool_call = first_choice.message.tool_calls[0]
        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)
        
        if not tool_function:
            print("\nBot: Không tìm thấy chức năng phù hợp.")
            break
        
        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        response = get_completion(messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    print("\nBot:", first_choice.message.content)
    messages.append(
        {"role": "assistant", "content": first_choice.message.content}
    )
