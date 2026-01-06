import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
import yfinance as yf
from tavily import TavilyClient

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------- WEATHER TOOL ----------------
@tool
def tool1_weather(query: str) -> str:
    """
    Use for weather-related questions only.
    """
    try:
        import re
        from datetime import datetime, timedelta

        q = query.lower()
        is_forecast = "forecast" in q or "7-day" in q
        is_yesterday = "yesterday" in q

        cities = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", query)
        city = cities[0] if cities else query.strip()

        if is_yesterday:
            yday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
            url = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={city}&dt={yday}"
            data = requests.get(url).json()
            day = data["forecast"]["forecastday"][0]["day"]
            return f"Yesterday in {city}: {day['avgtemp_c']}°C, {day['condition']['text']}"

        if is_forecast:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days=7"
            data = requests.get(url).json()
            return "\n".join(
                f"{d['date']}: {d['day']['avgtemp_c']}°C, {d['day']['condition']['text']}"
                for d in data["forecast"]["forecastday"]
            )

        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        data = requests.get(url).json()
        return f"{city}: {data['current']['temp_c']}°C, {data['current']['condition']['text']}"

    except Exception as e:
        return f"Weather error: {e}"


# ---------------- STOCK TOOL ----------------
# ---------------- STOCK TOOL ----------------
@tool
def tool2_stock(query: str) -> str:
    """
    Stock Tool using Alpha Vantage.
    Returns FINAL answers only.
    """
    try:
        import re

        q = query.lower()
        parts = re.findall(r"([a-zA-Z.]+)", q)

        if not parts:
            return "FINAL ANSWER: No valid stock symbol detected in your query."

        symbol = parts[0].upper()

        # -------- Historical prices --------
        if "last week" in q or "historical" in q:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={symbol}"
                f"&apikey={ALPHA_VANTAGE_KEY}"
            )

            data = requests.get(url, timeout=10).json()

            if "Note" in data:
                return f"FINAL ANSWER: Alpha Vantage rate limit reached. Try again later."

            ts = data.get("Time Series (Daily)")
            if not ts:
                return f"FINAL ANSWER: Historical data not available for {symbol}."

            dates = sorted(ts.keys(), reverse=True)[:7]
            result = f"FINAL ANSWER:\n📊 Last 7 Days Prices for {symbol}:\n"

            for date in dates:
                result += f"{date}: {ts[date]['4. close']}\n"

            return result.strip()

        # -------- Current price --------
        else:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=GLOBAL_QUOTE"
                f"&symbol={symbol}"
                f"&apikey={ALPHA_VANTAGE_KEY}"
            )

            data = requests.get(url, timeout=10).json()

            if "Note" in data:
                return "FINAL ANSWER: Alpha Vantage rate limit reached. Try again later."

            quote = data.get("Global Quote")

            if not quote or not quote.get("05. price"):
                return f"FINAL ANSWER: Current price not available for {symbol}."

            return f"FINAL ANSWER:\n📈 Current Price of {symbol}: {quote['05. price']} USD"

    except Exception as e:
        return f"FINAL ANSWER: Stock API error occurred – {str(e)}"



# ---------------- GENERAL QA (TAVILY) ----------------
@tool
def tool3_general_search(query: str) -> str:
    """
    Use for general knowledge questions.
    Fetches answer using Tavily web search.
    """
    try:
        results = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )

        if not results or "results" not in results:
            return "No relevant information found."

        answer = []
        for r in results["results"]:
            answer.append(f"- {r['content']}")

        return "🔍 Search Results:\n" + "\n".join(answer)

    except Exception as e:
        return f"Tavily error: {e}"
