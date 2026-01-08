import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient
from datetime import datetime, timedelta
import re

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------- WEATHER TOOL ----------------
@tool
def tool1_weather(query: str) -> str:
    """Weather tool"""
    try:
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
@tool
def tool2_stock(query: str) -> str:
    """Stock tool"""
    try:
        q = query.lower()
        parts = re.findall(r"([a-zA-Z.]+)", q)
        if not parts:
            return "No stock symbol detected."

        symbol = parts[0].upper()

        if "last week" in q or "historical" in q:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            )
            data = requests.get(url).json()
            ts = data.get("Time Series (Daily)")
            if not ts:
                return f"No historical data for {symbol}"

            dates = sorted(ts.keys(), reverse=True)[:7]
            return "\n".join(f"{d}: {ts[d]['4. close']}" for d in dates)

        else:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            )
            data = requests.get(url).json()
            quote = data.get("Global Quote", {})
            return f"{symbol} price: {quote.get('05. price', 'N/A')} USD"

    except Exception as e:
        return f"Stock error: {e}"


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
            if "content" in r:
                answer.append(f"- {r['content']}")

        if not answer:
            return "No relevant information found."

        return "🔍 Search Results:\n" + "\n".join(answer)

    except Exception as e:
        return f"Tavily error: {e}"
