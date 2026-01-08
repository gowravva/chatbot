import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from tools import tool1_weather, tool2_stock, tool3_general_search

load_dotenv()

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

TOOLS = {
    "weather": tool1_weather,
    "stock": tool2_stock,
    "search": tool3_general_search
}

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Multi-Tool AI Chatbot", layout="centered")
st.title("🤖 Multi-Tool AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask about weather, stocks, or anything...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        prompt = f"""
You are a router.
If question is about weather → respond: WEATHER
If about stocks → respond: STOCK
Else → respond: SEARCH

Question: {user_input}
"""
        decision = llm.invoke(prompt).content.lower()

        if "weather" in decision:
            reply = tool1_weather.invoke(user_input)
        elif "stock" in decision:
            reply = tool2_stock.invoke(user_input)
        else:
            reply = tool3_general_search.invoke(user_input)

    st.session_state.chat_history.append(AIMessage(content=reply))

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
