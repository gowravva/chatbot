import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import HumanMessage, AIMessage

from tools import tool1_weather, tool2_stock, tool3_general_search

# ----------------------------- ENV + LLM -----------------------------
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ----------------------------- TOOLS -----------------------------
llm_tools = [
    Tool(name="Weather Tool", func=tool1_weather, description="Use for weather-related questions."),
    Tool(name="Stock Tool", func=tool2_stock, description="Use for stock-related questions."),
    Tool(name="General QA Tool", func=tool3_general_search, description="Use for general knowledge questions via Tavily.")
]

# ----------------------------- AGENT -----------------------------
agent_executor = initialize_agent(
    tools=llm_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ----------------------------- STREAMLIT UI -----------------------------
st.set_page_config(page_title="Multi-Tool AI Chatbot", layout="centered")
st.title("🤖 Multi-Tool AI Chatbot")
st.markdown("Ask about **weather**, **stock prices**, or general questions!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("🤔 Thinking..."):
        try:
            response = agent_executor.run(input=user_input)
            bot_reply = response
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}"

    st.session_state.chat_history.append(AIMessage(content=bot_reply))

# Display chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content.replace("\n", "  \n"))
