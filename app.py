import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from tools import tool1_weather, tool2_stock

# -----------------------------
# ENV + LLM
# -----------------------------
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

tools = [tool1_weather, tool2_stock]

# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "Use tools ONLY for weather or stock-related questions. "
        "For general questions, answer directly."
    ),
    MessagesPlaceholder("messages"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Multi-Tool AI Chatbot", layout="centered")
st.title("🤖 Multi-Tool AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("🤔 Thinking..."):
        response = agent_executor.invoke({
            "messages": st.session_state.chat_history
        })
        bot_reply = response["output"]

    st.session_state.chat_history.append(AIMessage(content=bot_reply))

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").markdown(msg.content)
