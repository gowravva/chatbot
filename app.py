import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# 👉 ChromaDB imports (ADDED)
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from tools import tool1_weather, tool2_stock, tool3_general_search

# ----------------------------- ENV + LLM -----------------------------
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ----------------------------- CHROMADB SETUP (ADDED) -----------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding
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

    # ------------------ RETRIEVE CONTEXT FROM CHROMADB ------------------
    similar_docs = chroma_db.similarity_search(user_input, k=3)
    memory_context = "\n".join([doc.page_content for doc in similar_docs])

    final_prompt = f"""
Previous context:
{memory_context}

User question:
{user_input}
"""

    with st.spinner("🤔 Thinking..."):
        try:
            response = agent_executor.run(input=final_prompt)
            bot_reply = response
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}"

    st.session_state.chat_history.append(AIMessage(content=bot_reply))

    # ------------------ STORE CONVERSATION IN CHROMADB ------------------
    chroma_db.add_documents([
        Document(page_content=f"User: {user_input}\nAssistant: {bot_reply}")
    ])

# Display chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content.replace("\n", "  \n"))
