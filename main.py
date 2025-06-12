# app.py
import os
import sys
import random
import asyncio
import streamlit as st
from dotenv import load_dotenv

from config import PINECONE_INDEX_NAME, PINECONE_ENV, PDF_PATH
from tarot import draw_cards
from utils import build_tarot_prompt, classify_intent

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.readers.file import PyMuPDFReader
from langchain_ollama import ChatOllama
from llama_index.llms.langchain import LangChainLLM

# Async fix for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    st.error("Pinecone API key missing. Please check your .env file.")
    st.stop()

@st.cache_resource(show_spinner="Loading model and index...")
def load_engine_and_index():
    # Set up model and embedding
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2")
    Settings.llm = LangChainLLM(llm)
    Settings.embed_model = embed_model

    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index)

    persist_dir = "./storage"
    os.makedirs(persist_dir, exist_ok=True)

    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
    except Exception:
        documents = PyMuPDFReader().load(PDF_PATH)
        docstore = SimpleDocumentStore()
        index_store = SimpleIndexStore()
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            index_store=index_store,
            persist_dir=persist_dir
        )
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir=persist_dir)

    retriever = index.as_retriever(similarity_top_k=8)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    chat_engine = ContextChatEngine.from_defaults(retriever=retriever, memory=memory, llm=Settings.llm)
    return chat_engine


# Streamlit UI
st.set_page_config(page_title="\ud83d\udcc4 Document Chatbot", layout="wide")
st.title("\ud83d\udcc4 Document Q&A Chatbot")
st.markdown("Ask questions about tarot cards. The model uses interpretations from a tarot knowledge base PDF.")

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = load_engine_and_index()
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the document:")

if user_input:
    with st.spinner("Thinking..."):
        intent = classify_intent(user_input)
        st.write(f"**\ud83d\udd0d Detected Intent:** `{intent}`")

        drawn_cards = draw_cards()
        drawn_card_names = [f"{c['name']} ({c['orientation']})" for c in drawn_cards]
        st.write("**\ud83c\udccf Drawn Cards:**", ", ".join(drawn_card_names))

        prompt = build_tarot_prompt(user_input, intent, drawn_cards)
        response = st.session_state.chat_engine.chat(prompt)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response.response))

if st.session_state.chat_history:
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")
