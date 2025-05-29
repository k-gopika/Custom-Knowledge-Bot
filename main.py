import os
import time
import pinecone
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from langchain_ollama import ChatOllama
from llama_index.llms.langchain import LangChainLLM
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    )
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec # Pinecone vector store
from llama_index.core.node_parser import SimpleNodeParser

def load_documents(filepath):
    loader = PyMuPDFReader()
    return loader.load(filepath)

def Setup_Model():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2")  # <--- changed here to gemma:2b
    llm_wrapped = LangChainLLM(llm)
    Settings.llm = llm_wrapped
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in .env file")
PINECONE_INDEX_NAME = "llama-index"
PINECONE_ENV = "us-east-1"
# Pinecone v3 client
pc = Pinecone(api_key=PINECONE_API_KEY)
region_spec = ServerlessSpec(cloud='aws', region='us-east-1')



from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore


def GetIndex(filepath, force_rebuild=False):
    persist_dir = "./storage"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=region_spec,
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index)

    if not force_rebuild:
        try:
            print("Trying to load existing index from storage...")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Failed to load index: {e}. Proceeding to rebuild...")

    # Explicitly define all storage components
    documents = load_documents(filepath)
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store,
      #  kvstore=kvstore,
        persist_dir=persist_dir
    )

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    return index




def chat(filepath):
    Setup_Model()
    index = GetIndex(filepath)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    retriever = index.as_retriever(similarity_top_k=8)

    print("\n🤖 Smart bot is ready. Ask questions about the document. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Ask: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            total_start = time.time()

            # 1. Retrieve nodes manually
            retrieval_start = time.time()
            retrieved_nodes = retriever.retrieve(user_input)
            retrieval_time = time.time() - retrieval_start

            # 2. Generate response from retrieved nodes
            response_start = time.time()
            chat_engine = ContextChatEngine.from_defaults(
                retriever=retriever,
                memory=memory,
                llm=Settings.llm
            )
            response = chat_engine.chat(user_input)
            response_time = time.time() - response_start

            total_time = time.time() - total_start

            print("→", response.response, "\n")
            print(f" Timings:")
            print(f"    Retrieval time:  {retrieval_time:.2f}s")
            print(f"    LLM response time: {response_time:.2f}s")
            print(f"    Total time:      {total_time:.2f}s\n")

        except KeyboardInterrupt:
            print("\n Interrupted. Goodbye!")
            break




if __name__ == "__main__":
    chat("harry_potter_50pg.pdf")