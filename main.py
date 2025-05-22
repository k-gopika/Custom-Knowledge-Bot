import os.path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter   
from langchain_ollama import ChatOllama
from llama_index.llms.langchain import LangChainLLM
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine

def load_documents(filepath):
    loader = PyMuPDFReader()
    return loader.load(file_path=filepath)

# Set up embeddings
def Setup_Model():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2")  # Adjust model size as needed
    llm_wrapped = LangChainLLM(llm)
    Settings.llm = llm_wrapped
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

#Storing index
def GetIndex(filepath, Index_stored = "./storage"):
    if not os.path.exists(Index_stored):
        print("Index not found, Rebuilding...")
        documents = load_documents(filepath)
        index = VectorStoreIndex.from_documents(documents, settings=Settings)
        index.storage_context.persist(persist_dir=Index_stored)
    else:
        print("Using cached index.")
        storage_context = StorageContext.from_defaults(persist_dir=Index_stored)
        index = load_index_from_storage(storage_context)
    return index

def chat(filepath):
    Setup_Model()
    index = GetIndex(filepath)
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    
    # Use ContextChatEngine which supports memory
    chat_engine = ContextChatEngine.from_defaults(
        retriever=index.as_retriever(),
        memory=memory,
        llm=Settings.llm
    )

    print("\nCustom bot ready. Ask anything, or type 'exit' to quit.\n")
    while True:
        query = input("Ask: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print(" Goodbye!")
            break

        response = chat_engine.chat(query)
        print("→", response.response, "\n")
        print("Source Chunks Used:")
        for source_node in response.source_nodes:
            print("— Chunk Score:", source_node.score)
            print(source_node.node.get_content().strip(), "\n")

if __name__ == "__main__":
    chat("harry_potter_50pg.pdf")