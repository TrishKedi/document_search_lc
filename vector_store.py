import os
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_db"

def create_vectorstore(text):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    return vectorstore

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
