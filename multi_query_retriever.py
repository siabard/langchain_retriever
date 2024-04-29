import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


# Load blog post
loader = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': DEVICE}
encode_kwargs = { 'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)

# MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama

question = '삼성전자 갤럭시 S24는 어떻게 예정되는가?'
llm = ChatOllama(temperature=0, model='EEVE-Korean-10.8B')
retriever_from_llm = MultiQueryRetriever.from_llm(llm=llm, retriever=vectordb.as_retriever())

# Logging
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(question)
