import torch 


DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


# Using ensemble retriever for retrieval
# rank_bm25 설치를 해야함
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': DEVICE}
encode_kwargs = { 'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



loaders = [
    PyPDFLoader("data/[홈넘버메타] 간략 소개서_24011501(2).pdf"),
    PyPDFLoader("data/1.김수한_전문가오피니언_중국+제4차+경제총조사(普査)+주요+내용+및+시사점.pdf")
]

docs = []

for loader in loaders:
    docs.extend(loader.load_and_split())


# Text Splitter 
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
texts =  splitter.split_documents(docs)

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2

faiss_vectorstore = FAISS.from_documents(texts, embedding=ko_embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# Init ensemble Retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights= [0.5, 0.5])

docs = ensemble_retriever.invoke("경제 정책")
for i in docs:
    print(i.metadata)
    print(":")
    print(i.page_content)
    print("-"*100)

from langchain.chains import RetrievalQA 
from langchain_community.chat_models import ChatOllama

chatOllama = ChatOllama(model='llama3:8b')
qa = RetrievalQA.from_chain_type(chain_type = 'stuff', llm = chatOllama, retriever = ensemble_retriever, return_source_documents = True)
query = "중국의 경제 정책 변경점을 한국어로 번역하여 알려줘."
result = qa(query)
print(result['result'])