from datetime import datetime, timedelta

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS 

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': DEVICE}
encode_kwargs = { 'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

# Vectore store initialization
# ko_embedding 의 model 이 가지는 pooling layer에서 word_embedding_dimention 값을 embedding_size로 삼는다.
embedding_size = 768
index = faiss.IndexFlatL2(embedding_size)
faiss_store = FAISS(
    index=index, 
    embedding_function=ko_embedding,
    docstore=InMemoryDocstore({}), index_to_docstore_id={})
retriever = TimeWeightedVectorStoreRetriever(vectorstore = faiss_store, decay_rate=0.99, k=1)

yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents(
    [Document(page_content="영어는 훌륭합니다.", metadata={"last_accessed_at": yesterday})]
)

retriever.add_documents([Document(page_content="한국어는 훌륭합니다")])
retrieved_docs = retriever.invoke("영어가 좋아요")

print(retrieved_docs)