import torch 

from langchain.retrievers import ParentDocumentRetriever

# 저장소에 키/값을 Dict 형태로 저장
# Parent와 Child 가 Dict 형태로 엮여져있어야 문서를 찾아갈 수 있음..
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


loaders = [
    PyPDFLoader("data/[홈넘버메타] 간략 소개서_24011501(2).pdf"),
    PyPDFLoader("data/1.김수한_전문가오피니언_중국+제4차+경제총조사(普査)+주요+내용+및+시사점.pdf")
]

docs = []

for loader in loaders:
    docs.extend(loader.load_and_split())

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': DEVICE}
encode_kwargs = { 'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

# Text Splitter 
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=ko_embedding
)


## 본문의 CHUNK가 너무 길다면... parent_splitter 를 적용한다.

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)

# Storage Layer for parent Documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore = vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(docs, ids=None)
sub_docs = vectorstore.similarity_search("자동차")

print("글 길이: {}\n\n".format(len(sub_docs[0].page_content)))
print(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("자동차")

print("글 길이: {}\n\n".format(len(retrieved_docs[0].page_content)))
print(retrieved_docs[0].page_content)
