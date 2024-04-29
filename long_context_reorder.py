from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import torch 
from langchain_community.embeddings import HuggingFaceEmbeddings


DEVICE = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


texts = [
    "바스켓볼은 훌륭한 스포츠입니다.",
    "플라이 미 투 더 문은 제가 가장 좋아하는 노래 중 하나입니다.",
    "셀틱스는 제가 가장 좋아하는 팀입니다.",
    "보스턴 셀틱스에 관한 문서입니다.", "보스턴 셀틱스는 제가 가장 좋아하는 팀입니다.",
    "저는 영화 보러 가는 것을 좋아해요",
    "보스턴 셀틱스가 20점차로 이겼어요",
    "이것은 그냥 임의의 텍스트입니다.",
    "엘든 링은 지난 15 년 동안 최고의 게임 중 하나입니다.",
    "L. 코넷은 최고의 셀틱스 선수 중 한 명입니다.",
    "래리 버드는 상징적 인 NBA 선수였습니다.",
]

# VectorDB
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': DEVICE}
encode_kwargs = { 'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# Create a retriever
retriever = Chroma.from_texts(texts, embedding=ko_embedding).as_retriever(
    search_kwargs={"k": 10}
)
query = "셀틱스에 대해 어떤 이야기를 들려주시겠어요?"

# Get relevant documents ordered by relevance score
docs = retriever.invoke(query)


# Reordering
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)


from langchain.chains import RetrievalQA 
from langchain_community.chat_models import ChatOllama



document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

template = """Given this text extracts:
-----
{context}
-----
Please answer the following question in Korean:
{query}"""
prompt = PromptTemplate(
    template=template, input_variables=["context", "query"]
)
openai = ChatOllama(model_name="llama3:8b", temperature = 0)

llm_chain = LLMChain(llm=openai, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="context"
)

reordered_result = chain.run(input_documents=reordered_docs, query=query)
result = chain.run(input_documents=docs, query=query)

print(reordered_result)
print("-"*100)
print(result)