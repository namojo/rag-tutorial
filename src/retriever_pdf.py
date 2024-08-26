# Chroma 사용시에 sqlite3의 버전 문제가 발생하므로 삽입된 코드
# pip install pypdf
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from model import get_sr_llm_model
from embedding import CustomEmbedding

from langchain.prompts import PromptTemplate

import warnings

warnings.filterwarnings("ignore")

# 단계 1: 문서 로드(Load Documents)
# PDF 파일 로드. 파일의 경로 입력
# AI현황에 대한 pdf 문서
loader = PyPDFLoader("./data/AI_Brief.pdf")
docs = loader.load()


# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

splits = text_splitter.split_documents(docs)

# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=CustomEmbedding())

# 단계 4: 검색(Search)
# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 5: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)

# 단계 6: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = get_sr_llm_model()


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 단계 7: 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 단계 8: 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "AI저작권 문제에 대해 설명해주세요."
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")

# 유사성 검색
query = "AI저작권 문제에 대해 설명해주세요. "
retriever = vectorstore.as_retriever(search_type="similarity")
search_result = retriever.get_relevant_documents(query)
print(search_result[0].page_content)

# 유사성 검색으로 Threshold 0.8 이상인 것을 찾습니다.
#query = "AI저작권 문제에 대해 설명해주세요. "
#retriever = vectorstore.as_retriever(
#    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
#)
#search_result = retriever.get_relevant_documents(query)
#print(search_result[0].page_content)


# MMR 기법으로 검색
#query = "AI관련 현황에 대해 설명해주세요 "
#retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
#search_result = retriever.get_relevant_documents(query)
#print(search_result[0].page_content)