__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import asyncio
import warnings

from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding import CustomEmbedding
from model import get_sr_llm_model
from prompt import get_prompt
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

def rag_chain():
    # Langchain에서 제공하는 ConfluenceLoader를 사용해 RAG를 하고자 하는 컨플루언스 페이지를 가져옵니다.
    print("load confluence page...")
    load_dotenv()

    confluence_page_ids = ["351246149"]  # 검색하고자 하는 컨플루언스 페이지 아이디 리스트

    loader = ConfluenceLoader(
            url="https://devops.sdsdev.co.kr/confluence", token=os.getenv("CONFLUENCE_TOKEN"), page_ids=confluence_page_ids
        )

    docs = loader.load()
    # Text Splitter 를 사용해 문서를 일정 사이즈로 분할합니다.
    # chunk_size는 텍스트를 분할할 크기를 나타내며, chunk_overlap은 청크들 간에 겹치는 부분의 양을 나타냅니다.
    # 각 값은 상황에 따라 적절히 조정합니다.
    print("split text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 분할한 텍스트 문서를 Vector DB에 저장합니다.
    # 사내 임베딩 모델을 사용하여 텍스트를 임베딩 합니다.
    print("embedding...")
    #vectorstore = Chroma.from_documents(documents=splits, embedding=CustomEmbedding())
    vectorstore = FAISS.from_documents(documents=splits, embedding=CustomEmbedding())
    # VectorStore를 검색기로서 사용합니다.
    retriever = vectorstore.as_retriever()

    # llm 모델에 전달할 프롬프트를 가져옵니다.
    prompt = get_prompt()

    # 사내 llm 모델을 가져옵니다.
    llm = get_sr_llm_model()

    def format_docs(_docs):
        return "\n\n".join(doc.page_content for doc in _docs)

    # LCEL(LangChain Expression Language) 방식을 사용하여 Chain 을 구성합니다.
    # LCEL 에 대한 설명은 다음 링크를 참고합니다.i
    # https://python.langchain.com/docs/expression_language/
    print("get chain")
    _rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return _rag_chain


if __name__ == "__main__":
    async def stream_llm(chain, query):
        async for event in chain.astream_events(query, version="v1"):
            kind = event["event"]

            if kind == "on_llm_stream" and event["data"]["chunk"]:
                print(event["data"]["chunk"], end="")


    _chain = rag_chain()
    while True:
        print()
        print()
        _query = input("You: ")
        print("AI: ", end="")
        asyncio.run(stream_llm(_chain, _query))
