# 필요한 라이브러리 및 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings

# 표준 라이브러리 및 유틸리티 임포트
from typing import List, Union, Dict, Any, Optional  # 중복 임포트 제거 및 필요한 타입 추가
import fitz
import requests
import asyncio
import random
import textwrap
import numpy as np
from rank_bm25 import BM25Okapi

# 상수 정의
# 하드코딩된 URL을 상수로 추출
EMBEDDING_API_URL = 'http://sds-embed.serving.70-220-152-1.sslip.io/v1/models/embed:predict'


class CustomEmbedding(Embeddings):
    """
    SDS 임베딩 API를 사용하여 문서와 쿼리를 임베딩하는 클래스.
    
    LangChain의 Embeddings 인터페이스를 구현하여 LangChain 생태계와 호환되도록 합니다.
    """
    def __init__(self, api_url: str = EMBEDDING_API_URL):
        """
        CustomEmbedding 클래스의 초기화 함수.
        
        Args:
            api_url: 임베딩 API의 URL. 기본값은 EMBEDDING_API_URL 상수입니다.
        """
        self.embed_url = api_url

    def call_embed(self, url: str, texts: List[str]) -> List[List[float]]:
        """
        임베딩 API를 호출하여 텍스트 임베딩을 생성합니다.
        
        Args:
            url: 임베딩 API URL
            texts: 임베딩할 텍스트 목록
            
        Returns:
            생성된 임베딩 벡터 리스트
            
        Raises:
            Exception: API 호출 중 오류가 발생한 경우
        """
        try:
            data = {
                "instances": texts
            }
            response = requests.post(url=url, json=data)
            response.raise_for_status()  # 오류 응답일 경우 예외 발생
            result = response.json()
            return result['predictions']
        except Exception as e:
            # 오류 처리 추가
            print(f"임베딩 API 호출 중 오류 발생: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서텍스트를 임베딩하여 벡터로 변환합니다.
        
        Args:
            texts: 임베딩할 문서 텍스트 목록
            
        Returns:
            문서 임베딩 벡터 목록
        """
        embed_list = self.call_embed(url=self.embed_url, texts=texts)
        return embed_list

    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 텍스트를 임베딩하여 벡터로 변환합니다.
        
        Args:
            text: 임베딩할 쿼리 텍스트
            
        Returns:
            쿼리 임베딩 벡터
        """
        embed_list = self.call_embed(url=self.embed_url, texts=[text])
        return embed_list[0]


# 텍스트 처리 유틸리티 함수
def replace_t_with_space(list_of_documents: List[Any]) -> List[Any]:
    """
    문서 목록의 각 문서 내용에서 탭 문자('\t')를 공백으로 대체합니다.

    Args:
        list_of_documents: 'page_content' 속성을 가진 문서 객체 목록

    Returns:
        탭 문자가 공백으로 대체된 문서 목록
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # 탭을 공백으로 대체
    return list_of_documents


def text_wrap(text: str, width: int = 120) -> str:
    """
    텍스트를 지정된 너비로 줄바꿈 처리합니다.

    Args:
        text: 줄바꿈 처리할 입력 텍스트
        width: 텍스트를 줄바꿈할 너비

    Returns:
        줄바꿈 처리된 텍스트
    """
    return textwrap.fill(text, width=width)


# 문서 인코딩 및 벡터 저장소 생성 함수
def encode_document(source: Union[str, List[str]], 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200, 
                 is_file: bool = True) -> FAISS:
    """
    문서(파일 또는 문자열)를 인코딩하여 벡터 저장소를 생성합니다.
    
    이 함수는 기존의 encode_pdf와 encode_from_string 함수를 통합했습니다.
    
    Args:
        source: 문서 소스(파일 경로 또는 문자열 콘텐츠)
        chunk_size: 각 텍스트 청크의 크기
        chunk_overlap: 연속된 청크 간의 중복 정도
        is_file: 소스가 파일인지 여부
        
    Returns:
        인코딩된 문서 콘텐츠가 포함된 FAISS 벡터 저장소
    """
    # 문서 로드
    if is_file:
        # PDF 파일에서 문서 로드
        loader = PyPDFLoader(source)
        documents = loader.load()
    else:
        # 문자열 콘텐츠로부터 문서 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([source])
        
        # 관련성 점수 메타데이터 추가
        for doc in documents:
            doc.metadata['relevance_score'] = 1.0
        
        return create_vector_store(documents)
        
    # 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    return create_vector_store(cleaned_texts)


def create_vector_store(documents: List[Any]) -> FAISS:
    """
    문서 목록으로부터 FAISS 벡터 저장소를 생성합니다.
    
    Args:
        documents: 벡터 저장소에 저장할 문서 목록
        
    Returns:
        생성된 FAISS 벡터 저장소
    """
    # 임베딩 및 벡터 저장소 생성
    embeddings = CustomEmbedding()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore


# 이전 함수와 호환성을 위한 래퍼 함수들
def encode_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    PDF 파일을 인코딩하여 벡터 저장소를 생성합니다.
    
    이 함수는 호환성을 위해 유지되며, 내부적으로는 encode_document를 호출합니다.
    
    Args:
        path: PDF 파일 경로
        chunk_size: 각 텍스트 청크의 크기
        chunk_overlap: 연속된 청크 간의 중복 정도
        
    Returns:
        인코딩된 책 콘텐츠가 포함된 FAISS 벡터 저장소
    """
    return encode_document(path, chunk_size, chunk_overlap, is_file=True)


def encode_from_string(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    문자열 콘텐츠를 인코딩하여 벡터 저장소를 생성합니다.
    
    이 함수는 호환성을 위해 유지되며, 내부적으로는 encode_document를 호출합니다.
    
    Args:
        content: 인코딩할 문자열 콘텐츠
        chunk_size: 각 텍스트 청크의 크기
        chunk_overlap: 연속된 청크 간의 중복 정도
        
    Returns:
        인코딩된 콘텐츠가 포함된 FAISS 벡터 저장소
    """
    return encode_document(content, chunk_size, chunk_overlap, is_file=False)


# 문서 검색 및 쿼리 관련 함수
def retrieve_context_per_question(question: str, chunks_query_retriever: Any) -> List[str]:
    """
    주어진 질문에 대해 관련 문서를 검색합니다.
    
    Args:
        question: 검색할 질문
        chunks_query_retriever: 문서 검색에 사용할 검색기
        
    Returns:
        검색된 문서의 콘텐츠 목록
    """
    try:
        # LangChain 1.0+ 에서는 invoke 메서드 사용
        docs = chunks_query_retriever.invoke(question)
        
        # 문서 콘텐츠 추출
        context = [doc.page_content for doc in docs]
        
        return context
    except Exception as e:
        print(f"문서 검색 중 오류 발생: {str(e)}")
        # 예외 발생 시 빈 리스트 반환
        return []


# 문서 처리 및 PDF 관련 함수
def read_pdf_to_string(path: str) -> str:
    """
    PDF 문서를 읽어 텍스트로 변환합니다.

    Args:
        path: PDF 문서의 경로

    Returns:
        PDF의 모든 페이지의 텍스트 콘텐츠를 합쳐놓은 문자열

    이 함수는 'fitz' 라이브러리(PyMuPDF)를 사용하여 PDF를 열고, 각 페이지를 처리합니다.
    """
    try:
        # 지정된 경로에 있는 PDF 문서 열기
        doc = fitz.open(path)
        content = ""
        # 문서의 각 페이지 반복
        for page_num in range(len(doc)):
            # 현재 페이지 가져오기
            page = doc[page_num]
            # 현재 페이지에서 텍스트 추출 및 콘텐츠에 추가
            content += page.get_text()
        return content
    except Exception as e:
        print(f"PDF 읽기 오류: {str(e)}")
        return ""


# 문서 조회 및 디스플레이 함수
def show_context(context: List[str]) -> None:
    """
    컨텍스트 목록의 내용을 출력합니다.

    Args:
        context: 출력할 컨텍스트 항목 목록

    각 컨텍스트 항목을 순서를 표시하는 헤더와 함께 출력합니다.
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c)
        print("\n")


# Q&A 관련 클래스 및 함수
class QuestionAnswerFromContext(BaseModel):
    """
    쿼리에 대한 답변을 생성하는 모델 클래스.
    
    Attributes:
        answer_based_on_content (str): 컨텍스트를 기반으로 생성된 답변.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm: Any) -> Any:
    """
    질문에 대한 답변을 생성하는 체인을 생성합니다.
    
    Args:
        llm: 사용할 언어 모델
        
    Returns:
        질문-답변 체인
    """
    # 언어 모델 초기화
    question_answer_from_context_llm = llm

    # 사고 과정을 위한 프롬프트 템플릿 정의
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # 프롬프트 템플릿 객체 생성
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # 프롬프트 템플릿과 언어 모델을 결합하여 체인 생성
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question: str, context: List[str], question_answer_from_context_chain: Any) -> Dict[str, Any]:
    """
    주어진 컨텍스트를 기반으로 질문에 답변합니다.

    Args:
        question: 답변할 질문
        context: 답변에 사용할 컨텍스트
        question_answer_from_context_chain: 사용할 질문-답변 체인

    Returns:
        답변, 컨텍스트, 질문을 포함하는 딕셔너리
    """
    try:
        input_data = {
            "question": question,
            "context": context
        }
        print("Answering the question from the retrieved context...")

        output = question_answer_from_context_chain.invoke(input_data)
        answer = output.answer_based_on_content
        return {"answer": answer, "context": context, "question": question}
    except Exception as e:
        print(f"질문 답변 생성 중 오류 발생: {str(e)}")
        return {"answer": "답변을 생성할 수 없습니다.", "context": context, "question": question}


# BM25 검색 관련 함수
def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    BM25 알고리즘을 사용하여 상위 k개의 관련 텍스트 청크를 검색합니다.

    Args:
        bm25: 사전 계산된 BM25 인덱스
        cleaned_texts: BM25 인덱스에 해당하는 정제된 텍스트 청크 목록
        query: 검색 쿼리 문자열
        k: 검색할 텍스트 청크 수

    Returns:
        BM25 점수 기반 상위 k개의 정제된 텍스트 청크 목록
    """
    # 쿼리 토큰화
    query_tokens = query.split()

    # 쿼리에 대한 BM25 점수 계산
    bm25_scores = bm25.get_scores(query_tokens)

    # 상위 k개 점수의 인덱스 가져오기
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # 상위 k개의 정제된 텍스트 청크 검색
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


# 백오프 및 재시도 함수
async def exponential_backoff(attempt: int) -> None:
    """
    지수 백오프 전략과 함께 지터를 구현합니다.
    
    Args:
        attempt: 현재 재시도 시도 번호
        
    지정된 시간 동안 대기한 후 작업을 재시도합니다.
    대기 시간은 (2^attempt) + 0에서 1 사이의 임의의 값으로 계산됩니다.
    """
    # 지수 백오프 및 지터를 사용한 대기 시간 계산
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"속도 제한에 도달했습니다. {wait_time:.2f}초 후 재시도...")
    
    # 계산된 대기 시간 동안 비동기적으로 대기
    await asyncio.sleep(wait_time)

async def retry_with_exponential_backoff(coroutine: Any, max_retries: int = 5) -> Any:
    """
    속도 제한 오류 발생 시 지수 백오프를 사용하여 코루틴을 재시도합니다.
    
    Args:
        coroutine: 실행할 코루틴
        max_retries: 최대 재시도 횟수
        
    Returns:
        성공 시 코루틴의 결과
        
    Raises:
        Exception: 모든 재시도가 실패한 경우 마지막으로 발생한 예외
    """
    for attempt in range(max_retries):
        try:
            # 코루틴 실행 시도
            return await coroutine
        except Exception as e:  # 일반적인 예외 처리로 변경 (RateLimitError 대신)
            # 마지막 시도에서도 실패한 경우 예외 발생
            if attempt == max_retries - 1:
                raise e
            
            # 재시도 전 지수 백오프 기간 동안 대기
            await exponential_backoff(attempt)
    
    # 최대 재시도 횟수에 도달했으나 성공하지 못한 경우 예외 발생
    raise Exception("최대 재시도 횟수에 도달했습니다.")
