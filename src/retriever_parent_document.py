# Chroma 사용시에 sqlite3의 버전 문제가 발생하므로 삽입된 코드
# pip install pypdf
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from model import get_sr_llm_model
from embedding import CustomEmbedding

import warnings

warnings.filterwarnings("ignore")

# 단계 1: 문서 로드(Load Documents)

loaders = [
    #첫번째 문서를 로드합니다.
    TextLoader("./data/appendix-keywords.txt"),
    #두번째 문서를 로드합니다.
    TextLoader("./data/reference.txt"),
]

docs = [] #빈 문서 리스트를 초기화 합니다.

for loader in loaders:
    docs.extend(loader.load()) #각 로더에서 문서를 로드하여 docs 리스트에 추가합니다.


# 단계 2: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# ParentDocument Retriever를 생성하고 문서를 추가합니다. 

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=CustomEmbedding()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

# 등록된 문서들은 다음과 같습니다.
print(f"문서 리스트:\n {list(store.yield_keys())}")
print("===" * 20)


# 벡터저장소에 저장된 작은 Chunk 들의 숫자
sub_docs = vectorstore.similarity_search("Word2Vec")

print(f"Sub Docs 숫자: {len(sub_docs)}")
print(f"Sub Docs 1번째\n {sub_docs[0].metadata}")
print("===" * 20)

# 전체 리트리버에서 작은 청크를 기준으로 검색해서, 원래 문서를 반환
relevant_doc = retriever.get_relevant_documents("Word2Vec")

print(f"검색결과: {relevant_doc[0].page_content}")