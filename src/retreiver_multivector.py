# Chroma 사용시에 sqlite3의 버전 문제가 발생하므로 삽입된 코드
# pip install pypdf
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from model import get_sr_llm_model
from embedding import CustomEmbedding

from langchain.retrievers.multi_vector import SearchType

import uuid
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


# 단계 2: 자식 chunk를 인덱싱하는데 사용할 벡터 저장
#uuid는 Universally Unique IDentifier, 네트워크 상에서 중복되지 않은 ID를 만드는 방법


vectorstore = Chroma(
    collection_name="full_documents", embedding_function=CustomEmbedding()
)

# 부모 문서의 저장소 Layer
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

# 등록된 문서들은 다음과 같습니다.
print(f"문서 리스트:\n {doc_ids}")
print("===" * 20)


# 단계3 : 자식 문서를 chunk해서 sub_docs에 넣습니다. 
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


# 단계4 : 벡터스토어만을 이용해서 작은 청크를 검색
print(f"검색결과-Vectorstore: \n")
print(retriever.vectorstore.similarity_search("Word2Vec")[0])

print("===" * 20)

retriever.search_type = SearchType.mmr

print(f"검색된 상위 문서의 크기:")
print(len(retriever.get_relevant_documents("Word2Vec")[0].page_content))

