# Chroma 사용시에 sqlite3의 버전 문제가 발생하므로 삽입된 코드
# pip install pypdf
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

from model import get_sr_llm_model
from embedding import CustomEmbedding
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

# 모델(LLM) 을 생성합니다.
llm = get_sr_llm_model()


# 단계2 : 문서를 요약해주는 Chain을 생성합니다. 

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 2})


# 단계3 : 요약된 결과를 작은 자식 청크로 나눠 벡터스토어에 저장합니다. 
vectorstore = Chroma(collection_name="summaries", embedding_function=CustomEmbedding())

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


# # We can also add the original chunks to the vectorstore if we so want
# for i, doc in enumerate(docs):
#     doc.metadata[id_key] = doc_ids[i]
# retriever.vectorstore.add_documents(docs)

sub_docs = vectorstore.similarity_search("Word2Vec")


# 요약결과를 중심으로 작은 청크를 검색
print(f"검색결과-subdocs: \n")
print(sub_docs[0])

print("===" * 20)

retrieved_docs = retriever.get_relevant_documents("Word2Vec")

print(f"검색된 상위 문서의 크기:")
print(len(retrieved_docs[0].page_content))

