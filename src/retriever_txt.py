# pip install -qU chromadb pysqlite3-binary

# Chroma 사용시에 sqlite3의 버전 문제가 발생하므로 삽입된 코드
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.vectorstores.faiss import FAISS
from embedding import CustomEmbedding


# 텍스트를 600자 단위로 분할
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=0)

# TextLoader 를 통해 텍스트 파일을 로드
split_docs = TextLoader("./data/appendix-keywords.txt").load_and_split(text_splitter)

# Chroma 를 통해 벡터 저장소 생성
chroma_db = Chroma.from_documents(documents=split_docs, embedding=CustomEmbedding())
# faiss_db = FAISS.from_documents(documents=split_docs, embedding=CustomEmbedding())

# 유사도 검색(쿼리)
# similar_docs = faiss_db.similarity_search("TF IDF 에 대하여 알려줘")
similar_docs = chroma_db.similarity_search("TF IDF 에 대하여 알려줘")
print(similar_docs[0].page_content)

print("===" * 20)
print (" embedding_Vector 를 통해 유사도 검색")

# embedding_Vector 를 통해 유사도 검색
embedding_vector = CustomEmbedding().embed_query("TF IDF 에 대하여 알려줘")
#similar_docs = faiss_db.similarity_search_by_vector(embedding_vector)
similar_docs = chroma_db.similarity_search_by_vector(embedding_vector)

print(similar_docs[0].page_content)

# 동일한 쿼리를 임베딩 한 다음, 임베딩으로 검색을 합니다.
# 그러므로 당연스럽게 동일한 결과가 나옵니다.

print("===" * 20)
print ("유사도 검색(쿼리)")

# 유사도 검색(쿼리)
# similar_docs = faiss_db.similarity_search("Word2Vec 에 대하여 알려줘")
similar_docs = chroma_db.similarity_search("Word2Vec 에 대하여 알려줘")

print(similar_docs[0].page_content)


print("===" * 20)
print ("VectorStore Retriever")

# retriever 생성
retriever = chroma_db.as_retriever(search_type="mmr", search_kwargs={"k": 2})
#retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# mmr_search 를 통해 유사도 높은 2개 문서를 검색
relevant_docs = retriever.get_relevant_documents("VectorStore, TF IDF 에 대하여 알려줘")

print(f"문서의 개수: {len(relevant_docs)}")
print("[검색 결과]\n")

print("결과1")
print(relevant_docs[0].page_content)

print("결과2")
print(relevant_docs[1].page_content)
