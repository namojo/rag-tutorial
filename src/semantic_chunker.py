# pip install -qU langchain_experimental 
# pip install -qU  sentence_transformers

from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from embedding import CustomEmbedding

import warnings

warnings.filterwarnings("ignore")

# 파일경로
filepath = "./data/AI_Brief.pdf"

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader(filepath)

documents = loader.load()

text_splitter = SemanticChunker(CustomEmbedding())

docs = text_splitter.split_documents(documents)

print(docs[24].page_content)

