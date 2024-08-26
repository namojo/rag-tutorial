from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings

warnings.filterwarnings("ignore")


# 파일경로
filepath = "./data/AI_Brief.pdf"

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader(filepath)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    length_function=len,
)

# 문서 로드 및 분할 (load_and_split)
split_doc = loader.load_and_split(text_splitter=text_splitter)
print(f"RecursiveCharacterTextSplitter \t 사용시 문서의 수: {len(split_doc)}")

print(split_doc[0])  # 분할된 문서의 첫 번째 문서를 출력합니다.
print("===" * 20)
print(split_doc[1])  # 분할된 문서의 두 번째 문서를 출력합니다.
