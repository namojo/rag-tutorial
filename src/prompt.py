from langchain_core.prompts import PromptTemplate


def get_prompt():
    prompt_template = """
    HUMAN

    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Please answer in Korean.
    
    Question: {question} 
    
    Context: {context} 
    
    AI:
    """
    return PromptTemplate.from_template(prompt_template)
