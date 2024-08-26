from langchain_community.llms import HuggingFaceTextGenInference

#os.environ['NO_PROXY'] = 'sr-llm-65b-instruct.serving.70-220-152-1.sslip.io'
#os.environ['no_proxy'] = 'sr-llm-65b-instruct.serving.70-220-152-1.sslip.io'

     


def get_sr_llm_model():
    llm = HuggingFaceTextGenInference(
        # 아래의 inference 모델 중에 선택 적용도 가능합니다. 
        inference_server_url="http://meta-llama-3-1-70b-instruct-tgi.serving.70-220-152-1.sslip.io",
        # inference_server_url="http://eeve-korean-instruct-10-8b-tgi.serving.70-220-152-1.sslip.io",
        # inference_server_url="http://mixtral-8x7b-instruct.serving.70-220-152-1.sslip.io",          
        # inference_server_url="http://meta-llama-3-70b-instruct.70-220-152-1.sslip.io",
        max_new_tokens=2048,        
        temperature=0.1,
        streaming=False,
    )
    return llm