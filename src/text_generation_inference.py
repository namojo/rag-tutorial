import warnings

from langchain_community.llms import HuggingFaceTextGenInference

warnings.filterwarnings("ignore")

llm = HuggingFaceTextGenInference(
    inference_server_url="http://eeve-korean-instruct-10-8b-tgi.serving.70-220-152-1.sslip.io"
)

question = """
Suggest three names for an animal that is a superhero.
Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
If Animal is Camel, What the Names you suggest?
"""

print(f"[Answer]: {llm.invoke(question)}")
