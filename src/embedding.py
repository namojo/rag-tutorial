from typing import (List)

import requests
from langchain_core.embeddings import Embeddings


class CustomEmbedding(Embeddings):
    def __init__(self):
        self.embed_url = 'http://sds-embed.serving.70-220-152-1.sslip.io/v1/models/embed:predict'

    def call_embed(self, url, texts):
        data = {
            "instances": texts
        }
        response = requests.post(url=url, json=data)
        result = response.json()
        return result['predictions']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        주어진 텍스트를 임베딩하여 벡터로 반환 합니다.
        """
        embed_list = self.call_embed(url=self.embed_url, texts=texts)
        return embed_list

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

        embed_list = self.call_embed(url=self.embed_url, texts=[text])
        return embed_list[0]
