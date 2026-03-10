import os
from typing import Literal, Optional, List

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.callbacks import BaseCallbackHandler

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        api_key = self.config.azure_kwargs.api_key or os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY")
        azure_deployment = self.config.azure_kwargs.azure_deployment or os.getenv("EMBEDDING_AZURE_DEPLOYMENT")
        azure_endpoint = self.config.azure_kwargs.azure_endpoint or os.getenv("EMBEDDING_AZURE_ENDPOINT")
        api_version = self.config.azure_kwargs.api_version or os.getenv("EMBEDDING_AZURE_API_VERSION")
        default_headers = self.config.azure_kwargs.default_headers

        # If the API key is not provided or is a placeholder, use DefaultAzureCredential.
        if api_key is None or api_key == "" or api_key == "your-api-key":
            self.credential = DefaultAzureCredential()
            azure_ad_token_provider = get_bearer_token_provider(
                self.credential,
                SCOPE,
            )
            api_key = None
        else:
            azure_ad_token_provider = None

        self.client = AzureOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=azure_ad_token_provider,
            api_version=api_version,
            api_key=api_key,
            http_client=self.config.http_client,
            default_headers=default_headers,
        )

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None, callbacks: Optional[List[BaseCallbackHandler]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
            callbacks (list[BaseCallbackHandler], optional): List of callback handlers. Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.config.model)

        if response.usage:
            token_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": 0,
                "total_tokens": response.usage.total_tokens,
            }
            # print(f"Token usage: {token_usage}\t Model: {response.model}")
            ai_message = AIMessage(
                content="",
                usage_metadata=token_usage,
                response_metadata={
                    "model_name": response.model,
                },
            )

            if callbacks:
                for cb in callbacks:
                    if isinstance(cb, BaseCallbackHandler) and hasattr(cb, "on_llm_end"):
                        cb.on_llm_end(
                            LLMResult(
                                generations=[[ChatGeneration(message=ai_message)]],
                                llm_output={
                                    "token_usage": token_usage,
                                    "model_name": response.model,
                                }
                            )
                        )

        return response.data[0].embedding
