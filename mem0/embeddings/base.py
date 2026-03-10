from abc import ABC, abstractmethod
from typing import Literal, Optional, List

from mem0.configs.embeddings.base import BaseEmbedderConfig

from langchain_core.callbacks import BaseCallbackHandler


class EmbeddingBase(ABC):
    """Initialized a base embedding class

    :param config: Embedding configuration option class, defaults to None
    :type config: Optional[BaseEmbedderConfig], optional
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            self.config = BaseEmbedderConfig()
        else:
            self.config = config

    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]], callbacks: Optional[List[BaseCallbackHandler]] = None):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
            callbacks (list[BaseCallbackHandler], optional): List of callback handlers. Defaults to None.
        Returns:
            list: The embedding vector.
        """
        pass
