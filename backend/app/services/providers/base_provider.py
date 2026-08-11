from abc import ABC, abstractmethod
from app.schemas.provider import ProviderGenerationRequest, ProviderGenerationResponse, ProviderStatusResponse

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        """Executes text generation for a specified model and prompt."""
        pass

    @abstractmethod
    def get_status(self) -> ProviderStatusResponse:
        """Inspects provider configuration and operational availability status."""
        pass
