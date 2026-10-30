import os

from src.llm.http_provider import HTTPChatLLMProvider
from src.llm.mock_provider import MockLLMProvider
from src.llm.xai_provider import XAIResponsesLLMProvider


def create_llm_provider_from_env():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=groq, mas GROQ_API_KEY está vazia. "
                "Adicione sua chave ao arquivo .env."
            )
        return HTTPChatLLMProvider(
            api_base_url=os.getenv(
                "GROQ_API_BASE_URL",
                "https://api.groq.com/openai/v1/chat/completions",
            ).strip(),
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        )

    if provider == "xai":
        api_key = os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=xai, mas XAI_API_KEY está vazia. "
                "Adicione sua chave ao arquivo .env."
            )
        return XAIResponsesLLMProvider(
            api_key=api_key,
            model=os.getenv("XAI_MODEL", "grok-4.7").strip(),
        )

    if provider == "http":
        api_base_url = os.getenv("LLM_API_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")

        if api_base_url and api_key and model:
            return HTTPChatLLMProvider(
                api_base_url=api_base_url,
                api_key=api_key,
                model=model,
            )

    return MockLLMProvider()
