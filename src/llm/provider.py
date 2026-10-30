from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_prompt, context):
        """Gera uma resposta textual com base no contexto informado."""

    def interpret_query(self, question, dataset_context):
        """Return a structured analysis plan, when the provider supports planning."""
        return None
