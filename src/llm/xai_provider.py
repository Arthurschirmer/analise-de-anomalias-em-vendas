import json
import urllib.error
import urllib.request

from src.llm.provider import LLMProvider


class XAIResponsesLLMProvider(LLMProvider):
    """xAI provider using the Responses API without an extra SDK dependency."""

    api_url = "https://api.x.ai/v1/responses"

    def __init__(self, api_key, model="grok-4.7", timeout=60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt, user_prompt, context):
        evidence = json.dumps(context, ensure_ascii=False, indent=2)
        response = self._create_response(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\nEvidências em JSON:\n{evidence}",
                },
            ]
        )
        return self._extract_text(response)

    def interpret_query(self, question, dataset_context):
        system_prompt = (
            "Interprete perguntas sobre a base de vendas e produza apenas o plano JSON solicitado. "
            "Não calcule valores e não invente nomes, períodos ou filtros. "
            "Use operation summary, ranking, compare, trend, clarify ou unsupported. "
            "Use metric faturamento, quantidade, preco_unitario ou desconto. "
            "Use dimension produto, categoria, regiao, vendedor, periodo ou null. "
            "Só inclua filtros citados pelo usuário e existentes no contexto. "
            "Os períodos devem usar o formato YYYY-MM. Se faltar informação essencial, use clarify "
            "e escreva a pergunta em message."
        )
        user_content = json.dumps(
            {"pergunta": question, "contexto_da_base": dataset_context},
            ensure_ascii=False,
        )
        response = self._create_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            text_format={
                "type": "json_schema",
                "name": "sales_query_plan",
                "strict": True,
                "schema": _query_plan_schema(),
            },
        )
        content = self._extract_text(response).strip()
        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            return None
        return plan if isinstance(plan, dict) else None

    def _create_response(self, input_messages, text_format=None):
        payload = {
            "model": self.model,
            "input": input_messages,
            # The application resends its own short conversation context each turn.
            "store": False,
        }
        if text_format:
            payload["text"] = {"format": text_format}

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            try:
                error_data = json.loads(details)
                api_error = error_data.get("error", error_data)
                if isinstance(api_error, dict):
                    details = api_error.get("message") or str(api_error)
                else:
                    details = str(api_error)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(
                f"A API da xAI respondeu HTTP {error.code}: {details[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Não foi possível conectar à API da xAI: {error.reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("A API da xAI retornou uma resposta inválida.") from error

        if response_data.get("error"):
            raise RuntimeError(f"Erro na API da xAI: {response_data['error']}")
        if response_data.get("status") == "incomplete":
            reason = response_data.get("incomplete_details") or "resposta incompleta"
            raise RuntimeError(f"A API da xAI não concluiu a resposta: {reason}")
        return response_data

    @staticmethod
    def _extract_text(response_data):
        if isinstance(response_data.get("output_text"), str):
            return response_data["output_text"]

        text_parts = []
        for item in response_data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if text:
                        text_parts.append(text)

        if not text_parts:
            raise RuntimeError("A API da xAI não retornou texto para o chat.")
        return "\n".join(text_parts)


def _query_plan_schema():
    text_value = {"type": "string"}
    filter_dimensions = ("produto", "categoria", "regiao", "vendedor")
    return {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["summary", "ranking", "compare", "trend", "clarify", "unsupported"],
            },
            "metric": {
                "type": "string",
                "enum": ["faturamento", "quantidade", "preco_unitario", "desconto"],
            },
            "dimension": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": ["produto", "categoria", "regiao", "vendedor", "periodo"],
                    },
                    {"type": "null"},
                ]
            },
            "periods": {"type": "array", "items": text_value},
            "filters": {
                "type": "object",
                "properties": {
                    dimension: {"type": "array", "items": text_value}
                    for dimension in filter_dimensions
                },
                "required": list(filter_dimensions),
                "additionalProperties": False,
            },
            "limit": {
                "anyOf": [{"type": "integer"}, {"type": "null"}]
            },
            "message": {"type": "string"},
        },
        "required": ["operation", "metric", "dimension", "periods", "filters", "limit", "message"],
        "additionalProperties": False,
    }
