import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from src.llm.provider import LLMProvider


class HTTPChatLLMProvider(LLMProvider):
    CONCLUSION_MAX_COMPLETION_TOKENS = 1536
    PLAN_MAX_COMPLETION_TOKENS = 1024

    def __init__(self, api_base_url, api_key, model):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.last_usage = None

    def generate(self, system_prompt, user_prompt, context):
        self.last_usage = None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\nEvidencias em JSON:\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}",
                },
            ],
            **self._completion_options(self.CONCLUSION_MAX_COMPLETION_TOKENS),
        }

        response_data = self._post_chat_completion(payload)
        self.last_usage = response_data.get("usage")

        return self._response_content(response_data)

    def interpret_query(self, question, dataset_context):
        system_prompt = (
            "Interprete perguntas sobre a base de vendas e responda somente com um objeto JSON. "
            "Nao calcule valores e nao invente nomes, periodos ou filtros. "
            "Use operation: summary, ranking, compare, trend, clarify ou unsupported. "
            "Use metric: faturamento, quantidade, preco_unitario ou desconto. "
            "Use dimension: produto, categoria, regiao, vendedor, periodo ou null. "
            "Os filtros devem ser um objeto com dimensoes e valores mencionados pelo usuario. "
            "Os periodos devem estar no formato YYYY-MM. Se faltar informacao essencial, use clarify e explique em message. "
            "Formato: {operation, metric, dimension, periods, filters, limit, message}."
        )
        user_content = json.dumps(
            {"pergunta": question, "contexto_da_base": dataset_context},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **self._completion_options(self.PLAN_MAX_COMPLETION_TOKENS),
        }
        response_data = self._post_chat_completion(payload)

        content = self._response_content(response_data)
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                plan = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
        return plan if isinstance(plan, dict) else None

    def _completion_options(self, completion_tokens):
        # Keep optional provider-specific fields out of other compatible APIs.
        if urlsplit(self.api_base_url).hostname != "api.groq.com":
            return {}
        options = {"max_completion_tokens": completion_tokens}
        if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            # This task explains precomputed evidence; reserve room for the final
            # text without requesting the model's more expensive reasoning modes.
            options["reasoning_effort"] = "low"
        return options

    @staticmethod
    def _response_content(response_data):
        choices = response_data.get("choices") or []
        choice = choices[0] if choices else {}
        content = (choice.get("message") or {}).get("content")
        if choice.get("finish_reason") == "length":
            raise RuntimeError("A IA devolveu uma resposta incompleta. A explicação não foi utilizada.")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("A API de IA devolveu uma resposta sem texto.")
        return content.strip()

    def _post_chat_completion(self, payload):
        request = urllib.request.Request(
            self.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "AgenteAnomaliasTCC/1.0 (Python urllib)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            try:
                error_data = json.loads(details)
                api_error = error_data.get("error", error_data)
                if isinstance(api_error, dict):
                    message = api_error.get("message") or str(api_error)
                    code = api_error.get("code")
                    details = f"{message} (código: {code})" if code else message
                else:
                    details = str(api_error)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(
                f"A API respondeu HTTP {error.code}: {details[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Não foi possível conectar à API: {error.reason}"
            ) from error
