import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.agent.investigation_agent import InvestigationAgent
from src.data.adapters import load_sales_dataset
from src.llm.http_provider import HTTPChatLLMProvider
from src.ui.workspace import _ResilientProvider


class HTTPProviderTest(unittest.TestCase):
    def provider(self, model="openai/gpt-oss-20b", url="https://api.groq.com/openai/v1/chat/completions"):
        return HTTPChatLLMProvider(url, "offline-placeholder", model)

    def response(self, text, finish_reason="stop"):
        return {
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 50},
        }

    def test_conclusion_uses_compact_evidence_and_bounded_output(self):
        provider = self.provider()
        response = self.response("Explicação baseada nos cálculos.")
        with patch.object(provider, "_post_chat_completion", return_value=response) as post:
            result = provider.generate("Instruções", "Compare os meses.", {"vendas": [1, 2], "região": "Sul"})
        payload = post.call_args.args[0]
        self.assertEqual(result, "Explicação baseada nos cálculos.")
        self.assertEqual(provider.last_usage, response["usage"])
        self.assertEqual(payload["max_completion_tokens"], 1536)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertTrue(payload["messages"][1]["content"].endswith('{"vendas":[1,2],"região":"Sul"}'))

    def test_plan_retains_json_parsing_and_has_its_own_output_budget(self):
        provider = self.provider()
        plan = {"operation": "compare", "periods": ["2024-07", "2024-08"], "filters": {}}
        with patch.object(provider, "_post_chat_completion", return_value=self.response(json.dumps(plan))) as post:
            self.assertEqual(provider.interpret_query("Compare os meses", {"periodos": plan["periods"]}), plan)
        self.assertEqual(post.call_args.args[0]["max_completion_tokens"], 1024)
        self.assertEqual(post.call_args.args[0]["reasoning_effort"], "low")

    def test_gpt_oss_reasoning_option_is_limited_to_supported_groq_models(self):
        for model, url, expected in (
            ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1/chat/completions", True),
            ("other-model", "https://api.groq.com/openai/v1/chat/completions", False),
            ("openai/gpt-oss-20b", "https://example.invalid/chat/completions", False),
        ):
            with self.subTest(model=model, url=url):
                provider = self.provider(model, url)
                with patch.object(provider, "_post_chat_completion", return_value=self.response("Concluído")) as post:
                    provider.generate("Instruções", "Compare", {})
                payload = post.call_args.args[0]
                self.assertEqual("reasoning_effort" in payload, expected)
                if url.startswith("https://example.invalid/"):
                    self.assertNotIn("max_completion_tokens", payload)

    def test_empty_or_truncated_output_is_not_presented_as_a_complete_explanation(self):
        for text, finish_reason in ((None, "stop"), ("", "stop"), ("Parcial", "length")):
            with self.subTest(text=text, finish_reason=finish_reason):
                provider = self.provider()
                with patch.object(provider, "_post_chat_completion", return_value=self.response(text, finish_reason)):
                    with self.assertRaises(RuntimeError):
                        provider.generate("Instruções", "Compare", {})

    def test_transport_preserves_user_agent_and_sends_json_once(self):
        provider = self.provider()
        raw = json.dumps(self.response("Concluído")).encode("utf-8")
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = raw
            self.assertEqual(provider.generate("Instruções", "Compare", {}), "Concluído")
        request = urlopen.call_args.args[0]
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("AgenteAnomaliasTCC", request.get_header("User-agent"))
        self.assertEqual(json.loads(request.data)["max_completion_tokens"], 1536)

    def test_413_keeps_complete_local_evidence_and_specific_feedback(self):
        provider = self.provider()
        resilient = _ResilientProvider(provider, remote=True)
        data = load_sales_dataset("csv", csv_path="data/vendas_demo.csv")
        error_body = json.dumps({"error": {"message": "Request too large on tokens per minute (TPM)", "code": "rate_limit_exceeded"}}).encode()
        error = HTTPError(provider.api_base_url, 413, "Request too large", None, io.BytesIO(error_body))
        with patch("urllib.request.urlopen", side_effect=error) as urlopen:
            investigation = InvestigationAgent(resilient).investigate("Compare 2024-07 e 2024-08.", data)
        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(resilient.used_ai)
        self.assertEqual(resilient.feedback["code"], "request_too_large")
        self.assertEqual(investigation.evidence["period_comparison"]["periodo_atual"], "2024-08")
        self.assertTrue(investigation.evidence["revenue_decomposition"]["product_changes"])
        self.assertTrue(investigation.structured_report["causas_provaveis"])
        self.assertTrue(investigation.tool_results)
        self.assertIn("faturamento", investigation.final_conclusion.lower())


if __name__ == "__main__":
    unittest.main()
