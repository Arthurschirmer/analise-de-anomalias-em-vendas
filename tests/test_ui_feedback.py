import json
import unittest

from src.ui.feedback import classify_error, sanitize_details


class FeedbackTest(unittest.TestCase):
    def test_cloudflare_error_is_not_model_permission(self):
        feedback = classify_error(RuntimeError("A API respondeu HTTP 403: error code: 1010"))
        self.assertEqual(feedback["code"], "cloudflare_1010")
        self.assertIn("conexão", feedback["title"])
        self.assertNotIn("modelo escolhido", feedback["action"])

    def test_missing_key_is_configuration_instead_of_authentication(self):
        feedback = classify_error(ValueError("LLM_PROVIDER=groq, mas GROQ_API_KEY está vazia."))
        self.assertEqual(feedback["code"], "missing_api_key")
        self.assertEqual(feedback["kind"], "info")

    def test_no_anomaly_is_not_a_failure(self):
        feedback = classify_error(ValueError("Nenhuma anomalia foi detectada com os parâmetros atuais."))
        self.assertEqual(feedback["code"], "no_anomaly")
        self.assertEqual(feedback["kind"], "info")

    def test_api_failures_have_distinct_actions(self):
        cases = [("HTTP Error 401: Unauthorized", "authentication"),
                 ("HTTP 403 model_permission_blocked_project", "permissions"),
                 ("HTTP Error 429: Too Many Requests", "rate_limit"),
                 ("urlopen error: timed out", "network")]
        for message, code in cases:
            with self.subTest(code=code):
                self.assertEqual(classify_error(message)["code"], code)

    def test_oversized_request_is_not_a_temporary_quota_failure(self):
        message = (
            "A API respondeu HTTP 413: Request too large for model `example-model` "
            "in organization `org_example` service tier `on_demand` on tokens per "
            "minute (TPM): Limit 6000, Requested 28000. code: rate_limit_exceeded"
        )
        feedback = classify_error(message, context="api")
        self.assertEqual(feedback["code"], "request_too_large")
        self.assertEqual(feedback["kind"], "warning")
        self.assertIn("grande demais", feedback["title"])
        self.assertIn("análise local", feedback["action"])
        self.assertNotIn("Aguarde", feedback["action"])
        self.assertIn("HTTP 413", feedback["technical_details"])

    def test_context_window_and_payload_size_errors_have_specific_feedback(self):
        for message in (
            "HTTP 400: context_length_exceeded",
            "This model's maximum context length is 8192 tokens.",
            "Payload too large",
            "Request Entity Too Large",
            "status_code: 413",
            "The input is too large: input too long",
        ):
            with self.subTest(message=message):
                self.assertEqual(classify_error(message)["code"], "request_too_large")

    def test_size_error_takes_precedence_over_status_like_token_counts(self):
        feedback = classify_error("HTTP 413: Request too large. TPM Limit 1000, Requested 1010.")
        self.assertEqual(feedback["code"], "request_too_large")

    def test_real_temporary_tpm_limit_remains_a_rate_limit(self):
        feedback = classify_error(
            "HTTP 429: Rate limit reached on tokens per minute (TPM). "
            "Limit 6000, Used 5000, Requested 2000. Please try again in 15s."
        )
        self.assertEqual(feedback["code"], "rate_limit")
        self.assertIn("Aguarde", feedback["action"])

    def test_oversized_request_does_not_expose_credentials(self):
        secret = "gsk_FICTITIOUSFEEDBACKTEST1234567890"
        feedback = classify_error(f"HTTP 413: Request too large; Authorization: Bearer {secret}")
        serialized = json.dumps(feedback, ensure_ascii=False)
        self.assertEqual(feedback["code"], "request_too_large")
        self.assertNotIn(secret, serialized)
        self.assertIn("chave oculta", feedback["technical_details"])

    def test_an_ordinary_numeric_value_is_not_an_http_size_error(self):
        feedback = classify_error(ValueError("O valor informado na coluna faturamento foi 413."))
        self.assertEqual(feedback["code"], "unknown")

    def test_missing_months_is_not_a_no_anomaly_result(self):
        feedback = classify_error("Precisamos de dois periodos consecutivos para detectar uma variação mensal.")
        self.assertEqual(feedback["code"], "missing_consecutive_periods")
        self.assertEqual(feedback["kind"], "info")

    def test_empty_and_invalid_data_have_actionable_feedback(self):
        self.assertEqual(classify_error("A base não contém registros de vendas.", context="data")["code"], "empty_dataset")
        self.assertEqual(classify_error("O CSV contém datas ausentes ou inválidas.", context="data")["code"], "invalid_csv")
        self.assertEqual(classify_error("O CSV contém valores numéricos ausentes ou inválidos.", context="data")["code"], "invalid_csv")

    def test_validation_errors_and_period_guidance(self):
        self.assertEqual(classify_error("O CSV nao possui as colunas obrigatorias: data")["code"], "invalid_csv")
        self.assertEqual(classify_error("Não foi possível identificar dois períodos")["code"], "insufficient_periods")
        self.assertEqual(classify_error("Unable to parse string at position 0")["code"], "invalid_csv")

    def test_credentials_do_not_survive_any_feedback_field(self):
        secrets = ["gsk_abcdefgh123456789", "grok_abcdefgh_12345678", "xai-abcdefghijklmnop", "sk-abcdefghijklmnop"]
        message = "HTTP 401 " + " ".join(secrets) + " Authorization: Bearer anotherprivatevalue"
        serialized = json.dumps(classify_error(message), ensure_ascii=False)
        for secret in secrets + ["anotherprivatevalue"]:
            self.assertNotIn(secret, serialized)
        self.assertIn("HTTP 401", serialized)

    def test_api_key_assignments_are_redacted(self):
        for message in ["GROQ_API_KEY=privatevalue", 'api_key: "privatevalue"', "XAI_API_KEY='privatevalue'"]:
            with self.subTest(message=message):
                self.assertNotIn("privatevalue", sanitize_details(message))


if __name__ == "__main__":
    unittest.main()
