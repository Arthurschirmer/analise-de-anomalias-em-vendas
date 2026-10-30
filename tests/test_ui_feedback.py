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
