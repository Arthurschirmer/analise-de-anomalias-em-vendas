from copy import deepcopy
import importlib.util
import unittest

from src.evaluation.investigation_log import InvestigationLog
from src.evaluation.pdf_report import build_alert_pdf, build_investigation_pdf


def sample_log():
    report = {"periodos": {"periodo_anterior": "2011-07", "periodo_atual": "2011-08"},
              "resumo_comparativo": {"faturamento_anterior": 180., "faturamento_atual": 90., "variacao_faturamento": -90.,
                  "variacao_percentual_faturamento": -50., "quantidade_anterior": 18., "quantidade_atual": 9.},
              "causas_provaveis": [{"valor": "Produto <A> & especial", "dimensao": "produto", "metricas": {"variacao_faturamento": -90.}}],
              "contexto_da_base": {"currency": "GBP", "source_label": "UCI Online Retail", "discount_available": False,
                  "category_available": False, "seller_available": False, "limitations": ["Fonte sem rótulos de anomalias confirmadas."],
                  "comparison_window": {"mode": "aligned_days", "first_day": 1, "last_day": 9},
                  "comparison_window_definition": "Comparação somente dos dias 1 a 9. Dias sem registros não foram preenchidos."},
              "limitacoes": ["Associações não comprovam causalidade."]}
    return InvestigationLog("Queda nas vendas", "O faturamento caiu. Verifique os registros.", [], {}, report, created_at="2026-11-14T09:30:00")


class PdfReportTest(unittest.TestCase):
    def test_pdf_is_generated_from_saved_results_without_mutating_them(self):
        log = sample_log()
        before = deepcopy(log.to_dict())
        restored = InvestigationLog.from_dict(before)
        payload = build_investigation_pdf(restored)
        self.assertTrue(payload.startswith(b"%PDF-"))
        self.assertGreater(len(payload), 10000)
        self.assertEqual(restored.to_dict(), before)

    @unittest.skipUnless(importlib.util.find_spec("pymupdf"), "Extra opcional para inspecionar o PDF")
    def test_export_preserves_currency_window_limits_and_literal_user_text(self):
        import pymupdf
        payload = build_investigation_pdf(sample_log(), conclusion="Explicação <b>literal</b> & sem HTML.")
        with pymupdf.open(stream=payload, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
            self.assertIn("£ 180,00", text)
            self.assertIn("£ -90,00", text)
            self.assertNotIn("R$", text)
            self.assertIn("dias 1 a 9", text)
            self.assertIn("<b>literal</b>", text)
            self.assertIn("Produto <A> & especial", text)
            self.assertIn("não representam descontos observados", text)
            self.assertIn("14/11/2026 às 09:30:00", text)
            self.assertGreater(len(doc[0].get_drawings()), 2)
            self.assertTrue(all(page.get_text().strip() for page in doc))

    @unittest.skipUnless(importlib.util.find_spec("pymupdf"), "Extra opcional para inspecionar o PDF")
    def test_alert_pdf_handles_zero_reference_and_segment_scope(self):
        import pymupdf
        context = {"start_date": "2024-01-02", "end_date": "2024-01-02", "metric": "quantidade", "observed": 2.,
                   "expected": 0., "difference": 2., "percentage": None, "previous_observations": 60, "score": 4., "threshold": 3.,
                   "scope": {"dimension": "produto", "segmento": "A"}, "source": {"currency": "GBP", "source_label": "Loja"},
                   "calendar_context": "Promoção", "next_checks": ["Confira os registros."], "created_at": "2026-11-14"}
        payload = build_alert_pdf(context)
        with pymupdf.open(stream=payload, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
            self.assertIn("produto = A", text)
            self.assertIn("2,00 unidades", text)
            self.assertIn("referência nula ou negativa", text)
            self.assertNotIn("£", text)
            self.assertIn("Promoção", text)


if __name__ == "__main__":
    unittest.main()
