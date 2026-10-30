"""Compare explanations for one investigation. Offline unless --with-ai is explicit."""
import argparse
from src.agent.investigation_agent import InvestigationAgent
from src.data.adapters import load_sales_dataset
from src.evaluation.llm_comparison import compare_investigation_reports, save_comparison
from src.llm.mock_provider import MockLLMProvider


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", choices=["csv", "olist"], default="csv")
    parser.add_argument("--csv-path", default="data/vendas_demo.csv")
    parser.add_argument("--input-dir", default="data/olist_raw")
    parser.add_argument("--previous-period", default="2024-07")
    parser.add_argument("--current-period", default="2024-08")
    parser.add_argument("--output-dir", default="outputs/model_comparison")
    parser.add_argument("--with-ai", action="store_true")
    args = parser.parse_args()
    data = load_sales_dataset(args.dataset_type, csv_path=args.csv_path, input_dir=args.input_dir)
    investigation = InvestigationAgent(MockLLMProvider()).investigate(
        f"Compare {args.previous_period} e {args.current_period}.", data,
        extra_evidence={"periodo_anterior": args.previous_period, "periodo_atual": args.current_period})
    provider = None
    if args.with_ai:
        from src.llm.factory import create_llm_provider_from_env
        provider = create_llm_provider_from_env()
    result = compare_investigation_reports(investigation, provider)
    directory = save_comparison(result, args.output_dir)
    print("Estado da IA:", result["ai"]["status"])
    print("Parecer humano:", result["human_review"]["status"])
    print("Artefatos:", directory)

