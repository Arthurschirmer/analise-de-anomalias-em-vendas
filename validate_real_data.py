"""Run the local, reproducible validation of the real Olist dataset."""

import argparse

from src.evaluation.real_data_validation import run_real_data_validation


def parse_args():
    parser = argparse.ArgumentParser(description="Validação offline da base real Olist, sem uso de API.")
    parser.add_argument("--input-dir", default="data/olist_raw", help="Pasta com os CSVs originais da Olist.")
    parser.add_argument("--output-dir", default="outputs/real_data/olist", help="Pasta dos artefatos de validação.")
    parser.add_argument("--start-period", default=None, help="Primeiro mês do recorte, no formato AAAA-MM.")
    parser.add_argument("--end-period", default=None, help="Último mês do recorte, no formato AAAA-MM.")
    parser.add_argument("--threshold-pct", default=30.0, type=float, help="Limiar da variação percentual.")
    parser.add_argument("--min-period-value", default=1000.0, type=float, help="Faturamento mínimo em ambos os meses.")
    parser.add_argument("--max-investigations", default=5, type=int, help="Máximo de sinais percentuais investigados.")
    parser.add_argument("--selection-note", default=None, help="Justificativa metodológica do recorte, registrada no relatório.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_real_data_validation(
        input_dir=args.input_dir, output_dir=args.output_dir,
        start_period=args.start_period, end_period=args.end_period,
        threshold_pct=args.threshold_pct, min_period_value=args.min_period_value,
        max_investigations=args.max_investigations,
        selection_note=args.selection_note,
    )
    selected = summary["escopos"]["selecionado"]
    print("Validação concluída em modo local, sem chamadas de API.")
    print(f"Recorte: {selected['inicio']} a {selected['fim']}; {selected['linhas']} linhas; {selected['quantidade']} itens.")
    print(f"Sinais percentuais: {len(summary['sinais_percentuais_selecionados'])}; investigações: {len(summary['investigacoes'])}.")
    print(f"Relatório: {summary['outputs']['report']}")
    print(f"CSV padronizado do recorte: {summary['outputs']['selected_standardized']}")
    print("No site, use Olist local para preservar as definições da fonte; o CSV isolado não guarda esses metadados.")


if __name__ == "__main__":
    main()
