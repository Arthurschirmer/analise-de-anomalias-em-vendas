import argparse
from pathlib import Path

from src.agent.investigation_agent import InvestigationAgent
from src.analysis.anomaly_detection import (
    build_anomaly_description,
    detect_period_anomalies,
    get_most_relevant_anomaly,
)
from src.data.adapters import load_sales_dataset
from src.llm.factory import create_llm_provider_from_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agente de IA para investigacao de anomalias em dados de vendas."
    )
    parser.add_argument(
        "--dataset-type",
        default="csv",
        choices=["csv", "olist", "online_retail"],
        help="Tipo da base de entrada.",
    )
    parser.add_argument(
        "--csv",
        default="data/vendas_demo.csv",
        help="Caminho do CSV no formato padrao do projeto.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Planilha .xlsx ou ZIP original da UCI Online Retail.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/olist_raw",
        help="Pasta com os arquivos brutos quando a base exigir adaptador.",
    )
    parser.add_argument(
        "--export-standardized",
        default=None,
        help="Opcional: salva a base convertida no formato padrao do projeto.",
    )
    parser.add_argument(
        "--anomalia",
        default="O faturamento caiu significativamente em agosto em comparacao com julho.",
        help="Descricao da anomalia que deve ser investigada.",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Detecta automaticamente uma anomalia por variacao percentual entre periodos.",
    )
    parser.add_argument(
        "--metric",
        default="faturamento",
        choices=["faturamento", "quantidade"],
        help="Metrica usada na deteccao automatica.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=30.0,
        help="Limite percentual minimo para marcar uma variacao como anomalia.",
    )
    parser.add_argument(
        "--detection-method",
        default="percentual",
        choices=["percentual", "zscore", "iqr", "media_movel", "combinado"],
        help="Metodo usado na deteccao automatica de anomalias.",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=2.0,
        help="Limite absoluto de z-score para deteccao automatica.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Multiplicador do intervalo interquartil para deteccao automatica.",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=3,
        help="Quantidade de periodos anteriores usada na media movel.",
    )
    parser.add_argument(
        "--min-period-value",
        type=float,
        default=0.0,
        help="Valor minimo da metrica em ambos os periodos para considerar anomalia.",
    )
    parser.add_argument(
        "--log",
        default="outputs/investigation_log.json",
        help="Caminho do arquivo JSON em que o registro da investigacao sera salvo.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_sales_dataset(
        dataset_type=args.dataset_type,
        csv_path=args.csv,
        input_dir=args.input_dir,
        input_path=args.input_file,
    )
    if args.export_standardized:
        standardized_path = Path(args.export_standardized)
        standardized_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(standardized_path, index=False)
        print(f"Base padronizada salva em: {standardized_path}")

    llm_provider = create_llm_provider_from_env()

    anomaly_description = args.anomalia
    automatic_detection = None
    if args.auto_detect:
        anomalies = detect_period_anomalies(
            data,
            metric=args.metric,
            threshold_pct=args.threshold_pct,
            method=args.detection_method,
            zscore_threshold=args.zscore_threshold,
            iqr_multiplier=args.iqr_multiplier,
            moving_average_window=args.moving_average_window,
            min_period_value=args.min_period_value,
        )
        automatic_detection = get_most_relevant_anomaly(anomalies)
        if automatic_detection is None:
            raise ValueError(
                "Nenhuma anomalia foi detectada com o limite percentual informado."
            )
        anomaly_description = build_anomaly_description(automatic_detection)
        print(f"Anomalia detectada automaticamente: {anomaly_description}")

    agent = InvestigationAgent(llm_provider=llm_provider)
    investigation = agent.investigate(
        anomaly_description,
        data,
        extra_evidence=automatic_detection,
    )

    output_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    investigation.save(output_path)

    print("\nConclusao da investigacao:\n")
    print(investigation.final_conclusion)
    print(f"\nRegistro salvo em: {output_path}")


if __name__ == "__main__":
    main()
