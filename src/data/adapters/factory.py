from src.data.adapters.generic_csv_adapter import GenericCSVAdapter
from src.data.adapters.olist_adapter import OlistAdapter


def load_sales_dataset(dataset_type="csv", csv_path=None, input_dir=None):
    adapter = create_sales_data_adapter(
        dataset_type=dataset_type,
        csv_path=csv_path,
        input_dir=input_dir,
    )
    return adapter.load()


def create_sales_data_adapter(dataset_type="csv", csv_path=None, input_dir=None):
    if dataset_type == "csv":
        if not csv_path:
            raise ValueError("Informe o caminho do CSV em csv_path.")
        return GenericCSVAdapter(csv_path)

    if dataset_type == "olist":
        if not input_dir:
            raise ValueError("Informe a pasta da base Olist em input_dir.")
        return OlistAdapter(input_dir)

    raise ValueError(f"Tipo de base desconhecido: {dataset_type}")
