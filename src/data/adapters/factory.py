from src.data.adapters.generic_csv_adapter import GenericCSVAdapter
from src.data.adapters.olist_adapter import OlistAdapter
from src.data.adapters.online_retail_adapter import OnlineRetailAdapter


def load_sales_dataset(dataset_type="csv", csv_path=None, input_dir=None, input_path=None):
    adapter = create_sales_data_adapter(
        dataset_type=dataset_type,
        csv_path=csv_path,
        input_dir=input_dir,
        input_path=input_path,
    )
    return adapter.load()


def create_sales_data_adapter(dataset_type="csv", csv_path=None, input_dir=None, input_path=None):
    if dataset_type == "csv":
        if not csv_path:
            raise ValueError("Informe o caminho do CSV em csv_path.")
        return GenericCSVAdapter(csv_path)

    if dataset_type == "olist":
        if not input_dir:
            raise ValueError("Informe a pasta da base Olist em input_dir.")
        return OlistAdapter(input_dir)

    if dataset_type == "online_retail":
        if not input_path:
            raise ValueError("Informe a planilha ou o ZIP Online Retail em input_path.")
        return OnlineRetailAdapter(input_path)

    raise ValueError(f"Tipo de base desconhecido: {dataset_type}")
