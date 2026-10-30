from src.data.adapters.base_adapter import SalesDataAdapter
from src.data.loader import load_sales_csv


class GenericCSVAdapter(SalesDataAdapter):
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        return load_sales_csv(self.csv_path)
