from abc import ABC, abstractmethod


class SalesDataAdapter(ABC):
    @abstractmethod
    def load(self):
        """Return sales data in the project's standard format."""
