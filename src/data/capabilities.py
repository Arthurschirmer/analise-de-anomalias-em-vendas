"""Availability of analytical dimensions, independent of placeholder columns."""


SALES_DIMENSIONS = ("produto", "categoria", "regiao", "vendedor")
_AVAILABILITY_FLAGS = {
    "produto": "product_available",
    "categoria": "category_available",
    "regiao": "region_available",
    "vendedor": "seller_available",
}


def dataset_metadata(data_or_context):
    """Read adapter metadata from a frame, evidence, report, or metadata dict."""
    attrs = getattr(data_or_context, "attrs", None)
    if isinstance(attrs, dict):
        value = attrs.get("dataset_context", {})
        return value if isinstance(value, dict) else {}
    if not isinstance(data_or_context, dict):
        return {}
    for key in ("dataset_context", "contexto_da_base"):
        if key in data_or_context:
            value = data_or_context[key]
            return value if isinstance(value, dict) else {}
    return data_or_context


def available_dimensions(data_or_context):
    """Return known dimensions; absent metadata preserves the historical default.

    An explicit empty list declares no usable dimension. Availability flags can
    further restrict a declared list, but cannot enable a dimension it excludes.
    """
    metadata = dataset_metadata(data_or_context)
    declared = metadata.get("available_dimensions")
    allowed = (
        {value for value in declared if isinstance(value, str)}
        if isinstance(declared, (list, tuple)) else set(SALES_DIMENSIONS)
    )
    return tuple(
        dimension for dimension in SALES_DIMENSIONS
        if dimension in allowed and metadata.get(_AVAILABILITY_FLAGS[dimension]) is not False
    )
