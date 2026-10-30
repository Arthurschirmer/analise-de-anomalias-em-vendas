import re
import unicodedata


MONTHS_PT = {
    "janeiro": "01",
    "fevereiro": "02",
    "marco": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
}


def extract_periods_from_text(text, data):
    text_lower = _normalize(text)
    available_years = sorted(data["data"].dt.year.unique())
    default_year = int(available_years[-1])

    mentions = _find_month_mentions(text_lower, default_year)
    if not mentions:
        return []

    if _looks_like_period_comparison(text_lower) and len(mentions) >= 2:
        current = mentions[0]
        previous = mentions[1]
        previous = _adjust_previous_period_year(previous, current)
        return [previous["periodo"], current["periodo"]]

    return [mention["periodo"] for mention in sorted(mentions, key=_period_sort_key)]


def _find_month_mentions(text, default_year):
    mentions = []
    for match in re.finditer(r"\b(\d{4})-(0[1-9]|1[0-2])\b", text):
        mentions.append(
            {
                "posicao": match.start(),
                "periodo": f"{match.group(1)}-{match.group(2)}",
                "ano_explicito": True,
            }
        )
    for month_name, month_number in MONTHS_PT.items():
        pattern = rf"\b{month_name}\b(?:\s+de\s+(\d{{4}}))?"
        for match in re.finditer(pattern, text):
            year_text = match.group(1)
            year = int(year_text) if year_text else default_year
            mentions.append(
                {
                    "posicao": match.start(),
                    "periodo": f"{year}-{month_number}",
                    "ano_explicito": year_text is not None,
                }
            )

    return sorted(mentions, key=lambda item: item["posicao"])


def _looks_like_period_comparison(text):
    comparison_terms = [
        "comparacao com",
        "comparado com",
        "comparada com",
        "em relacao a",
        "em relacao ao",
    ]
    return any(term in text for term in comparison_terms)


def _adjust_previous_period_year(previous, current):
    previous_year, previous_month = _split_period(previous["periodo"])
    current_year, current_month = _split_period(current["periodo"])

    if (
        previous_year == current_year
        and previous_month > current_month
        and not previous["ano_explicito"]
    ):
        previous = dict(previous)
        previous["periodo"] = f"{previous_year - 1}-{previous_month:02d}"

    return previous


def _period_sort_key(mention):
    year, month = _split_period(mention["periodo"])
    return year, month


def _split_period(period):
    year, month = period.split("-")
    return int(year), int(month)


def _normalize(value):
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
