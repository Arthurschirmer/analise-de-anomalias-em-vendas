"""Turn technical failures into clear, safe messages for the interface.

This module has no Streamlit dependency so every view can present the same
message and keep the sanitized original error in an optional details panel.
"""

import re
import unicodedata


def sanitize_details(value):
    """Remove credentials from text before showing or storing diagnostics."""
    text = str(value)
    text = re.sub(
        r"(?i)\bBearer\s+[^\s,;\"'<>]+",
        "Bearer [chave oculta]",
        text,
    )
    text = re.sub(
        r"(?i)(\b(?:[A-Z][A-Z0-9_]*API_KEY|API_KEY|api_key|apikey)\s*[=:]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;<>]+)",
        r"\1[chave oculta]",
        text,
    )
    return re.sub(
        r"(?i)\b(?:gsk_|grok_|xai-|sk-)[A-Za-z0-9_-]{8,}",
        "[chave oculta]",
        text,
    )


def _normalized(value):
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def classify_error(error, context="analysis"):
    """Return a serializable UI message with a suggested next action.

    ``context`` can be ``analysis``, ``api`` or ``data`` (``load`` is also
    accepted). It only changes the generic fallback; known errors are always
    identified by their content. No credentials are returned in any field.
    """
    raw = str(error) or type(error).__name__
    text = _normalized(raw)
    technical_details = sanitize_details(raw)[:2000]

    def result(code, kind, title, message, action):
        return {
            "code": code,
            "kind": kind,
            "title": title,
            "message": message,
            "action": action,
            "technical_details": technical_details,
        }

    if re.search(r"\b1010\b", text):
        return result(
            "cloudflare_1010", "warning", "A conexão com a IA foi bloqueada",
            "A proteção de rede do serviço recusou a identificação desta conexão. "
            "Os cálculos locais continuam disponíveis.",
            "Tente novamente. Se o bloqueio continuar, consulte o suporte do provedor "
            "com o código 1010 e os detalhes abaixo.",
        )

    if (
        ("api_key" in text or "chave" in text)
        and any(term in text for term in ("vazia", "vazio", "empty", "nao configurada", "missing api", "not set"))
    ):
        return result(
            "missing_api_key", "info", "Configure a IA para conversar com o modelo",
            "A chave da API ainda não foi preenchida. Você pode explorar os gráficos "
            "e executar os cálculos locais da base.",
            "Preencha a chave no arquivo .env do projeto e reinicie o aplicativo.",
        )

    if (
        re.search(r"\b401\b", text)
        or any(term in text for term in ("incorrect api key", "invalid api key", "invalid_api_key", "chave invalida", "unauthorized"))
    ):
        return result(
            "authentication", "warning", "A chave da IA não foi aceita",
            "O provedor recusou a autenticação. Isso não altera a base carregada "
            "nem os resultados calculados localmente.",
            "Confira se a chave no .env pertence ao provedor selecionado e está ativa. "
            "Depois, reinicie o aplicativo.",
        )

    if re.search(r"\b429\b", text) or any(term in text for term in ("rate_limit", "rate limit", "too many requests", "quota exceeded", "resource_exhausted")):
        return result(
            "rate_limit", "warning", "O limite de uso da IA foi atingido",
            "O serviço recebeu mais pedidos ou tokens do que a cota permite "
            "neste momento. A análise local continua disponível.",
            "Aguarde a renovação da cota e tente novamente. Consulte os limites "
            "do seu plano no console do provedor.",
        )

    if re.search(r"\b403\b", text) or any(term in text for term in ("permissions_error", "model_permission_blocked", "forbidden")):
        return result(
            "permissions", "warning", "O acesso ao modelo não está liberado",
            "O serviço recusou a solicitação por uma restrição de acesso. "
            "Os detalhes abaixo podem indicar a organização, o projeto ou o modelo afetado.",
            "Confira no console do provedor se o modelo escolhido está permitido "
            "para o projeto da chave utilizada.",
        )

    if any(term in text for term in ("nenhuma anomalia", "no anomaly", "no anomalies")):
        return result(
            "no_anomaly", "info", "Nenhuma anomalia com estes critérios",
            "As variações encontradas não ultrapassaram os critérios escolhidos. "
            "Esse é um resultado válido da detecção.",
            "Compare os períodos no gráfico ou ajuste o limite e o método de detecção.",
        )

    if "periodos consecutivos" in text:
        return result(
            "missing_consecutive_periods", "info", "Faltam meses consecutivos na base",
            "Há uma lacuna entre os meses disponíveis. A detecção mensal não trata os meses ausentes como vendas zero.",
            "Carregue meses consecutivos ou use Comparar períodos para escolher explicitamente dois meses observados.",
        )

    if any(term in text for term in ("dois periodos", "ao menos 2 periodos", "pelo menos 2 periodos", "at least two periods", "insufficient periods")):
        return result(
            "insufficient_periods", "info", "Precisamos de dois períodos para comparar",
            "A investigação precisa de um período de referência e outro para análise.",
            "Carregue dados de pelo menos dois meses e selecione os períodos da comparação.",
        )

    if "nao contem registros" in text or "emptydataerror" in text:
        return result(
            "empty_dataset", "info", "O arquivo não contém vendas para analisar",
            "A base enviada está vazia. Precisamos de registros de venda para montar os indicadores.",
            "Escolha outro arquivo ou carregue um dos cenários de demonstração.",
        )

    if any(term in text for term in ("colunas obrigatorias", "missing columns", "required columns")):
        return result(
            "invalid_csv", "warning", "Faltam colunas na base enviada",
            "O arquivo precisa seguir o formato de vendas utilizado pelo projeto. "
            "O detalhe técnico informa quais colunas estão faltando.",
            "Confira os nomes no modelo de CSV ou use uma base de demonstração para começar.",
        )

    if isinstance(error, FileNotFoundError) or any(term in text for term in ("arquivo csv nao encontrado", "no such file", "file not found")):
        return result(
            "file_not_found", "warning", "O arquivo de dados não foi encontrado",
            "O caminho informado não aponta para um arquivo disponível.",
            "Selecione o arquivo novamente ou confira a pasta da base de dados.",
        )

    if any(term in text for term in ("connection", "conectar", "timed out", "timeout", "urlopen error", "name resolution", "getaddrinfo", "connectionrefused", "ssl:", "certificate_verify")):
        return result(
            "network", "warning", "Não conseguimos conectar à IA",
            "O serviço está indisponível ou a conexão não foi concluída. "
            "Os gráficos e os cálculos locais continuam disponíveis.",
            "Confira a conexão com a internet e tente novamente em alguns instantes.",
        )

    if any(term in text for term in ("unable to parse", "could not convert", "error tokenizing", "time data", "dateparseerror", "parsererror", "no columns to parse", "out of bounds nanosecond", "doesn't match format", "datas ausentes", "valores numericos ausentes")):
        return result(
            "invalid_csv", "warning", "Há valores que não conseguimos ler",
            "Uma data ou um número do CSV está em um formato inesperado.",
            "Revise o separador do CSV, as datas e as colunas numéricas. "
            "Os detalhes abaixo ajudam a localizar o valor.",
        )

    if context in {"data", "load"}:
        return result(
            "unknown", "error", "Não foi possível carregar esta base",
            "O arquivo não pôde ser preparado para análise.",
            "Confira o formato esperado do CSV e consulte os detalhes para identificar o problema.",
        )
    return result(
        "unknown", "error", "Não foi possível concluir esta etapa",
        "A análise encontrou um problema inesperado. Sua base permanece carregada.",
        "Tente novamente. Se o problema continuar, use os detalhes abaixo para investigar.",
    )
