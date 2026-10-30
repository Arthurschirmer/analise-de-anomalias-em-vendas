# Agente para investigação de anomalias em dados de vendas

Protótipo de TCC em Ciência da Computação:

**Desenvolvimento e avaliação de um agente para investigação de anomalias em dados de vendas** (título provisório).

Autor: Arthur Thomé da Cruz Schirmer. Orientador: Paulo Sérgio Sausen.

## Objetivo

O sistema recebe dados de vendas em formato CSV, recebe uma descricao de anomalia e utiliza um agente para investigar evidencias nos dados. A primeira versao busca explicar variacoes como queda de faturamento entre periodos, observando fatores como produto, categoria, regiao, vendedor, quantidade, preco, desconto e faturamento.

Nesta versao, o agente usa ferramentas deterministicas para calcular os numeros e um provedor de LLM abstrato para gerar a conclusao. Por padrao, o projeto roda em modo `mock`, sem precisar de API externa.

Os arquivos CSV de demonstração estão incluídos no projeto. A configuração de
versionamento mantém fora do repositório as credenciais locais, o ambiente virtual,
os dados brutos da Olist e os resultados gerados. Os fontes e as entregas do
relatório do TCC também permanecem locais nesta configuração.

## Estrutura

```text
src/
  agent/      logica do agente de investigacao
  analysis/   calculos e analises sobre vendas
  data/       leitura, validacao e adaptadores de bases
    adapters/ conversao de bases diferentes para o formato padrao
  llm/        abstracao para provedores de LLM
  tools/      ferramentas disponiveis para o agente
  evaluation/ registros estruturados da investigacao
tests/        testes automatizados
data/         arquivos CSV de demonstracao
outputs/      logs e resultados gerados
main.py       ponto de entrada do projeto
app.py        interface web local com Streamlit
src/ui/       painel, gráficos e mensagens da interface
run_experiments.py executor de cenarios experimentais
```

## Instalar

Use Python 3.10 ou superior. O ambiente local foi verificado com Python 3.13 e
Streamlit 1.64. A demonstração funciona sem chave de API.

No PowerShell, execute a partir da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Esses comandos utilizam o ambiente virtual diretamente e dispensam a execução de
`Activate.ps1`. Em outros ambientes, após ativar o ambiente virtual, use
`python -m pip install -r requirements.txt` e `python -m streamlit run app.py`.

## Configurar API de LLM

O projeto funciona sem API no modo `mock`:

```text
LLM_PROVIDER=mock
```

Para conectar à Groq, crie uma chave em [console.groq.com/keys](https://console.groq.com/keys),
copie `.env.example` para `.env` e configure:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=sua-chave-da-groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_BASE_URL=https://api.groq.com/openai/v1/chat/completions
```

O projeto usa o formato Chat Completions compatível com a API Groq, sem dependência
adicional. A faixa gratuita tem limites de uso por modelo, que podem mudar. Guarde a
chave somente no `.env`, que é ignorado pelo Git; não a coloque no código nem a envie
em mensagens. A pergunta e o contexto analítico são enviados à API para gerar respostas,
portanto use dados anonimizados se a base contiver informações sensíveis.

Também é possível configurar outro provedor compatível com Chat Completions usando
`LLM_PROVIDER=http`, `LLM_API_BASE_URL`, `LLM_API_KEY` e `LLM_MODEL`.

## Executar

Para abrir a interface web local:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Na interface, o usuario pode carregar um CSV, usar a base demo, carregar a base
Olist local, conversar com o agente, executar deteccao automatica, visualizar
graficos e baixar o log da investigacao.

A interface abre com um cenário fictício e organiza a análise em seis áreas:

- **Painel:** indicadores, evolução do faturamento, variações mensais e contribuição por segmento.
- **Histórico:** referência diária ou semanal, tendência local, limites de decisão, alertas no teste e revisão humana persistida.
- **Investigação:** detecção automática ou comparação de dois meses, com evidências e gráficos antes/depois.
- **Assistente:** perguntas sobre a anomalia, com exemplos baseados nos meses da base carregada.
- **Avaliação:** experimento controlado, Olist com injeções conhecidas e comparação entre explicações locais e da IA.
- **Dados e relatório:** prévia do CSV e exportação da base, do relatório legível e do registro técnico.

Os quatro cenários fictícios podem ser trocados pela barra lateral. O menu do app
permite alternar entre tema claro e escuro. A assistência de IA pode ser desligada;
se o provedor falhar, a investigação continua com cálculos locais e mostra um aviso
com uma ação sugerida. O erro original fica nos detalhes técnicos, com chaves ocultas.

Os sinais do painel usam um limiar percentual entre meses consecutivos observados.
Meses ausentes não são preenchidos com zero. Um sinal pede investigação e não
demonstra significância estatística ou causalidade. As contribuições de dimensões
distintas não devem ser somadas porque compartilham os mesmos registros de vendas.

### Perguntas no chat

Depois de carregar uma base, a área `Assistente` aceita perguntas analiticas em linguagem
natural. Exemplos:

```text
Quais produtos tiveram maior faturamento em agosto?
Compare a quantidade vendida em julho e agosto.
Quanto faturou o produto Notebook na regiao Sul?
Mostre um grafico de faturamento por mes.
Por que o faturamento caiu em agosto em comparacao com julho?
```

O chat calcula faturamento, quantidade, preco unitario medio e desconto medio.
Tambem pode agrupar e filtrar por produto, categoria, regiao, vendedor ou periodo.
Os resultados aparecem na conversa como texto, tabela e, quando aplicavel, grafico.
Perguntas sobre causa de uma variacao usam o fluxo de investigacao existente.

Por padrao, o sistema interpreta perguntas comuns com regras locais e executa os
calculos no codigo. Com `LLM_PROVIDER=http`, o provedor configurado tambem pode
interpretar perguntas e gerar um plano estruturado; o plano e validado antes da
execucao das ferramentas. Nesse modo, a pergunta e o contexto recente da conversa
sao enviados ao provedor, enquanto as linhas da base permanecem locais. Metricas
que nao existem no formato padrao, como lucro e custo, nao sao calculadas.

```bash
python main.py
```

Por padrao, o sistema usa um CSV ja padronizado. O formato interno esperado pelo
agente possui as colunas:

```text
data, produto, categoria, regiao, vendedor, quantidade, preco_unitario, desconto, faturamento
```

Ou informando uma anomalia:

```bash
python main.py --anomalia "O faturamento caiu significativamente em agosto em comparacao com julho."
```

Tambem e possivel informar explicitamente uma base CSV no formato padrao:

```bash
python main.py --dataset-type csv --csv data/vendas_demo.csv
```

Para usar a base real da Olist, obtenha os arquivos pela fonte indicada na seção
de dados reais e coloque estes quatro arquivos em `data/olist_raw/`:

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_products_dataset.csv
olist_customers_dataset.csv
```

O arquivo `product_category_name_translation.csv` é opcional. Os arquivos brutos
não são distribuídos neste repositório. Depois de colocá-los na pasta, execute:

```bash
python main.py --dataset-type olist --input-dir data/olist_raw --auto-detect
```

Se quiser salvar a versao convertida da Olist no formato padrao do projeto:

```bash
python main.py --dataset-type olist --input-dir data/olist_raw --export-standardized data/olist_padronizado.csv
```

O adaptador da Olist converte os dados assim:

```text
order_purchase_timestamp -> data
product_id               -> produto
product_category_name    -> categoria
customer_state           -> regiao
seller_id                -> vendedor
price                    -> preco_unitario e faturamento
```

Como a Olist nao possui uma coluna direta de desconto, o campo `desconto` e
preenchido com `0.0`. Esse preenchimento não comprova que não houve descontos.
Cada item de pedido e tratado como uma unidade vendida. As linhas padronizadas
agregam itens por dia/produto/categoria/UF/vendedor; não representam pedidos.
O faturamento é a soma dos preços dos itens de pedidos entregues, pela data da
compra, sem frete. A dimensão `regiao` contém UFs dos clientes. Os identificadores
de produto e vendedor não contêm nomes comerciais. Essas definições acompanham
as evidências e os relatórios para limitar interpretações indevidas.

### Validar com os dados reais da Olist

Com os CSVs em `data/olist_raw`, a validação reproduzível roda localmente, sem API:

```powershell
& ".\.venv\Scripts\python.exe" validate_real_data.py --start-period 2017-01 --end-period 2018-07
```

Esse é um recorte exploratório contínuo que separa os meses esparsos de 2016 e a
borda final de agosto/2018. Ele não certifica completude de cada mês. Para usar
todo o histórico, omita os dois argumentos de período. Os arquivos brutos não
são modificados, e o relatório também apresenta a comparação com o histórico integral.

Os artefatos ficam em `outputs/real_data/olist/`: relatório legível, auditoria de
qualidade e reconciliação, hashes dos arquivos de entrada, série mensal, resultados
dos cinco detectores, sensibilidade ao limiar, investigações locais e CSV selecionado.
O limiar padrão é 30%, com piso de R$ 1.000 nos dois meses; ambos são parâmetros
exploratórios documentados, não limites calibrados por rótulos reais.

No site, selecione **Olist local**, use opcionalmente **Recortar por mês** e clique
em **Carregar Olist**. O painel mostra cobertura e limitações da fonte. Sem rótulos
de anomalia ou causa, essa execução valida o funcionamento e as evidências do sistema;
não permite afirmar acurácia, precisão, fraude ou causalidade confirmada.

Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Tambem e possivel deixar o sistema detectar automaticamente uma anomalia entre
periodos consecutivos. O metodo padrao usa variacao percentual:

```bash
python main.py --auto-detect
```

Por padrao, a deteccao automatica procura variacoes de faturamento iguais ou
superiores a 30%. O limite pode ser alterado:

```bash
python main.py --auto-detect --threshold-pct 20
```

Tambem existem outros metodos simples de deteccao:

```bash
python main.py --auto-detect --detection-method zscore
python main.py --auto-detect --detection-method iqr
python main.py --auto-detect --detection-method media_movel
python main.py --auto-detect --detection-method combinado
```

O metodo `combinado` marca uma anomalia quando pelo menos um dos metodos
disponiveis identifica uma variacao relevante.

Em bases reais, pode ser util ignorar comparacoes em que um dos periodos tem
valor muito baixo:

```bash
python main.py --dataset-type olist --input-dir data/olist_raw --auto-detect --min-period-value 1000
```

Tambem e possivel detectar anomalias pela quantidade vendida:

```bash
python main.py --auto-detect --metric quantidade
```

O registro estruturado da investigacao sera salvo em:

```text
outputs/investigation_log.json
```

## Executar cenarios experimentais

Os cenarios ficam em:

```text
data/scenarios.json
```

Para executar todos:

```bash
python run_experiments.py
```

Cada cenario pode usar uma base CSV padronizada ou outro tipo de base
suportado pelos adaptadores. Por exemplo:

```json
{
  "id": "cenario_olist",
  "dataset_type": "olist",
  "input_dir": "data/olist_raw",
  "anomalia": "O faturamento aumentou em fevereiro de 2017 em comparacao com janeiro de 2017.",
  "expected_cause": {
    "dimensao": "regiao",
    "valor": "SP"
  },
  "expected_terms": ["SP"]
}
```

Cada cenario gera um log JSON em `outputs/experiments/`. Um resumo geral e salvo em:

```text
outputs/experiments/summary.csv
```

Tambem sao gerados arquivos de avaliacao:

```text
outputs/experiments/evaluation_summary.json
outputs/experiments/evaluation_report.md
```

Esses arquivos podem ser usados futuramente para avaliar o agente no TCC, pois registram a anomalia investigada, as ferramentas utilizadas, as evidencias encontradas, a conclusao final, a causa esperada, se a causa encontrada pelo ranking do agente foi correta e o tempo de execucao de cada cenario.

As metricas geradas atualmente incluem:

- acerto Top 1;
- acerto Top K;
- posicao da causa esperada no ranking;
- termos esperados encontrados;
- verificacao simples de numeros nao suportados pelas evidencias;
- tempo medio de execucao por cenario.

## Exemplo de investigacao

O arquivo `data/vendas_demo.csv` contem dados ficticios de julho e agosto. Existe uma anomalia proposital em agosto: queda de faturamento associada principalmente ao produto Notebook, com reducao de quantidade vendida.

Tambem existem cenarios adicionais:

- `data/vendas_queda_regiao_sul.csv`: queda concentrada na regiao Sul em setembro.
- `data/vendas_aumento_desconto.csv`: queda de faturamento em outubro associada ao aumento de desconto no produto Notebook.
- `data/vendas_queda_preco_monitor.csv`: queda de faturamento em novembro associada a reducao de preco do produto Monitor.

O agente:

1. recebe a descricao da anomalia;
2. identifica os periodos citados;
3. executa ferramentas de comparacao;
4. consulta vendas por produto, categoria, regiao e vendedor;
5. monta um ranking de causas provaveis;
6. gera uma conclusao baseada somente nos resultados calculados.

O log JSON da investigacao contem:

- descricao da anomalia;
- ferramentas utilizadas;
- evidencias retornadas pelas ferramentas;
- relatorio estruturado;
- ranking de causas provaveis;
- avaliacao automatica do cenario, quando executado por `run_experiments.py`;
- conclusao final.

No modo automatico, antes da investigacao o sistema:

1. agrega as vendas por periodo;
2. compara cada periodo com o periodo anterior;
3. calcula os indicadores do metodo de deteccao escolhido;
4. seleciona a anomalia mais relevante que ultrapassa o limite configurado;
5. transforma essa variacao em uma descricao de anomalia para o agente investigar.

## Histórico e avaliação acadêmica

As telas **Histórico** e **Avaliação** incluem referência por dia da semana,
tendência local, separação cronológica e revisão humana. As investigações mostram
os efeitos de volume, mix e valor unitário, além de indicadores operacionais da Olist.
No Histórico, selecione **Base inteira**, **Por categoria** ou **Por região / UF**.
A calibração padrão usa mediana e MAD; a regra anterior de quantil continua disponível.
Em **Avaliação**, a suíte ampliada compara mudanças pequenas, graduais e persistentes
em várias sementes. A opção **Revisão humana** aceita uma amostra com e sem alertas,
preparada no Histórico, e mantém as métricas pendentes até os pareceres conclusivos.
O protocolo, fórmulas, limitações e comandos estão em [docs/avaliacao.md](docs/avaliacao.md).

```powershell
& ".\.venv\Scripts\python.exe" run_benchmark.py --olist-dir data/olist_raw
& ".\.venv\Scripts\python.exe" run_benchmark.py --suite --seeds 42 137 2026 --olist-dir data/olist_raw --output-dir outputs/benchmark/suite
& ".\.venv\Scripts\python.exe" compare_reports.py
```

A avaliação dos detectores é offline. A comparação de explicações só consulta a
API com `--with-ai` ou quando essa opção é marcada no site. Notas humanas ficam
pendentes até o preenchimento; não há precisão/F1 reais sem rótulos independentes.

## Testes automatizados

```powershell
$env:LLM_PROVIDER = "mock"
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## Limitacoes da primeira versao

- Sem um provedor HTTP configurado, a interpretacao do chat atende a um conjunto de intencoes comuns baseado em regras.
- Com provedor HTTP, o LLM interpreta a pergunta, mas os calculos continuam limitados as metricas e dimensoes implementadas.
- O modo `mock` nao possui raciocinio generativo real.
- A deteccao automatica possui metodos estatisticos simples, mas ainda nao substitui tecnicas mais robustas de series temporais.
- A referência histórica considera padrão semanal e tendência local; não modela feriados ou sazonalidade anual. A calibração pode ser conservadora em dados voláteis.
- O ranking indica associacoes nos dados, mas ainda nao prova causalidade externa.
- A avaliacao automatica depende de causas esperadas cadastradas manualmente nos cenarios.
- O adaptador da Olist precisa preencher desconto como zero porque essa informacao nao existe diretamente na base.

## Proximos passos

- Avaliar a interpretacao do chat com perguntas variadas, ambiguas e fora do escopo.
- Revisar os alertas da Olist com conhecimento do negócio e construir rótulos independentes.
- Ampliar os cenarios experimentais para cobrir mais tipos de anomalia.
- Comparar os resultados do agente com uma análise manual e avaliar outras alternativas além dos detectores já comparados.
