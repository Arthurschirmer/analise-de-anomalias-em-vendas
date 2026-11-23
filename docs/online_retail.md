# Online Retail da UCI

Fonte: [Online Retail — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).
O arquivo original baixado pelo autor permanece preservado. Uma cópia do ZIP
fica em `data/online_retail_raw/online_retail.zip`, fora do versionamento.

## Abrir no site

1. Escolha **UCI Online Retail** em **Origem dos dados**, na barra lateral.
2. Use o caminho local sugerido ou informe a planilha `.xlsx`/ZIP da UCI.
3. Clique em **Carregar Online Retail**.
4. Use **Analisar** para os alertas diários ou comparações entre meses e
   **Conversar** para consultar faturamento, quantidade, produtos e países.

A primeira leitura da planilha pode demorar; ela é armazenada temporariamente
em cache para evitar novas leituras a cada interação. Mudanças no tamanho ou na
data de modificação do arquivo invalidam essa leitura em cache.

## Significado dos campos

| Campo do programa | Origem e interpretação |
|---|---|
| Data | `InvoiceDate`, agrupada por dia. |
| Produto | `StockCode`, preservado como código textual. |
| Região | `Country`: país do cliente. |
| Quantidade | Soma de `Quantity` das linhas elegíveis. |
| Faturamento | Soma de `Quantity × UnitPrice`, em GBP. |
| Preço unitário | Faturamento dividido pela quantidade de cada grupo. |
| Categoria e vendedor | Não informados: marcadores técnicos, fora dos agrupamentos e do ranking. |
| Desconto | Não informado: o zero técnico não é um desconto observado. |

Não há conversão de moeda. O painel, os gráficos e as explicações monetárias
identificam a libra esterlina por **£**. O valor calculado não representa lucro
nem uma receita líquida de devoluções, impostos ou frete. `CustomerID` e
`Description` não são incorporados à tabela analítica.

## Escopo e conferência

A importação exclui, nesta ordem, faturas iniciadas por `C`, linhas com quantidade
não positiva e linhas restantes com preço não positivo. A ordem torna os
descartes disjuntos e permite conferir suas contagens e somas. Essa seleção define
o escopo das vendas analisadas; não classifica devoluções ou cancelamentos como
anomalias. Linhas repetidas não são removidas automaticamente. A ausência de
identificação do cliente não exclui uma venda.

Na importação local conferida, o arquivo tinha 541.909 linhas. Foram mantidas
530.104 linhas de venda e excluídas 11.805. O agrupamento diário por produto e país
resultou em 304.760 registros, com 3.922 códigos de produto e 38 países.
As contagens e os totais reconciliaram com a origem, respeitada a tolerância
numérica. O registro completo fica em
`outputs/real_data/online_retail/import_audit.json` e também pode ser consultado
em **Cobertura e limites desta base real**, no site.

As datas observadas vão de 01/12/2010 a **09/12/2011**. O último mês cobre menos
dias do calendário, o que deve ser considerado antes de interpretar uma queda
mensal. Para uma comparação entre meses inteiros, outubro e novembro de 2011
são uma opção de investigação, sem supor uma causa comercial conhecida.

## Exportação e limites da avaliação

Em **Dados e resultados**, baixe a base padronizada e o arquivo de definições e
auditoria. O CSV isolado não armazena os atributos de moeda e disponibilidade.
Para recarregar os dados corretamente no site, use **UCI Online Retail** com o
arquivo original, em vez de enviar o CSV isolado como uma base genérica.

A base não fornece rótulos independentes de anomalias nem um calendário de
promoções. Os alertas são hipóteses para revisão; sua contagem não mede
precisão ou acurácia. O experimento com gabarito disponível na área acadêmica
continua sendo fictício. Nenhum resultado da UCI foi acrescentado automaticamente
ao relatório do TCC.
