# Experimento fictício com promoções e quedas

Este experimento verifica se o calendário pode ajudar a distinguir uma mudança prevista no cenário de uma queda inesperada. Ele não usa a Olist e não faz consultas a APIs.

O gerador cria vendas diárias de quatro produtos com padrão semanal, tendência leve e ruído. No cenário, promoções aumentam a quantidade vendida e aplicam desconto. Dias com e sem promoção aparecem no treino, na calibração e no teste. Cinco quedas-surpresa adicionais são introduzidas somente no teste: duas em dias consecutivos sem promoção e três em dias distintos de promoção.

A relação entre promoções e vendas foi programada no gerador. Os resultados mostram o comportamento nesse mundo fictício; não provam que uma promoção causou uma mudança em uma empresa real.

## Executar

No site, carregue **Demonstração → Promoções e quedas → Carregar base de exemplo**.
A tela **Analisar** apresenta os dias sinalizados e permite entender cada um deles.
O modo **Experimentos do TCC → Promoções e quedas** reúne a comparação entre métodos e o gabarito.
Quando outra base está ativa, esse modo também permite gerar um experimento fictício independente.

No PowerShell, a partir da pasta do projeto:

```powershell
.\.venv\Scripts\python.exe run_context_experiment.py
```

O padrão usa 420 dias e as sementes `42`, `137` e `2026`. Outras sementes podem ser declaradas antes da avaliação:

```powershell
.\.venv\Scripts\python.exe run_context_experiment.py --seeds 42 137 2026 --output-dir outputs/contextual
```

## O que cada detector recebe

Os primeiros 50% dos dias formam o treino; os próximos 25%, a calibração; o restante, o teste. O detector histórico recebe as vendas e usa o método já disponível no sistema. O contextual recebe as vendas e um calendário contendo somente `data` e `promocao`.

O modelo contextual estima suas associações no treino e mantém esse ajuste fixo. Seu limite de alerta é definido apenas na calibração. A referência do método histórico continua usando observações passadas, como no comportamento já implementado. Ambos são avaliados nos mesmos dias elegíveis do teste.

O gabarito fica separado. As funções de detecção não recebem a classificação dos eventos nem seus efeitos esperados. Os rótulos só são associados aos resultados após as previsões.

## Como interpretar os números

Nesta tarefa, uma anomalia é uma **mudança inesperada adicional**. A mudança de venda prevista pelo gerador em um dia de promoção é um comportamento esperado. Esse é um critério específico do experimento, não uma definição universal de anomalia comercial.

- **Precisão:** entre os alertas emitidos, quantos correspondem a surpresas conhecidas.
- **Recuperação:** entre as surpresas conhecidas avaliáveis, quantas foram encontradas.
- **F1:** combinação da precisão e da recuperação.
- **Alertas em contexto esperado:** alertas nos dias de promoção sem surpresa adicional. Eles são erros conforme o critério deste cenário.
- **Surpresas em dias com contexto:** situações em que há promoção e também uma queda inesperada. Conhecer o contexto não deve dispensar a investigação dessas quedas adicionais.

Datas sem contexto suficiente podem ficar sem decisão. Uma comparação indisponível é registrada como tal e não recebe métricas de sucesso. As colunas terminadas em `_n` indicam quantas execuções contribuíram para cada média. Não há ajuste de parâmetros escolhido a partir dos rótulos do teste nem afirmação de superioridade geral.

## Arquivos exportados

Os arquivos ficam em `outputs/contextual/`:

| Arquivo | Conteúdo |
|---|---|
| `vendas.csv` | Vendas no formato padrão do projeto, da primeira semente |
| `calendario.csv` | Contexto diário da primeira semente, sem gabarito |
| `gabarito.csv` | Eventos, efeitos programados e classificação da primeira semente |
| `point_results.csv` | Previsões dos dois métodos, todas as sementes e máscara de avaliação |
| `metrics.csv` | Métricas por método e semente |
| `aggregate_metrics.csv` | Médias, desvios e contagens de valores válidos |
| `summary.json` | Configuração, divisões temporais, modelos, hashes e limitações |
| `report.md` | Resumo legível dos resultados |

As vendas podem ser abertas pelo importador CSV existente. O calendário separado é utilizado pelo experimento contextual; o importador comum de vendas não o incorpora automaticamente.

O relatório do TCC não é alterado por esse comando. Resultados deste experimento devem ser discutidos como testes sintéticos caso sejam incluídos em uma revisão posterior do texto.

## Referência contextual

O modelo usa o logaritmo da receita, dia da semana, tendência e um indicador de promoção. O ajuste robusto é feito somente no treino, com mínimos quadrados ponderados iterativamente e pesos de Huber. As associações permanecem congeladas na calibração e no teste. Não são estimativas de efeitos causais nem previsões validadas de vendas futuras.

O ajuste no logaritmo utiliza apenas observações positivas do treino. Um zero observado e completo na calibração ou no teste pode ser avaliado como queda total. Valores negativos, registros ausentes e datas sem informação de contexto ficam sem decisão.

Referências técnicas: [mínimos quadrados no NumPy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) e [modelos lineares robustos e funções de peso](https://www.statsmodels.org/stable/rlm.html). A implementação usa NumPy; não depende de instalar statsmodels.
