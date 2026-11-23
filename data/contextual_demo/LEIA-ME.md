# Vendas fictícias com contexto

Exemplo gerado com a semente **42**, por `generate_contextual_sales` em `src/data/contextual_demo.py`.

São 420 dias, de 2 de janeiro de 2023 a 25 de fevereiro de 2024, com quatro produtos. Há dias com e sem promoção. Cinco dias do teste recebem uma queda extra inesperada: dois dias consecutivos sem promoção e três dias distintos durante promoções.

| Arquivo | Uso |
|---|---|
| `vendas.csv` | Vendas no formato aceito pelo programa: 1.680 registros |
| `calendario.csv` | Uma linha por dia com `data` e `promocao` |
| `gabarito.csv` | Classificação e efeitos programados, usados somente após a detecção |

`True` indica presença do acontecimento, e `False`, ausência conhecida. Informação ausente no calendário não equivale a `False`.

Na simulação, uma promoção aumenta a quantidade em 45% e aplica desconto de 10%. As quantidades são arredondadas, por isso o gabarito registra também o efeito realizado. Essas relações são hipóteses do gerador, não afirmações sobre empresas reais.

Neste teste, mudanças previstas de promoção são comportamentos esperados. O alvo da detecção são as quedas extras inesperadas, inclusive quando acontecem junto com promoção. O calendário não contém essa classificação.

Para carregar esta base no site, selecione **Demonstração → Promoções e quedas → Carregar base de exemplo**. As vendas e o calendário são carregados juntos, e a tela **Analisar** apresenta os dias sinalizados, o gráfico e a explicação de cada dia.

Para comparar métodos e conferir o gabarito, abra **Experimentos do TCC → Promoções e quedas**. Quando outra base estiver ativa, **Gerar base e comparar** permite criar outras execuções. O experimento aprende o padrão na primeira metade das datas, define o limite de alerta nos próximos 25% e avalia os últimos 25%.

O CSV de vendas também pode ser aberto em **Enviar CSV**. Esse importador comum não associa automaticamente o calendário; a comparação contextual fica na área de avaliação.
