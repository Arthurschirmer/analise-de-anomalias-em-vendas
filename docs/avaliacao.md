# Protocolo de avaliação e explicação

## Detecção histórica

A tela **Histórico** agrega por dia ou por semana (segunda a domingo).
Na análise diária, cada ponto é comparado ao mesmo dia da semana nas oito semanas
anteriores; no mínimo quatro observações são necessárias. Na semanal, são usadas
semanas anteriores completas. O usuário pode alterar a janela e o histórico mínimo.

Sem tendência, a referência é a mediana do passado. Com tendência, usa-se a mediana
das inclinações entre todos os pares anteriores e a mediana dos interceptos,
extrapolando para o ponto atual. A escala é o maior entre MAD dos resíduos × 1,4826,
5% do módulo da referência e 1 unidade. O escore é a distância absoluta do observado
à referência dividida pela escala. São limites empíricos de decisão, sem garantia
de cobertura probabilística. A tendência é local; não há modelo de feriados,
sazonalidade anual ou efeitos comerciais.

Os primeiros 50% do calendário são treino, os 25% seguintes calibração, e o restante
teste. Na configuração padrão atual, o limiar é o máximo entre 3 e
**mediana dos escores + 3 × 1,4826 × MAD dos escores** elegíveis da calibração.
A opção anterior, máximo entre 3 e quantil 98% (interpolação superior), permanece
disponível na interface e nos experimentos. O método escolhido permanece congelado durante o teste.
A referência usa somente pontos anteriores ao avaliado, inclusive observações
anteriores do teste, em uma avaliação sequencial. Não usa o valor atual nem dados
futuros para definir sua referência. Rótulos do teste não calibram o limiar.

Datas sem transações são desconhecidas por padrão. A alternativa de preencher zero
é uma hipótese explícita do usuário, apenas no intervalo observado. Semanas de
borda ou com datas desconhecidas são inelegíveis. A extração retrospectiva da Olist
usa status finais; não simula o que estaria disponível numa aplicação em tempo real.

## Experimento reproduzível

Execute na raiz do projeto:

```powershell
& ".\.venv\Scripts\python.exe" run_benchmark.py --olist-dir data/olist_raw
```

O gerador controlado usa 420 dias, quatro produtos, tendência, padrão semanal e
ruído com semente 42. No teste são injetados quatro eventos de dois dias:
queda de volume, pico de volume, aumento de valor unitário do maior produto local
e queda de volume da maior categoria local. A alteração de quantidade é arredondada
e a receita varia proporcionalmente, preservando a identidade de faturamento.
É uma simulação de intensidade, sem garantia de plausibilidade comercial.

Os três métodos usam a mesma máscara de pontos elegíveis no teste:

- Baseline: variação absoluta acima de 30% contra o mesmo dia da semana anterior.
- Histórico: referência sazonal com tendência, MAD e limiar calibrado anteriormente.
- Histórico anterior: a mesma referência com quantil 98%, preservado como comparação.

Nos dados controlados, os dias não injetados são normais por definição do gerador:
podem ser calculados precisão, recall, F1 e matriz de confusão por ponto/dia.
Isso não converte a taxa de acerto sintética em taxa de acerto na Olist.

Na segunda etapa, copia-se o recorte Olist 2017-01 a 2018-07 para injetar os mesmos
tipos de evento. Os arquivos brutos ficam preservados. O limiar usa a calibração
original, sem injeções. Somente pontos injetados recebem rótulos positivos;
todos os demais pontos reais continuam sem rótulo. Portanto, publicam-se recall
das injeções e quantidade de alertas sem rótulo, sem precisão ou F1 reais.
No site é usado o recorte Olist atualmente carregado, informado nos metadados.
Um evento de segmento pode ser diluído no total diário e não ser detectado.

Os arquivos de saída em `outputs/benchmark/` são:

- `summary.json`: semente, parâmetros, partições, hash da entrada padronizada, métricas e limites.
- `point_results.csv`: decisões, referência, limites, rótulos e máscara de avaliação.
- `events.csv`: eventos e detecção por método.
- `metrics.csv`: métricas por etapa e detector.
- `manual_review.csv`: modelo de revisão dos alertas reais sem rótulo.
- `report.md`: relatório com protocolo e limitações.

Uma execução demonstrativa não demonstra superioridade geral. Repita com outras
sementes e outros períodos, documentando hipóteses e escolhendo parâmetros antes
do teste final. Não ajuste o detector com os rótulos do teste e apresente esse
mesmo teste como validação independente. Resultado fraco também deve ser relatado.

## Explicações e investigação

Cada comparação mensal agora inclui uma identidade contábil:

1. Na cesta de produtos comuns, calcula-se o efeito de volume ao mix e valores
   unitários da referência.
2. Depois o efeito de mix ao volume atual e valores unitários da referência.
3. Depois o efeito de valor unitário nas quantidades atuais.
4. Receitas de entradas e saídas observadas são adicionadas separadamente.

A soma reconcilia a mudança de faturamento. A ordem é declarada, pois influencia
os efeitos. Valor unitário efetivo é receita/quantidade do mesmo produto, não
necessariamente preço de tabela. Entrada/saída significa presença nesses meses,
não lançamento/retirada definitiva. O cálculo é descritivo e não prova causalidade.

Na Olist, pedidos cancelados e atrasos são indicadores separados, por mês de compra.
O cancelamento usa todos os pedidos da coorte; o atraso usa somente entregas com
datas válidas e status entregue. Compara-se a data da entrega com o dia previsto.
Pendentes e datas ausentes ficam fora do denominador; a taxa é observada e pode
sofrer censura. Os resultados não compartilham o mesmo universo da receita de itens.

## Comparação local e IA

Na tela **Avaliação → Relatório local e IA**, primeiro faça uma investigação.
Por padrão é gerado o relatório determinístico e a IA fica `not_run`.
Marque a consulta à API e execute explicitamente para obter uma resposta remota.
Falhas são registradas como `failed`; a resposta local continua disponível.
Uma falha nunca é contada como avaliação bem sucedida da IA.

Também existe um comando offline:

```powershell
& ".\.venv\Scripts\python.exe" compare_reports.py
```

Para incluir a API configurada, acrescente `--with-ai`. Para a Olist:

```powershell
& ".\.venv\Scripts\python.exe" compare_reports.py --dataset-type olist --previous-period 2017-10 --current-period 2017-11
```

As duas explicações recebem prompts e contexto idênticos, calculados localmente.
São registrados hashes, modelo, latência de cada geração e uso de tokens quando o
provedor fornece esse dado. A verificação numérica procura correspondências nas
evidências, reconhece formatos decimais comuns e ignora datas ISO e identificadores
longos. É heurística: não verifica sentido, unidade, causalidade, nem valida todos
os valores derivados. Formatos numéricos ambíguos exigem revisão manual.
Latência de uma chamada não caracteriza o desempenho geral do provedor.

Fidelidade, clareza e utilidade recebem notas humanas de 1 a 5, com responsável
e justificativa. Antes desse preenchimento, o parecer é `pending`.
Os registros são salvos em `outputs/model_comparison/<hash-da-requisição>/`.

## Revisão de alertas reais

Na tela Histórico, selecione um alerta e registre o responsável, parecer e
justificativa. Os pareceres ficam em `outputs/reviews/`, vinculados ao hash dos
dados e dos parâmetros, e podem ser exportados. Eles não são reaproveitados
automaticamente na calibração. A revisão apenas de alertas não mede recall global;
para isso seria necessário revisar também pontos não sinalizados e definir uma
amostra e protocolo independentes. O CSV do benchmark é um artefato para revisão
externa e não é importado automaticamente pela interface.

## Referências de método

- [Forecasting: Principles and Practice — componentes de séries](https://otexts.com/fpp3/decomposition.html).
- [SciPy — desvio absoluto mediano e robustez](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
- [scikit-learn — validação temporal e uso do passado](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [Fonte pública Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

O projeto implementa suas próprias rotinas de referência, MAD e divisão temporal;
os links são referências conceituais, não dependências adicionais.

## Calibração e detecção por categoria/UF

O quantil alto dos escores pode ser dominado por poucos dias extremos da
calibração. A alternativa de mediana + 3 MAD escalados é uma regra operacional
explícita, não uma garantia estatística de cobertura ou taxa de falsos positivos.
Os diagnósticos registram mediana, MAD, máximo, ambos os limiares, proporção de
excedências na calibração e sua diferença. A proporção de excedências não é FPR.

No recorte Olist 2017-01 a 2018-07, com a divisão fracionária anterior, a calibração
2017-10-18 a 2018-03-09 produziu limiar 15,1503 pelo quantil e 6,1476 pela regra
robusta. Esse diagnóstico motivou uma alternativa exploratória; não demonstra
melhor desempenho geral nem autoriza chamar o teste já examinado de independente.

A tela Histórico também permite analisar categorias e regiões (UF na Olist).
Todos os segmentos compartilham calendário e limites temporais. Os grupos são
ordenados pela métrica no treino, com participação mínima padrão de 1%,
máximo padrão de 12 grupos e pelo menos 28 dias observados no treino (8 semanas,
na agregação semanal). A referência e a calibração são próprias de cada grupo.
A tabela mostra os motivos de exclusão e os pontos sem decisão.
Grupos novos, raros ou esparsos não são classificados automaticamente como normais.

Datas ausentes continuam desconhecidas por padrão. Com zero explícito, só se
assume ausência de vendas do segmento em dias em que a base global contém registros
e após sua primeira observação. Dias sem qualquer registro global continuam
desconhecidos. Uma categoria que desaparece sem registros pode, portanto, ficar
sem decisão; distinguir ausência de venda e falha na coleta exige informação externa.

A seleção de vários segmentos aumenta as oportunidades de sinalização.
Não há correção formal para múltiplas comparações; os alertas precisam de revisão.
As evidências de categorias e UFs não devem ser somadas entre si.

## Suíte ampliada

```powershell
& ".\.venv\Scripts\python.exe" run_benchmark.py --suite --seeds 42 137 2026 --olist-dir data/olist_raw --output-dir outputs/benchmark/suite
```

São onze cenários predefinidos: controle sem alteração; quedas e picos de 10%,
30% e 60%; queda gradual de 14 dias; queda persistente de 10 dias; queda em categoria
e em UF compensada por outros segmentos, mantendo o faturamento global constante.

Nos choques de volume, arredonda-se o TOTAL diário uma vez. A quantidade inteira
é distribuída entre linhas por maiores restos, com desempate reprodutível pela
semente; os preços são preservados e a receita acompanha as quantidades realizadas.
Isso evita que linhas de uma unidade tornem os choques pequenos inócuos.
A intensidade nominal é de quantidade, não necessariamente de receita:
a composição dos itens mantidos/removidos influencia o efeito realizado.
O CSV `suite_injection_effects.csv` registra quantidades alvo/realizadas e receitas
antes/depois por dia. Cenários sem efeito efetivo ficam identificados.

Os cenários compensados usam um choque no valor unitário efetivo do segmento-alvo,
compensado no restante do mercado. São construções experimentais para verificar
a diluição no total, sem alegação de mecanismo comercial realista.
O total preservado não recebe rótulo de anomalia; o segmento-alvo é avaliado separado.
Segmentos compensatórios não são classificados automaticamente como normais.

O conjunto controlado usa a divisão 50%/25%/25%. Na suíte Olist os blocos são
treino jan–ago/2017, calibração set–dez/2017 e teste jan–jul/2018.
**Esses dados já foram examinados no desenvolvimento:** o experimento é exploratório,
mesmo com blocos cronológicos distintos. A avaliação final independente exige dados
ou um protocolo ainda não usado para escolher as alterações.
As sementes da Olist variam posições de injeção no mesmo histórico, sem criar
bases reais independentes.

Os três detectores recebem a mesma máscara de avaliação em cada canal.
A suíte registra métricas por cenário/semente, atraso até a primeira detecção,
médias, desvios e número de valores válidos de cada métrica.
Controles sem positivos não usam F1 como medida de qualidade; em dados controlados,
a taxa de falsos alertas descreve os negativos conhecidos.
Não há intervalos de confiança nem afirmação de superioridade geral.

A suíte pode mostrar que um método recupera mais injeções e produz mais alertas
sem rótulo na Olist. Esses alertas não são falsos positivos confirmados.
Métricas reais globais continuam indisponíveis sem rótulos humanos adequados.

## Amostra de avaliação humana

No Histórico, abra **Preparar avaliação humana com pontos sem alerta**. A amostra
inclui apenas pontos elegíveis do teste e sorteia sem reposição dentro de dois
estratos: sinalizados e não sinalizados. O padrão divide a amostra entre ambos,
reaproveitando vagas se um estrato tiver poucos pontos. A alocação é desproporcional;
cada registro guarda população do estrato, tamanho da amostra, probabilidade de
inclusão e peso `N_h/n_h`.

O painel é salvo com identificação da base, parâmetros, semente, pontos e desenho.
Depois acesse **Avaliação → Revisão humana**. Classifique cada ponto como
Anomalia confirmada, Comportamento esperado ou Inconclusivo, com responsável e
justificativa. Pontos sem alerta não recebem um rótulo automático de normalidade.
A alteração de um parecer mantém seu histórico. Pareceres de outro painel
explicitamente identificado não são reutilizados.

A opção de ocultar decisão esconde alerta, escore, referência e limites enquanto
o painel está pendente. Valores ausentes preenchidos como zero são identificados
como hipóteses, inclusive em semanas parcialmente preenchidas. O histórico geral
continua acessível: esta interface isolada não garante um estudo cego formal.
A visibilidade da decisão no momento de cada parecer é registrada.

Enquanto faltar um parecer conclusivo, não se publicam precisão, recall ou F1
desse painel. Quando ele estiver completo, a matriz ponderada estima concordância
do detector com a referência humana nos pontos elegíveis do teste.
As razões ponderadas são estimativas amostrais, sem garantia de ausência de viés
e sem intervalo de confiança. Um censo integral é identificado separadamente.
Pareceres humanos podem divergir e não demonstram causalidade nem verdade absoluta.

A revisão direcionada apenas dos alertas permanece disponível no Histórico,
mas seus resultados não são usados como recall global. Nenhum parecer é inventado
ou preenchido automaticamente pelo sistema.


