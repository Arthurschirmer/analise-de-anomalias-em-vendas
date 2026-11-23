# Uso dos recursos de análise e preparação

## Ficha de cada alerta

Em **Analisar → Por dia ou semana**, execute a análise e selecione **Alerta para
investigar**. A ficha reúne valor observado, referência, diferença e diferença
percentual quando a referência é positiva. Ela informa o número de observações
anteriores, o escore e o limite que gerou o alerta.

Escolha uma dimensão em **Detalhar os registros do alerta por** para conferir
os grupos observados naquele dia ou semana. Campos indisponíveis na fonte ficam
fora das opções. A distribuição não é uma decomposição da referência estatística
e não confirma uma causa. A ficha também pode ser baixada em JSON. A revisão humana
utiliza o mesmo alerta selecionado, evitando duas seleções diferentes.

## Cobertura e comparação de dias

O painel e a comparação mensal mostram as primeiras e últimas datas observadas
de cada mês, a quantidade de dias com registros e os dias do calendário. Datas
que não alcançam as bordas do mês geram um aviso. Esse aviso não comprova que
faltaram registros: também pode haver dias sem vendas.

Em **Analisar → Entre meses → Comparar períodos**, escolha o intervalo:

- **Meses conforme os registros da base:** conserva os totais mensais originais.
- **Mesmo intervalo de dias nos dois meses:** utiliza o início mais tardio e o
  término mais cedo entre os dias observados nas bordas dos dois meses.

Por exemplo, se um mês tem registros do dia 1 ao 31 e o outro do dia 1 ao 9, a
segunda opção compara somente os dias 1 a 9 de ambos. O recorte acompanha o
relatório e as explicações. Nenhum dia sem registros é preenchido com zero.
Igualar os números dos dias não iguala dias da semana, feriados ou promoções.

A base completa permanece preservada. Os gráficos por grupo da investigação
usam o mesmo recorte dos cálculos. Indicadores operacionais mensais da Olist
ficam fora da comparação recortada, pois têm outro universo temporal.

## Importação guiada de CSV e Excel

1. Escolha **Enviar CSV / Excel** na barra lateral e selecione um arquivo `.csv`
   ou `.xlsx`, de até 64 MB.
2. Confira o separador e a codificação do CSV, ou escolha a aba da planilha.
3. Veja a prévia e associe as colunas do arquivo às do programa. As sugestões
   são apenas pontos de partida e precisam de conferência.
4. Informe moeda, formato de data e separador decimal. Não há conversão cambial.
5. Escolha se preço e faturamento vêm de colunas ou se um deles será calculado.
6. Clique em **Validar mapeamento**, confira o resultado e depois em **Usar esta
   base**. As escolhas do formulário são validadas novamente no carregamento.

Data, produto e quantidade são necessários. Faturamento e preço podem vir de
colunas, ou um pode ser derivado do outro por uma definição explícita. Os dois
não podem ser derivados ao mesmo tempo. Categoria, região, vendedor e desconto
podem ficar indisponíveis. Marcadores técnicos não viram informações observadas.

Calcular faturamento como quantidade × preço × (1 − desconto) exige que essa
fórmula tenha sentido na origem. Sem desconto informado, o zero é uma hipótese
explícita do cálculo, e não uma observação de ausência de desconto. Ao calcular
preço por faturamento ÷ quantidade, quantidade zero torna a linha inválida.

As regras específicas dos adaptadores Olist e UCI continuam nas opções próprias.
Usar a importação genérica não aplica automaticamente filtros de cancelamento
ou as definições comerciais dessas bases.

## Validação e qualidade da base

O carregador, os adaptadores e a interface compartilham a validação estrutural.
Ela confere colunas, nomes duplicados, datas, produto e números finitos. Datas
numéricas sem uma definição de formato não são interpretadas como instantes
próximos de 1970. Valores negativos de negócio geram avisos e permanecem nos
dados; a política de devoluções deve ser definida para cada origem.

Na importação guiada, o padrão é bloquear linhas inválidas e mostrar os problemas.
A exclusão só ocorre ao escolher explicitamente essa política. O resumo registra
quantas linhas foram válidas, inválidas e carregadas, com posições de exemplo
na tabela lida. Linhas de texto dentro de um campo CSV ou linhas em branco podem
fazer essa posição diferir da linha física do arquivo.

Em **Dados e resultados → Qualidade e preparação da base**, confira o resumo
e as definições aplicadas. Uma validação estrutural bem-sucedida não certifica
a correção comercial dos valores ou a completude da coleta.

## Histórico das investigações

Novas investigações mensais concluídas são salvas automaticamente em
`outputs/investigations/`. Cada arquivo guarda as evidências, o relatório, o modo
local/IA, o horário da execução e uma identificação dos dados e de suas definições.
As chaves de API configuradas na aplicação não são incluídas nesses registros.

Em **Dados e resultados → Investigações salvas**, escolha um registro e clique
em **Reabrir investigação**. A reabertura usa os resultados guardados, sem
refazer cálculos ou chamar a IA. A mesma base e as mesmas definições precisam
estar carregadas. Registros de outras bases podem ser identificados pela opção
**Mostrar também registros de outras bases**; não são apresentados como análises
da base ativa.

Selecione duas investigações da mesma base para conferir seus valores lado a lado.
Períodos e recortes diferentes precisam ser considerados na interpretação; a
tabela não mede superioridade de um método. O histórico guarda resultados e
configurações, não uma cópia completa dos dados de entrada.

O armazenamento é local neste computador. Falhas ao salvar são avisadas sem
descartar o resultado já calculado. Arquivos ilegíveis são sinalizados sem
ocultar os demais. Para testes automatizados, `ANOMALY_HISTORY_DIR` permite
usar uma pasta temporária e manter separado o histórico de uso.

## Conversa vinculada ao alerta

Em **Analisar → Por dia ou semana**, selecione um alerta e clique em
**Perguntar sobre este alerta**. A conversa mantém o intervalo, o indicador,
a referência, o limite e o segmento selecionado. Você pode perguntar
"Por que este dia foi sinalizado?", "Qual foi a quantidade?", "Mostre por
produto" ou "O que conferir agora?". Os cálculos e as respostas deste modo
são locais, sem consulta à API. A distribuição dos grupos usa somente os
registros do intervalo e do segmento; ela não atribui a referência aos grupos.

Na demonstração com promoções, o mesmo botão aparece no dia
sinalizado, com o calendário informado. Esse calendário faz parte da
referência e não confirma a causa do desvio restante. Perguntas com datas
explícitas fora do intervalo são orientadas a sair desse contexto.

**Sair do alerta** ou **Nova conversa** libera as consultas gerais.
Carregar outra base também limpa o contexto e as mensagens anteriores.
O conjunto de perguntas locais é limitado aos fatos e consultas implementados.

## Conversa sobre uma comparação mensal

Depois de investigar dois meses, **Conversar sobre esta investigação** abre o
chat com a comparação escolhida. A tela mostra os meses em foco. Perguntas
como "e por que isso ocorreu?" ou "o que devo conferir agora?" reutilizam os
resultados guardados, incluindo moeda e recorte de dias, sem repetir a chamada
ao agente ou à API. Informar outros meses inicia outra comparação.

Também é possível perguntar "por que as vendas mudaram em dezembro de 2014?".
Quando há registros em novembro, o programa usa esse mês como referência e
informa essa escolha na resposta. Um mês sem predecessor disponível exige
escolher outra referência. Uma pergunta sem datas e sem comparação em foco
pede esclarecimento; a aplicação não escolhe silenciosamente um dos alertas.

Na conversa geral, a seleção **Escolha uma variação sinalizada para conversar**
lista as mudanças mensais do painel. Selecione uma e use **Investigar a variação
selecionada**. **Nova conversa**, **Sair da comparação** ou o carregamento de
outra base limpam o contexto anterior. O fluxo dos alertas diários e semanais
mantém seu próprio intervalo e segmento.

## Perfis de importação

Depois de validar o arquivo, abra **Guardar estas configurações como perfil**,
dê um nome e clique em **Salvar perfil**. São guardadas as opções da última
validação: associações de colunas, moeda, formato de datas e números, desconto,
cálculos derivados, política para linhas inválidas e leitura de CSV/aba de Excel.
Se alterar as opções, valide novamente antes de salvar o novo perfil.

Na próxima importação, escolha **Perfil de importação salvo → Aplicar perfil**.
As colunas e a aba precisam existir no novo arquivo. Confira as opções,
valide os novos registros e clique em **Usar esta base**. Um perfil não
autoriza o carregamento automático nem certifica os dados de outro arquivo.

Os perfis ficam em `outputs/import_profiles/` neste computador e não incluem
os registros de vendas nem as credenciais da aplicação. É possível definir
outra pasta com `ANOMALY_IMPORT_PROFILE_DIR`. Arquivos inválidos são ignorados
com aviso. Os perfis não substituem os filtros específicos dos adaptadores
Olist e UCI.

## Relatórios em PDF

**Baixar investigação em PDF** está disponível no resultado mensal e em
**Dados e resultados**. O documento inclui origem, horário da análise, modo
local/IA, comparação de valores, gráfico, principais grupos, decomposição,
definições e limitações. Moeda e recorte de dias acompanham o resultado;
reabrir uma investigação permite exportá-la sem executar a análise novamente.

**Baixar alerta em PDF** exporta a ficha do dia ou semana: valores observado
e esperado, diferença, escore, limite, gráfico e próximos passos. A ficha
histórica também inclui a distribuição selecionada; a demonstração contextual
inclui o calendário informado. A geração usa ReportLab localmente, sob demanda,
sem API, navegador ou compilação de LaTeX. Estes são relatórios do programa.
