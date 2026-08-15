# DermoBench e Clinical Context Ablation: comparação pré-treino e E1

## Material Passport

- Origin Skill: `experiment-agent`
- Origin Mode: `validate`
- Origin Date: 2026-08-09; atualizações E1 e Qwen 3.8 27B em 2026-08-14/15
- Verification Status: `ANALYZED`
- Version Label: `dermobench_qwen_comparison_v4_e1_consolidated`
- Overall Confidence: `CAUTION`

## 1. Objetivo e estado da avaliação

Esta nota compara os modelos pré-treino **Qwen 3.5 4B** e **Qwen 3.6 27B**
com dois checkpoints de três épocas da fase label-only E1:

- **E1 Frozen:** LoRA na componente linguística, com a visão congelada;
- **E1 Vision LoRA:** LoRA na linguagem e nas camadas visuais.

Acrescenta também o novo **Qwen 3.8 27B local** apenas ao Clinical Context
Ablation. O DermoBench não foi executado para este modelo; as respetivas
células permanecem assinaladas com “—”.

Foram consideradas duas avaliações:

1. o DermoBench filtrado, com tarefas de descrição, morfologia, diagnóstico,
   reasoning e DDI;
2. o Clinical Context Ablation, que compara a mesma imagem com e sem contexto
   clínico reportado pelo utilizador.

Os resultados foram calculados diretamente a partir dos ficheiros locais
`predictions.jsonl` e `metrics.json`. Todas as respostas preservadas no RunPod
foram sincronizadas localmente.

O estado final dos dados é:

| Condição | DermoBench preservado | Clinical Context | Cobertura interpretável |
|---|---:|---:|---|
| Qwen 3.5 4B pré-treino | 29.099 / 29.099 | 522 / 522 | 9 tarefas determinísticas; 4 abertas sem score clínico final |
| Qwen 3.6 27B pré-treino | 28.173 / 29.099 | 522 / 522 | 9 determinísticas; reasoning incompleto |
| Qwen 3.8 27B pré-treino | — | 522 / 522 | Clinical Context completo; DermoBench não executado |
| E1 Frozen, época 3 | 10.701 / 10.701 | 522 / 522 | 7 tarefas de diagnóstico selecionadas |
| E1 Vision LoRA, época 3 | 10.701 / 10.701 | 522 / 522 | 7 tarefas de diagnóstico selecionadas |

O DermoBench pré-treino ainda **não tem resultados clínicos finais** para as
quatro tarefas que requerem LLM-as-a-judge. Os checkpoints E1 não foram
executados nessas quatro tarefas nem nas duas tarefas MCQ de morfologia.
Por isso, esta nota apresenta:

- resultados finais das nove tarefas determinísticas;
- qualidade e validade dos outputs;
- resultados completos do Clinical Context Ablation;
- estado de cobertura das tarefas abertas;
- limitações que impedem declarar um vencedor global definitivo no
  DermoBench.

Nas tabelas, **“—” significa que a tarefa não foi avaliada ou não dispõe de
score final**; nunca significa zero.

## 2. Protocolo executado

| Parâmetro | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Backend | vLLM local | vLLM local |
| Precisão | BF16 | BF16 |
| Thinking | disabled | disabled |
| Temperature | 0,6 | 0,6 |
| Top-p | 0,95 | 0,95 |
| Top-k | 20 | 20 |
| Min-p | 0,0 | 0,0 |
| Presence penalty | 1,5 | 0,0 |
| Repetition penalty | 1,0 | 1,0 |
| Concorrência máxima | 8 | 8 |

Os casos completos foram emparelhados por `task_id`: os dois modelos receberam
a mesma imagem, prompt, opções e referência. Outputs inválidos ou truncados
permanecem no denominador da accuracy, representando desempenho end-to-end e
não apenas a resposta do modelo quando o formato é válido.

Esta não é uma experiência pura de scaling. Além do número de parâmetros,
mudam a geração do modelo, a versão Qwen e o `presence_penalty`. As diferenças
não devem ser atribuídas exclusivamente a 4B versus 27B parâmetros.

Os dois checkpoints E1 usaram o mesmo perfil histórico do Qwen 3.5 4B
pré-treino: `temperature=0,6`, `top_p=0,95`, `top_k=20`,
`presence_penalty=1,5`, seed 42 e thinking desativado. A comparação Frozen
versus Vision LoRA é, por isso, a ablação E1 mais controlada. Todos os casos
E1 foram emparelhados pelos mesmos IDs das tarefas. A época 3 permanece nesta
tabela porque os checkpoints de continued fine-tuning das épocas 4 e 5 foram
rejeitados pelo critério congelado de seleção em `sft_dev`.

O Qwen 3.8 27B usou BF16 vLLM local, commit
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, temperatura zero, seed 42 e
thinking desativado numa RTX PRO 6000 Blackwell. Por diferir no perfil de
geração e hardware, a comparação com os runs históricos abaixo é descritiva,
não uma ablação controlada.

## 3. Resultados determinísticos do DermoBench

### 3.1 Accuracy principal

| Tarefa | Casos | Qwen 3.5 4B | Qwen 3.6 27B | Diferença 27B - 4B | Melhor resultado |
|---|---:|---:|---:|---:|---|
| Derm7pt morphology MCQ | 5.530 | 38,35% | **52,37%** | +14,01 pp | 27B |
| SkinCon morphology MCQ | 9.736 | **62,53%** | 62,21% | -0,32 pp | Empate prático |
| Diagnosis MCQ, 25 opções | 1.757 | **45,53%** | 21,86% | -23,68 pp | 4B |
| Diagnosis MCQ, 4 opções | 1.757 | 61,47% | **71,31%** | +9,85 pp | 27B |
| DDI diagnosis MCQ | 656 | 48,02% | **60,37%** | +12,35 pp | 27B |
| Derm1M EDU diagnosis MCQ | 3.615 | 40,86% | **60,33%** | +19,47 pp | 27B |
| Derm7pt diagnosis MCQ | 2.022 | 36,35% | **45,50%** | +9,15 pp | 27B |
| SNU134 diagnosis MCQ | 240 | 56,25% | **69,17%** | +12,92 pp | 27B |
| DDI Fairness MCQ | 654 | 49,24% | **62,54%** | +13,30 pp | 27B |

O Qwen 3.6 27B obteve a melhor accuracy em **sete das nove** tarefas
determinísticas. A diferença no SkinCon é de apenas 0,32 pontos percentuais e
não constitui uma vantagem convincente do 4B.

Como resumo descritivo, e não como métrica oficial do DermoBench:

| Agregação das nove tarefas | Qwen 3.5 4B | Qwen 3.6 27B |
|---|---:|---:|
| Média não ponderada da accuracy por tarefa | 48,73% | **56,18%** |
| Micro-accuracy sobre 25.967 requests | 50,34% | **56,46%** |

A micro-accuracy é dominada pelas tarefas SkinCon e Derm7pt, que têm muito mais
casos. A média por tarefa atribui o mesmo peso a benchmarks de tamanhos e
objetivos diferentes. Nenhum destes dois resumos deve substituir a tabela por
tarefa.

### 3.2 Comparação com a tabela publicada no estudo DermoGPT

O estudo que introduziu o DermoBench é o preprint de Ru et al. (2026),
*DermoGPT: Open Weights and Open Data for Morphology-Grounded Dermatological
Reasoning MLLMs* (arXiv:2601.01868). A Tabela 3 do artigo compara 16 baselines,
quatro variantes DermoGPT e uma baseline humana. A tabela abaixo reproduz os
valores publicados e acrescenta os quatro modelos/checkpoints ISEP. Conserva
apenas as oito colunas de accuracy que foram executadas na totalidade pelos
dois modelos Qwen pré-treino; nos checkpoints E1, as duas colunas de morfologia
ficam marcadas com um traço porque não foram executadas.

#### Legenda das tarefas e colunas

Todas as oito tarefas abaixo são avaliadas deterministicamente por
`exact-choice match`: a resposta conta como correta apenas quando a opção
selecionada corresponde à referência. Cada célula da tabela principal apresenta
a **Top-1 accuracy**; valores mais altos são melhores.

| Coluna | Tarefa DermoBench | O que avalia | Formulação | Casos no artigo | Casos ISEP |
|---|---|---|---|---:|---:|
| `T1.3 D7pt` | T1.3, Derm7pt morphology MCQ | Reconhecimento de atributos dermoscópicos do checklist Derm7pt, como pigment network, streaks e outros padrões. | A pergunta seleciona um atributo e o modelo escolhe o respetivo estado entre as opções válidas. | 5.530 | 5.530 |
| `T1.4 SkinCon` | T1.4, SkinCon morphology MCQ | Reconhecimento de conceitos morfológicos em fotografias clínicas, como nodule, plaque, scale ou excoriation. | O modelo escolhe o conceito visual compatível com a anotação SkinCon. | 11.682 | 9.736 |
| `ID 4-way` | T2.1, ID diagnosis 4-way MCQ | Diagnóstico fino ao nível da doença. Os distractores são preferencialmente doenças vizinhas ou irmãs na ontologia, tornando as opções clinicamente confundíveis. | Uma imagem e quatro diagnósticos possíveis. | 2.000 | 1.757 |
| `ID 25-way` | T2.2, ID diagnosis 25-way MCQ | Triagem diagnóstica mais abrangente. As 325 doenças finas da ontologia são agrupadas em 25 categorias clínicas de nível mais geral. | Uma imagem e um menu fixo de 25 categorias. | 2.000 | 1.757 |
| `OOD Derm1M` | T2.4, Derm1M EDU diagnosis MCQ | Generalização para o split educacional externo do Derm1M. | Diagnóstico MCQ construído no espaço de labels original do Derm1M. | 3.615 | 3.615 |
| `OOD DDI` | T2.4, DDI diagnosis MCQ | Generalização para fotografias clínicas do DDI, um conjunto concebido para incluir diversidade de tons de pele. | Diagnóstico MCQ construído no espaço de labels original do DDI. | 656 | 656 |
| `OOD D7pt` | T2.4, Derm7pt diagnosis MCQ | Generalização diagnóstica para imagens dermoscópicas do Derm7pt, em vez de reconhecimento isolado de atributos. | Diagnóstico MCQ construído no espaço de labels original do Derm7pt. | 2.022 | 2.022 |
| `OOD SNU` | T2.4, SNU134 diagnosis MCQ | Generalização para fotografias clínicas externas do SNU134. | Diagnóstico MCQ construído no espaço de labels original do SNU134. | 240 | 240 |

Abreviaturas usadas na tabela:

- **MCQ/MCQA**: pergunta ou avaliação de escolha múltipla;
- **ID (in-distribution)**: imagens estritamente separadas do treino, mas
  provenientes das mesmas fontes usadas para construir o DermoInstruct;
- **OOD (out-of-distribution)**: fontes externas e distribuições visuais
  diferentes. As labels permanecem no espaço original de cada dataset e não
  são convertidas para a ontologia unificada;
- **D7pt**: Derm7pt;
- **SFT**: supervised fine-tuning;
- **RL**: reinforcement learning;
- **CCT**: Confidence-Consistency Test-time adaptation, a agregação de vários
  rollouts usada pelas variantes DermoGPT `+ CCT`;
- **Params**: número aproximado de parâmetros publicamente indicado; `n/d`
  significa não disponível.

| Origem | Modelo | Params | T1.3 D7pt | T1.4 SkinCon | ID 4-way | ID 25-way | OOD Derm1M | OOD DDI | OOD D7pt | OOD SNU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Artigo | GPT-4o-mini | n/d | 41,19% | 61,09% | 59,50% | 34,75% | 52,12% | 58,54% | 56,48% | 59,17% |
| Artigo | Claude-Sonnet-4.5-Thinking | n/d | 29,73% | 59,20% | 55,35% | 34,15% | 53,64% | 52,90% | 50,40% | 68,75% |
| Artigo | Gemini-2.5-Flash | n/d | 39,28% | 66,59% | 72,60% | 47,20% | 66,33% | 59,15% | 53,96% | 65,42% |
| Artigo | GLM-4.5V | 106B | 45,50% | 52,03% | 63,65% | 28,85% | 45,51% | 48,17% | 43,08% | 57,08% |
| Artigo | Qwen2.5-VL-72B | 72B | 52,91% | 60,51% | 61,50% | 35,95% | 54,63% | 54,88% | 58,36% | 66,67% |
| Artigo | QVQ-72B-Preview | 72B | 49,77% | 59,20% | 64,65% | 47,30% | 60,53% | 53,66% | 56,92% | 62,92% |
| Artigo | Llama-3.2-90B | 90B | 35,84% | 49,19% | 47,85% | 51,65% | 44,76% | 49,09% | 37,14% | 49,58% |
| Artigo | Llama-3.2-11B | 11B | 39,13% | 29,93% | 29,25% | 16,50% | 25,50% | 21,80% | 26,90% | 42,92% |
| Artigo | Nemotron-Nano | 12B | 38,72% | 59,20% | 47,25% | 25,60% | 44,12% | 39,48% | 36,84% | 52,08% |
| Artigo | Qwen3-VL-32B | 32B | 46,15% | 60,67% | 64,25% | 38,05% | 48,13% | 57,93% | 63,11% | 69,58% |
| Artigo | Qwen3-VL-8B (Base) | 8B | 40,43% | 62,06% | 67,20% | 45,35% | 52,67% | 51,07% | 59,10% | 55,42% |
| Artigo | HuatuoGPT-Vis-7B | 7B | 33,82% | 38,15% | 51,60% | 26,05% | 31,40% | 36,13% | 41,64% | 47,92% |
| Artigo | LLaVA-Med-v1.5 | 7B | 40,15% | 56,42% | 49,65% | 32,40% | 41,38% | 36,74% | 33,63% | 37,08% |
| Artigo | SkinVL-PubMM | 7B | 43,62% | 61,31% | 57,15% | 38,75% | 51,12% | 48,93% | 58,95% | 54,58% |
| Artigo | Lingshu-32B | 32B | 43,47% | 52,39% | 53,45% | 38,40% | 30,29% | 34,91% | 32,24% | 45,83% |
| Artigo | Lingshu-7B | 7B | 43,92% | 46,08% | 49,55% | 31,90% | 25,95% | 32,16% | 33,88% | 40,00% |
| Artigo | DermoGPT-SFT | 8B | 53,69% | 75,56% | 89,55% | 64,30% | 68,91% | 62,80% | 65,88% | 59,17% |
| Artigo | DermoGPT-SFT + CCT | 8B | 54,10% | 75,92% | 89,75% | 64,45% | 70,65% | 64,33% | 65,58% | 61,25% |
| Artigo | DermoGPT-RL | 8B | 56,53% | 76,67% | 90,30% | 64,60% | 69,68% | 62,80% | 68,59% | 60,00% |
| Artigo | DermoGPT-RL + CCT | 8B | 56,94% | 77,22% | 89,60% | 65,40% | 71,56% | 62,96% | 70,13% | 61,25% |
| Artigo | Human Performance | — | 83,00% | 92,00% | 85,00% | 77,00% | 94,00% | 86,00% | 89,00% | 93,00% |
| **ISEP** | **Qwen 3.5 4B** | **4B** | **38,35%** | **62,53%** | **61,47%** | **45,53%** | **40,86%** | **48,02%** | **36,35%** | **56,25%** |
| **ISEP** | **Qwen 3.6 27B** | **27B** | **52,37%** | **62,21%** | **71,31%** | **21,86%** | **60,33%** | **60,37%** | **45,50%** | **69,17%** |
| **ISEP E1** | **Qwen 3.5 4B — Frozen (época 3)** | **4B** | **—** | **—** | **72,28%** | **54,47%** | **53,55%** | **54,57%** | **46,79%** | **58,33%** |
| **ISEP E1** | **Qwen 3.5 4B — Vision LoRA (época 3)** | **4B** | **—** | **—** | **73,82%** | **56,18%** | **54,08%** | **53,96%** | **50,00%** | **63,75%** |

#### Como interpretar esta tabela

Os resultados do artigo e os resultados ISEP aparecem juntos para facilitar a
leitura, mas **não formam um leaderboard estritamente emparelhado**:

- o artigo usou a release original e as configurações de inferência dos autores;
- a avaliação ISEP usou `temperature=0,6`, thinking desativado, vLLM e o
  contrato de output do adapter local;
- T1.4 foi reduzida de 11.682 para 9.736 casos e os dois MCQ ID foram reduzidos
  de 2.000 para 1.757 casos, depois da remoção de imagens ou grupos com
  sobreposição ao treino;
- T1.3 e as quatro avaliações OOD mantiveram os tamanhos publicados: 5.530,
  3.615, 656, 2.022 e 240 casos, respetivamente;
- outputs inválidos ou truncados contam como incorretos nos resultados ISEP;
  isto afeta especialmente o Qwen 3.6 27B no MCQ de 25 opções;
- as linhas `DermoGPT-*` não são modelos base comparáveis em igualdade de
  treino: foram especializadas com DermoInstruct, e algumas usam RL e CCT;
- as duas linhas `ISEP E1` também são modelos especializados: receberam o
  mesmo treino supervisionado label-only nas 21 classes do ISEP, ao contrário
  das linhas generalistas pré-treino;
- nos checkpoints E1, “—” em T1.3 e T1.4 significa que as tarefas de
  morfologia não foram executadas. As seis células diagnósticas vêm das seis
  tarefas visual-diagnosis comparáveis à tabela do artigo; a sétima tarefa E1,
  DDI Fairness, é apresentada separadamente na Secção 3.4 porque a coluna
  `Fair.` publicada não é comparável à metadata local;
- `Human Performance` foi calculado no artigo com uma amostra de 100 casos por
  tarefa, não com todos os casos da coluna.

As colunas T1.1, T1.2, T3.1 e T3.2 foram deliberadamente excluídas porque os
Qwen deste projeto ainda não têm scores do judge comparáveis. Nos checkpoints
E1, essas tarefas nem sequer foram executadas: a campanha foi deliberadamente
limitada às tarefas determinísticas de diagnóstico visual, sem
LLM-as-a-judge. A coluna
hierárquica também foi excluída porque essa tarefa de 2.000 rows não está na
release pública usada localmente. Finalmente, a coluna `Fair.` do artigo mede
`min(group accuracy) / max(group accuracy)`, enquanto a nossa execução só
preservou skin tone `unknown`; por isso, a accuracy bruta da Task 4 não foi
colocada indevidamente nessa coluna.

Em leitura apenas descritiva, o Qwen 3.6 27B fica próximo dos melhores modelos
generalistas publicados em T1.3, ID 4-way, DDI e SNU. No entanto, fica abaixo
das variantes DermoGPT especializadas e da baseline humana na maioria das
colunas. O resultado ID 25-way não deve ser interpretado como falta de
conhecimento clínico sem considerar a taxa de escolhas recuperáveis de apenas
41,83%.

**Fonte bibliográfica:** Ru, J., Yan, S., Yin, Y., Zou, Y., & Ge, Z. (2026).
*DermoGPT: Open weights and open data for morphology-grounded dermatological
reasoning MLLMs* [Preprint, arXiv:2601.01868].
[arXiv:2601.01868](https://arxiv.org/abs/2601.01868). Valores publicados
extraídos da Tabela 3, versão 1, submetida em 5 de janeiro de 2026. O
manuscrito deve ser citado como preprint; a fonte consultada não demonstra
revisão por pares ou aceitação.

### 3.3 Comparação emparelhada

Foi aplicado McNemar exato à correção Top-1 de cada caso, contando outputs sem
escolha recuperável como incorretos. Os nove testes foram corrigidos por Holm.

- As vantagens do 27B em Derm7pt morphology, DDI diagnosis, Derm1M EDU,
  Derm7pt diagnosis, diagnosis com quatro opções, SNU134 e DDI Fairness
  mantêm `p_Holm < 0,01`.
- A diferença SkinCon não é estatisticamente suportada
  (`p_Holm = 0,558`).
- A vantagem observada do 4B no MCQ de 25 opções mantém
  `p_Holm < 0,001`, mas é principalmente uma falha de geração/output do 27B,
  analisada abaixo.

### 3.4 Checkpoints E1 no inventário completo de tarefas

A tabela seguinte torna explícita a cobertura desigual. Os checkpoints E1
foram avaliados apenas nas sete tarefas que pedem diretamente o diagnóstico da
doença a partir da imagem. As células com “—” correspondem a tarefas não
executadas ou sem julgamento clínico final.

| Tarefa DermoBench | Casos | 4B pré-treino | 27B pré-treino | E1 Frozen | E1 Vision LoRA |
|---|---:|---:|---:|---:|---:|
| 1.1 Description sem morphology | 783 | — | — | — | — |
| 1.2 Description com morphology | 783 | — | — | — | — |
| 1.3 Derm7pt morphology MCQ | 5.530 | 38,35% | **52,37%** | — | — |
| 1.4 SkinCon morphology MCQ | 9.736 | **62,53%** | 62,21% | — | — |
| 2.1 Diagnosis MCQ, 25 opções | 1.757 | 45,53% | 21,86% | 54,47% | **56,18%** |
| 2.1 Diagnosis MCQ, 4 opções | 1.757 | 61,47% | 71,31% | 72,28% | **73,82%** |
| 2.1 DDI diagnosis MCQ | 656 | 48,02% | **60,37%** | 54,57% | 53,96% |
| 2.1 Derm1M EDU diagnosis MCQ | 3.615 | 40,86% | **60,33%** | 53,55% | 54,08% |
| 2.1 Derm7pt diagnosis MCQ | 2.022 | 36,35% | 45,50% | 46,79% | **50,00%** |
| 2.1 SNU134 diagnosis MCQ | 240 | 56,25% | **69,17%** | 58,33% | 63,75% |
| 3.1 Diagnostic reasoning sem morphology | 783 | — | — | — | — |
| 3.2 Diagnostic reasoning com morphology | 783 | — | — | — | — |
| 4 DDI Fairness MCQ | 654 | 49,24% | **62,54%** | 58,10% | 56,88% |

Para uma comparação estritamente comum, foram agregadas apenas as mesmas sete
tarefas de diagnóstico (`n=10.701` por modelo):

| Agregação das sete tarefas comuns | 4B pré-treino | 27B pré-treino | E1 Frozen | E1 Vision LoRA |
|---|---:|---:|---:|---:|
| Micro-accuracy | 45,45% | 53,35% | 55,95% | **57,28%** |
| Média não ponderada por tarefa | 48,24% | 55,87% | 56,87% | **58,38%** |
| Taxa ponderada de escolhas válidas | 99,89% | 87,65% | 95,63% | **97,27%** |

O E1 Vision LoRA superou o 27B no agregado comum, mas este resultado não
significa que venceu de forma uniforme: foi melhor em três das sete tarefas e
o 27B em quatro. O ganho agregado é influenciado fortemente pelo MCQ de 25
opções, onde o 27B sofreu truncamentos e outputs inválidos.

Nos testes emparelhados por tarefa, com correção de Holm:

- E1 Vision LoRA superou significativamente o 27B no MCQ de 25 opções e no
  Derm7pt diagnosis;
- o 27B superou significativamente E1 Vision LoRA em DDI, Derm1M EDU e DDI
  Fairness;
- as diferenças no MCQ de quatro opções e no SNU134 não permaneceram
  significativas;
- no agregado de todos os casos comuns, Vision LoRA superou Frozen por
  1,33 pp (`p=0,00049`), mas este teste agregado mistura tarefas e deve ser
  lido juntamente com os resultados por tarefa.

## 4. Validade e comportamento dos outputs

### 4.1 Escolhas recuperáveis nas tarefas determinísticas

| Tarefa | 4B: escolha válida | 27B: escolha válida |
|---|---:|---:|
| Derm7pt morphology | 100,00% | 98,81% |
| SkinCon morphology | 100,00% | 91,64% |
| Diagnosis, 25 opções | 99,83% | **41,83%** |
| Diagnosis, 4 opções | 99,94% | 99,49% |
| DDI diagnosis | 99,70% | 91,77% |
| Derm1M EDU diagnosis | 99,83% | 99,34% |
| Derm7pt diagnosis | 100,00% | 89,91% |
| SNU134 diagnosis | 100,00% | 100,00% |
| DDI Fairness | 100,00% | 98,62% |
| **Total, 25.967 requests** | **99,95%** | **91,52%** |

O principal problema do 27B é a tarefa de diagnóstico com 25 opções:

- 572 respostas terminaram com estado `ok`;
- 57 ficaram `format_invalid`;
- 1.128 ficaram `truncated_output`;
- o parser ainda recuperou uma escolha em parte das respostas truncadas,
  resultando em 735 escolhas válidas no total;
- 1.022 casos continuaram sem escolha recuperável.

Isto explica a queda para 21,86% de accuracy end-to-end. Entre as 735 respostas
em que foi possível recuperar uma escolha, o 27B acertou 384, ou 52,24%.
Este valor condicional **não substitui** a accuracy principal: os casos válidos
constituem uma amostra selecionada pelo próprio comportamento do modelo.

No mesmo conjunto de imagens, mas com apenas quatro opções, o 27B alcançou
71,31% de accuracy e 99,49% de escolhas válidas. Portanto, a falha de 25 opções
é mais consistente com output excessivo, truncation e dificuldade em obedecer
ao contrato de resposta do que com incapacidade geral para analisar as
imagens.

### 4.2 Estado bruto em todo o DermoBench preservado

| Estado | Qwen 3.5 4B, n=29.099 | Qwen 3.6 27B, n=28.173 preservados |
|---|---:|---:|
| `ok` | 29.080 (99,93%) | 25.808 (91,61%) |
| `format_invalid` | 2 (0,01%) | 1.229 (4,36%) |
| `truncated_output` | 17 (0,06%) | 1.136 (4,03%) |

O 4B é claramente mais fiável na entrega de uma resposta processável. O 27B
é clinicamente mais forte na maioria das tarefas, mas precisa de maior controlo
de output antes de ser usado para gerar dados sintéticos em escala.

## 5. Tarefas abertas e limitações de cobertura

As tarefas 1.1, 1.2, 3.1 e 3.2 requerem avaliação posterior pelo judge
**Gemini 3.5 Flash-Lite**. Até essa etapa ser executada, os scores de qualidade
da descrição, reasoning, factualidade e concordância clínica permanecem
ausentes.

| Tarefa aberta | 4B pré-treino | 27B pré-treino | E1 Frozen | E1 Vision | Comparação possível agora |
|---|---:|---:|---:|---:|---|
| 1.1 Description sem morphology | 783 / 783 | 783 / 783 | — | — | Apenas formato; ambos os modelos pré-treino 100% |
| 1.2 Description com morphology | 783 / 783 | 783 / 783 | — | — | Apenas formato: 98,21% vs 100% |
| 3.1 Reasoning sem morphology | 783 / 783 | **640 / 783** | — | — | Parcial; faltam 143 no 27B |
| 3.2 Reasoning com morphology | 783 / 783 | **0 / 783** | — | — | Impossível comparar |

Nos 640 IDs comuns da tarefa 3.1, a conformidade estrutural foi 77,03% no 4B e
99,53% no 27B. Isto mede apenas a presença e ordenação dos blocos exigidos; não
mede a qualidade clínica do reasoning.

### Limitação principal

O Qwen 3.6 27B tem **926 casos de reasoning em falta**:

- 143 casos restantes na tarefa 3.1;
- todos os 783 casos da tarefa 3.2.

A interrupção foi uma decisão operacional explícita para não continuar a
execução extensa de reasoning. Não foi uma recusa clínica, erro de imagem ou
falha automática do endpoint. Os casos em falta não devem ser codificados como
incorretos, mas também não podem ser ignorados para produzir um score global.

Consequentemente:

- não existe uma comparação completa de reasoning entre os dois modelos;
- não deve ser calculado um DermoBench total agregado como se ambos tivessem
  terminado todas as tarefas;
- os 640 outputs preservados do 27B representam uma execução parcial e não uma
  amostra aleatória pré-definida;
- o 27B não pode ser declarado vencedor global do DermoBench apenas com estes
  dados;
- a seleção final de teacher ainda precisa dos judgments das tarefas abertas
  que efetivamente foram preservadas.

## 6. Clinical Context Ablation

Esta avaliação contém 261 imagens SCIN e duas condições emparelhadas por
imagem: `image_only` e `image_plus_context`, totalizando 522 requests por
modelo.

### 6.1 Resultados por condição

| Modelo e condição | Top-1 | Top-3 | Top-6 | MRR | JSON válido | Schema válido |
|---|---:|---:|---:|---:|---:|---:|
| 4B — imagem | 27,20% | **66,28%** | **83,14%** | 48,07% | 100% | 99,23% |
| 4B — imagem + contexto | **31,03%** | 65,90% | 81,99% | **49,78%** | 100% | 100% |
| 27B — imagem | 38,31% | **81,99%** | **93,49%** | 60,16% | 100% | 100% |
| 27B — imagem + contexto | **41,38%** | 78,16% | 92,34% | **61,02%** | 100% | 100% |
| Qwen 3.8 27B — imagem | 39,85% | 78,54% | 90,04% | 58,88% | 100% | 100% |
| Qwen 3.8 27B — imagem + contexto | **44,44%** | **81,61%** | **91,57%** | **62,78%** | 100% | 100% |
| E1 Frozen — imagem | 37,16% | 80,84% | 93,49% | 59,03% | 100% | 100% |
| E1 Frozen — imagem + contexto | 40,23% | 79,31% | 94,25% | 60,97% | 100% | 100% |
| E1 Vision — imagem | 46,74% | 81,99% | 94,25% | 65,25% | 100% | 99,62% |
| E1 Vision — imagem + contexto | **47,13%** | **82,76%** | **95,79%** | **66,72%** | 100% | 100% |

O 27B foi superior ao 4B nas duas condições. A diferença Top-1 foi de 11,11
pontos com imagem e 10,34 pontos com imagem mais contexto. A comparação
emparelhada entre modelos permaneceu significativa após correção de Holm nas
duas condições (`p_Holm = 0,0044` e `0,0045`, respetivamente).

Após E1, Vision LoRA tornou-se a condição local com melhor Top-1 nas duas
variantes: superou o 27B pré-treino em 8,43 pp com imagem isolada e 5,75 pp com
imagem mais contexto. Esta melhoria mede especialização no corpus E1, não uma
vantagem geral de modelos pequenos sobre modelos grandes.

O Qwen 3.8 27B obteve o melhor Top-1 entre os três checkpoints generalistas
locais nesta ablação (39,85% sem contexto e 44,44% com contexto), mas continua
abaixo do E1 Vision nas duas condições. Esta observação não é uma comparação
de eficiência nem uma prova causal de vantagem arquitetural.

### 6.2 Efeito do contexto dentro de cada modelo

| Métrica: contexto - imagem | 4B pré-treino | 27B pré-treino | Qwen 3.8 27B | E1 Frozen | E1 Vision |
|---|---:|---:|---:|---:|---:|
| Top-1 | +3,83 pp | +3,07 pp | +4,60 pp | +3,07 pp | +0,38 pp |
| Top-3 | -0,38 pp | -3,83 pp | +3,07 pp | -1,53 pp | +0,77 pp |
| Top-6 | -1,15 pp | -1,15 pp | +1,53 pp | +0,77 pp | +1,53 pp |
| MRR | +1,71 pp | +0,86 pp | +3,90 pp | +1,94 pp | +1,47 pp |
| Pares que melhoraram / pioraram em Top-1 | 35 / 25 | 40 / 32 | 52 / 40 | 28 / 20 | 27 / 26 |
| IC 95% da diferença Top-1 | -1,92 a +9,58 pp | -3,45 a +9,58 pp | -2,30 a +11,88 pp | -2,30 a +8,43 pp | -4,98 a +5,75 pp |
| McNemar exato | p=0,245 | p=0,410 | p=0,251 | p=0,312 | p=1,000 |

Os cinco deltas Top-1 são positivos, mas todos os intervalos incluem zero e
nenhum teste emparelhado é significativo. O efeito torna-se praticamente nulo
no E1 Vision (+0,38 pp). Estes resultados **não demonstram** que adicionar
contexto melhora a classificação; mostram apenas uma tendência exploratória
dependente do checkpoint e da métrica.

### 6.3 Skin tone no Clinical Context

Os grupos Monk 1-3 e 4-6 têm suporte estatístico segundo o protocolo da
benchmark; Monk 7-10 tem apenas 17 casos e deve permanecer descritivo.

| Grupo | Casos | 4B imagem | 4B + contexto | 27B imagem | 27B + contexto | Qwen 3.8 27B imagem | Qwen 3.8 27B + contexto |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monk 1-3 | 164 | 30,49% | 34,15% | 39,63% | 42,68% | 42,07% | 42,07% |
| Monk 4-6 | 80 | 23,75% | 27,50% | 37,50% | 40,00% | 33,75% | 48,75% |
| Monk 7-10 | 17 | 11,76% | 17,65% | 29,41% | 35,29% | 47,06% | 47,06% |

Estas taxas são brutas e podem refletir diferenças de doença, fonte e
qualidade da imagem. Não devem ser interpretadas como prova de fairness.

## 7. DDI Fairness: o nome da tarefa não garante uma fairness válida

O 27B obteve 62,54% contra 49,24% do 4B na tarefa DDI Fairness. Contudo, todos
os 654 casos aparecem no grupo de skin tone `unknown` nos artefactos desta
execução. O `fairness_score=1.0` é produzido mecanicamente quando não existem
dois grupos conhecidos para comparar.

Assim:

- a accuracy geral DDI pode ser comparada;
- o valor `fairness_score=1.0` não deve ser reportado como evidência de
  equidade;
- é necessário recuperar ou derivar metadados de skin tone válidos antes de
  usar esta tarefa para uma conclusão de fairness.

## 8. Interpretação para teacher e student

### Evidência favorável ao Qwen 3.6 27B

- vence sete das nove tarefas determinísticas;
- apresenta ganhos grandes em Derm7pt morphology, DDI, Derm1M EDU, Derm7pt
  diagnosis e SNU134;
- supera o 4B no Clinical Context tanto com imagem isolada como com contexto;
- nos outputs de reasoning preservados, obedece melhor ao formato pedido.

### Evidência favorável ao Qwen 3.5 4B

- entrega outputs muito mais fiáveis: 99,95% de escolhas recuperáveis contra
  91,52% no conjunto determinístico;
- não entra no loop/truncation observado no MCQ de 25 opções;
- mantém uma accuracy competitiva apesar de ter aproximadamente seis vezes
  menos parâmetros;
- continua adequado à hipótese central da tese como student oficial.

### Evidência após especialização E1

- E1 Frozen supera o 4B pré-treino nas sete tarefas de diagnóstico comuns;
- E1 Vision LoRA obtém o melhor agregado comum, 57,28% de micro-accuracy, e
  supera Frozen por 1,33 pp;
- Vision LoRA é a melhor condição local no Clinical Context, com 46,74% Top-1
  em imagem e 47,13% em imagem mais contexto;
- o ganho visual não é uniforme: Frozen permanece ligeiramente melhor em DDI
  e DDI Fairness, e o 27B continua melhor em quatro das sete tarefas externas;
- como seis tarefas DermoBench não foram avaliadas nos checkpoints E1, estes
  resultados não constituem um score global do DermoBench.

### Conclusão provisória

Antes do fine-tuning, o Qwen 3.6 27B era o modelo clinicamente mais forte na
maioria das tarefas determinísticas, mas não era um teacher operacionalmente
pronto sem controlo adicional de comprimento e formato. Depois de três épocas
label-only, o 4B especializado alcançou melhor agregado nas sete tarefas comuns
e melhor desempenho no Clinical Context. Esta é evidência favorável à hipótese
da tese — um SLM especializado pode superar um modelo maior genérico num
domínio controlado — mas não prova superioridade universal: a direção depende
da tarefa e o 27B ainda vence quatro das sete tarefas externas.

Ainda não é metodologicamente correto escolher o teacher final apenas com esta
nota. Faltam os judgments de descrição e rationale, e a cobertura de reasoning
do 27B é incompleta. Para destilação, qualquer uso do 27B deve guardar o raw
output, aplicar validação determinística e excluir ou reparar exemplos
truncados antes de formar o dataset sintético.

## 9. Limitações e full disclosure

1. **Tarefas em falta no 27B:** faltam 143 casos da tarefa 3.1 e todos os 783
   casos da tarefa 3.2.
2. **Judge pendente:** ainda não existem scores clínicos Gemini 3.5 Flash-Lite
   para as quatro tarefas abertas.
3. **Missingness não aleatória:** a tarefa 3.1 foi interrompida durante a
   execução; os 640 casos preservados não constituem necessariamente um subset
   representativo.
4. **Output como parte da accuracy:** escolhas inválidas e truncadas contam
   como incorretas. Isto é adequado para desempenho end-to-end, mas mistura
   conhecimento clínico com obediência ao formato.
5. **Uma geração por caso:** não foram executadas repetições por seed; existe
   variabilidade de sampling.
6. **Não é uma ablação pura de tamanho:** versões do modelo e
   `presence_penalty` também diferem.
7. **Candidate-set effect:** a degradação do 27B em 25 opções e o bom resultado
   em quatro opções mostram que a dificuldade depende do desenho do MCQ.
8. **Fairness DDI indisponível:** todos os skin tones estão `unknown`.
9. **Clinical Context limitado:** 261 pares, classes e distribuição SCIN
   específicas; o ganho Top-1 não foi significativo.
10. **Sem conclusão externa:** estes resultados medem as releases locais
    filtradas e não garantem desempenho clínico real.
11. **Seleção e múltiplas análises:** as comparações determinísticas usaram
    correção de Holm, mas análises de subgrupos e métricas secundárias continuam
    exploratórias.
12. **Comparação com o artigo:** os valores publicados usam a release e o
    protocolo dos autores; as linhas ISEP usam uma release filtrada e outro
    protocolo de inferência. A tabela conjunta é contextual, não uma replicação
    direta nem um ranking formal.
13. **Estado da fonte:** o estudo DermoGPT consultado é o arXiv v1 de 5 de
    janeiro de 2026 e deve ser tratado como preprint.
14. **Cobertura E1 parcial:** as tarefas 1.1, 1.2, 1.3, 1.4, 3.1 e 3.2 não
    foram avaliadas nos checkpoints E1; “—” não é zero.
15. **Uma seed E1:** esta comparação usa apenas seed 3407 de treino e uma
    geração de benchmark por caso; a estabilidade entre seeds ainda não foi
    demonstrada.
16. **Agregado sensível ao contrato:** a vantagem agregada do E1 Vision sobre
    o 27B é ampliada pelo MCQ de 25 opções, onde o 27B teve muitos truncamentos.

## 10. Fallacy scan

- Coverage: **11/11 tipos verificados**.
- Simpson: não foi inferida uma tendência global a partir de subgrupos com
  direções potencialmente diferentes.
- Ecological fallacy: resultados de grupos Monk não foram convertidos em
  afirmações sobre indivíduos.
- Berkson/selection bias: a release é filtrada e não representa toda a
  dermatologia clínica.
- Collider bias: não foi estimado um modelo causal com covariáveis.
- Base-rate neglect: accuracy MCQ depende do candidate set e não representa
  valor preditivo numa população clínica.
- Regression to the mean: não aplicável a este desenho.
- Survivorship/attrition: os casos de reasoning ausentes no 27B foram
  explicitamente divulgados e excluídos de um agregado global.
- Look-elsewhere effect: os nove testes pré-treino e os sete testes E1 versus
  27B foram corrigidos por Holm nas respetivas famílias; restantes métricas
  são exploratórias.
- Garden of forking paths: a temperatura 0,6 foi selecionada numa análise
  anterior e deve ser tratada como configuração de sensibilidade congelada,
  não como um novo teste selado independente.
- Correlation versus causation: o efeito de contexto é uma comparação
  emparelhada dentro desta benchmark; não suporta causalidade clínica externa.
- Reverse causality: não aplicável à classificação emparelhada.

## 11. Artefactos analisados

- `outputs/dermobench_full_v1/temp_0_6_thinking_off/`
- `outputs/clinical_context_ablation_v1/temp_0_6_thinking_off/`
- `outputs/qwen_3_8_27b_full_benchmarks/clinical_context_ablation/`
- `outputs/e1_epoch3_historical_t06_benchmarks/`
  - E1 Frozen e E1 Vision LoRA, 14 tarefas cada na campanha selecionada;
  - cópia local verificada contra o RunPod: 227 ficheiros, 499.275.681 bytes;
  - digest agregado SHA-256:
    `244eaaaf27c55fa1ed6d51d8dd49a1b68de0a1eb9016ebfd1bb6c9734af574fa`.
- `runs/benchmarks/dermobench_then_context_controller.log`
- Ru et al. (2026), arXiv:2601.01868, Tabela 3.
- Hugging Face `mendicant04/DermoBench`, card e task manifest consultados em
  2026-08-09.

### Proveniência da pesquisa bibliográfica

- Pesquisa executada em 2026-08-09.
- Consulta principal: título exato de DermoBench associado ao repositório
  `mendicant04/DermoGPT`.
- Fonte primária dos resultados: arXiv:2601.01868v1, Tabela 3.
- Fonte primária da release pública: card oficial
  [mendicant04/DermoBench](https://huggingface.co/datasets/mendicant04/DermoBench).
- Critério de inclusão: apenas colunas com accuracy determinística e cobertura
  completa nos dois modelos Qwen.
- Critério de exclusão: judge pendente, reasoning incompleto, tarefa ausente da
  release pública ou métrica sem metadados necessários.
