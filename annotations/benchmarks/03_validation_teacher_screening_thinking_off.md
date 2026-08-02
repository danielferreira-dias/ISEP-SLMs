# Validation teacher screening: thinking-off results and full disclosure

Data: 2026-08-02

## 1. Objetivo e estado desta experiência

Esta nota regista a primeira fase de seleção de um possível teacher para a
tese: comparação de sete modelos multimodais nos mesmos subsets fixos de
Validation do ISEPDermaBench.

Esta fase não é uma avaliação final. Validation é um conjunto de desenvolvimento
e os resultados podem ser usados para selecionar modelos, prompts e parâmetros.
O Internal Benchmark continua selado para a comparação principal antes/depois
do fine-tuning.

Estado no final desta nota:

- fase `thinking_off` concluída para seis modelos;
- Qwen 3.7 Flash via OpenRouter adicionado e concluído com thinking desligado;
- Luna executado uma única vez com o seu protocolo congelado de reasoning high;
- quatro tasks avaliadas por modelo;
- mesmos IDs e mesma ordem de casos em todos os modelos;
- resultados raw, parsed, canonical e judged preservados nos outputs;
- teacher final ainda não escolhido: falta a comparação `thinking_on`.

## 2. Protocolo congelado

### 2.1 Cohorts

| Task | Unidade selecionada | Imagens únicas | Requests avaliados |
| --- | ---: | ---: | ---: |
| Visual Top-K closed set | caso | 100 | 100 |
| Visual Confusion Sets | par | 100 | 200 |
| Evidence-grounded diagnosis | caso | 100 | 100 |
| Open-ended diagnosis | caso | 100 | 100 |

Confusion Sets usa a mesma imagem duas vezes: uma condição de baixa
confusabilidade e outra de alta confusabilidade. Os 200 requests não são 200
imagens independentes.

Os selection hashes foram idênticos para todos os modelos dentro de cada task:

| Task | `selection_sha256` |
| --- | --- |
| Visual Top-K | `db229fd6cdd5488c3269725b9f457e1a7fe586888115d458e750dca760f354ab` |
| Confusion Sets | `4e4dd2b49cc7ea49ff23fd73e617c3cac77cad56d09bc9b02a3507e4d4d5d335` |
| Evidence-grounded diagnosis | `29a5d17d9ab535c801eadc542c7049c166b5d7d506a37d308c5b520887712b09` |
| Open-ended diagnosis | `8334c74328bb3d77e5c5c7ca1d6880fdea25183b543192784ec750bee5ca81e9` |

Os ficheiros de IDs estão em
`data/benchmarks/ISEPDermaBench/metadata/validation_screening_v1/`.

### 2.2 Modelos e thinking

| Modelo | Backend | Thinking efetivo | Observação |
| --- | --- | --- | --- |
| GPT-5.6 Luna | provider API | high | exceção congelada; executado uma vez |
| Qwen 3.5 4B | vLLM/Modal | off | override CLI `disabled` |
| Qwen 3.5 9B | vLLM/Modal | off | override CLI `disabled` |
| Qwen 3.6 27B | vLLM/Modal | off | override CLI `disabled` |
| Gemma 4 E4B | vLLM/Modal | off | override CLI `disabled` |
| Gemma 4 31B | vLLM/Modal | off | override CLI `disabled` |
| Qwen 3.7 Flash | OpenRouter | off | override CLI `disabled` |

No Qwen 3.7 Flash, os quatro manifests registam
`effective_reasoning_enabled: false`. Todos os 499 requests que chegaram a
uma resposta registaram zero reasoning tokens e não devolveram reasoning text
ou summary. Um request de Evidence e um de Open-ended falharam antes da
resposta.

O override tem precedência sobre o `thinking_mode` presente no perfil de
geração do YAML. Portanto, apesar de o perfil original do Qwen 3.7 Flash ter
sido criado como judge candidate, esta execução foi efetivamente sem thinking.

### 2.3 Parâmetros de geração

Não foram impostos parâmetros idênticos a famílias diferentes. Foram usados
os perfis recomendados ou congelados de cada configuração, com a alteração
experimental apenas no thinking:

- Qwen 3.5 4B/9B: temperature 1.0, top-p 0.95, top-k 20,
  presence penalty 1.5 e repetition penalty 1.0;
- Qwen 3.6 27B: temperature 1.0, top-p 0.95, top-k 20,
  presence penalty 0.0 e repetition penalty 1.0;
- Gemma 4: temperature 1.0, top-p 0.95, top-k 64 e repetition penalty 1.0;
- Qwen 3.7 Flash: temperature 0.0, top-p 1.0, presence penalty 0.0 e seed 42;
- Luna: parâmetros do provider e reasoning effort high.

Esta heterogeneidade mede os modelos nos perfis escolhidos para uso real, mas
impede atribuir diferenças exclusivamente à arquitetura ou ao tamanho do
modelo.

## 3. Como interpretar as métricas de output

O relatório separa capacidade clínica de obediência ao contrato de output.

### 3.1 Quatro níveis de validade

1. **Raw JSON validity**: `json.loads` aceita exatamente o texto final.
2. **Recoverable JSON validity**: aceita raw JSON ou um único objeto JSON
   totalmente contido numa fence Markdown, sem texto adicional.
3. **Schema compliance**: campos, tipos, cardinalidade, ranks e IDs obedecem ao
   schema da task.
4. **Semantic compliance**: regras entre campos são respeitadas, por exemplo
   confidence decrescente, `case_confidence` coerente, findings declarados e
   ausência de conteúdo clínico proibido na descrição visual.

O parser é determinístico e deliberadamente limitado. Não usa outro LLM, não
extrai diagnósticos de prosa, não corrige IDs por similaridade e não inventa
campos. As únicas normalizações relevantes são:

- remoção auditada de uma única JSON fence completa;
- no ranked closed set, conversão de uma lista inequívoca de disease IDs para
  objetos `{rank, disease_id}`.

### 3.2 Strict versus canonical accuracy

Nas tasks ranked closed set são reportadas duas accuracies:

- **strict accuracy**: outputs inválidos para o schema contam como ausência de
  prediction;
- **canonical accuracy**: usa apenas as recuperações determinísticas acima.

A canonical accuracy é usada na comparação clínica principal. Raw JSON,
schema e parser intervention são sempre reportados separadamente para não
esconder custos de integração.

Evidence-grounded diagnosis segue uma política field-level: campos
recuperáveis e individualmente válidos contribuem para a métrica clínica,
mesmo quando outra camada falha. Assim, um `case_confidence` incoerente não
apaga um diagnóstico correto. JSON, schema e semantic compliance continuam
como métricas independentes.

### 3.3 Denominadores do open-ended judge

`judge_top_1_accuracy` e `judge_top_3_accuracy` usam como denominador os casos
com judgment válido mais os model failures, que recebem um judgment sintético
de falha. `judge_invalid` é excluído desse denominador e aparece em
`judge_coverage`.

Por transparência, esta nota reporta também uma versão conservadora sobre os
100 casos, tratando qualquer judgment em falta como incorreto.

## 4. Resultados clínicos principais

Todas as percentagens seguintes são sobre os cohorts descritos acima. Visual
Top-K e Confusion usam a representação canonical. Evidence usa a política
field-level recuperável.

| Modelo | Visual Top-K 1/3/6 | Confusion 1/2 | Evidence 1/3/6 | Open judge 1/3 | Rationale 0–4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna, high | 39 / 64 / 78% | 73 / 92.5% | 35 / 56 / 72% | 23 / 41% | **2.52** |
| Qwen 3.5 4B | 34 / 54 / 67% | 72 / 88% | 36 / 62 / 71% | 18.1 / 35.1% | 1.53 |
| Qwen 3.5 9B | 42 / 59 / 76% | 70.5 / 90.5% | 34 / 55 / 73% | 24.0 / 41.7% | 1.80 |
| Qwen 3.6 27B | **44** / 63 / 79% | 71 / 91% | 42 / 62 / 79% | 25.8 / 43.3% | 1.93 |
| Gemma 4 E4B | 19 / 33 / 50% | 59 / 83.5% | 13 / 28 / 43% | 13 / 31% | 1.68 |
| Gemma 4 31B | 41 / **71 / 86%** | **75 / 92%** | **45 / 65 / 83%** | 29.3 / 47.5% | 2.20 |
| Qwen 3.7 Flash | 42 / 64 / 77% | **75 / 92%** | 41 / 57 / 74% | **35.8 / 53.7%** | 2.17 |

Interpretação provisória:

- Gemma 4 31B continua a ser o modelo local mais equilibrado e lidera
  Evidence e Visual Top-3/Top-6;
- Qwen 3.6 27B mantém o melhor Visual Top-1;
- Qwen 3.7 Flash empata o melhor Confusion Top-1 e obtém o melhor resultado
  open-ended, mas esse resultado tem limitações de judge descritas abaixo;
- Luna continua a obter a melhor qualidade média de clinical rationale,
  findings e grounding no open-ended, apesar de menor diagnosis accuracy;
- Gemma 4 E4B é o modelo mais fraco nesta fase.

### 4.1 Confusabilidade

| Modelo | Low-confusability Top-1 | High-confusability Top-1 | Gap |
| --- | ---: | ---: | ---: |
| Luna | 81% | 65% | 16 pp |
| Qwen 3.5 4B | 84% | 60% | 24 pp |
| Qwen 3.5 9B | 82% | 59% | 23 pp |
| Qwen 3.6 27B | 84% | 58% | 26 pp |
| Gemma 4 E4B | 68% | 50% | 18 pp |
| Gemma 4 31B | 88% | 62% | 26 pp |
| Qwen 3.7 Flash | 87% | 63% | 24 pp |

O gap é `accuracy_low - accuracy_high` sobre pares fixos. Um gap menor não
significa automaticamente um modelo melhor: pode resultar de desempenho baixo
nas duas condições.

### 4.2 Evidence e morfologia

| Modelo | Finding F1 | Description F1 | Visible-evidence precision | Unsupported finding rate | Grounded Top-1 | Semantic compliance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 54.0% | 52.5% | 59.1% | 48.0% | 10% | 78% |
| Qwen 3.5 4B | 53.0% | 54.0% | 60.8% | 47.3% | 13% | 34% |
| Qwen 3.5 9B | 59.6% | 60.1% | 66.1% | 46.7% | **15%** | 42% |
| Qwen 3.6 27B | 55.8% | 56.7% | 61.3% | 48.5% | 6% | 44% |
| Gemma 4 E4B | 49.9% | 50.4% | 54.3% | 57.7% | 3% | 20% |
| Gemma 4 31B | 60.0% | 59.5% | 64.2% | 44.2% | 11% | **86%** |
| Qwen 3.7 Flash | **61.9%** | **61.8%** | **67.0%** | **42.6%** | 13% | 21% |

Qwen 3.7 Flash foi o melhor a reproduzir os conceitos morfológicos de
referência, mas apenas 21% dos outputs cumpriram todas as regras semânticas. A
principal causa não foi o diagnóstico: 71 casos tinham `case_confidence`
incoerente com a confidence numérica do primeiro diagnóstico e 35 descreviam
conceitos que não estavam declarados na lista estruturada de findings. Quatro
descrições acionaram o léxico determinístico de conteúdo não visual/proibido.

Isto demonstra porque clinical accuracy, morphology similarity e semantic
compliance não devem ser fundidas numa única accuracy.

### 4.3 Métricas completas de morfologia e descrição

As tabelas seguintes completam o resumo anterior com todas as métricas
principais apresentadas pelo report de Evidence. Precision, recall e F1 são
calculados deterministicamente contra os conceitos de referência. O macro-F1
suportado inclui apenas os seis conceitos com pelo menos 20 casos positivos.

| Modelo | Finding precision | Finding recall | Micro-F1, todos os conceitos | Macro-F1, conceitos suportados |
| --- | ---: | ---: | ---: | ---: |
| Luna | 50.3% | 61.3% | 55.0% | 64.3% |
| Qwen 3.5 4B | 54.3% | 55.8% | 52.6% | 60.5% |
| Qwen 3.5 9B | 55.3% | 68.6% | 59.0% | 66.8% |
| Qwen 3.6 27B | 51.8% | 64.1% | 56.3% | 64.7% |
| Gemma 4 E4B | 43.9% | 62.9% | 50.1% | 61.0% |
| Gemma 4 31B | 56.6% | 66.5% | 59.7% | 69.1% |
| Qwen 3.7 Flash | **58.2%** | **70.1%** | **61.8%** | **70.9%** |

| Modelo | Description precision | Description recall | Findings consistency | Unsupported description concept rate |
| --- | ---: | ---: | ---: | ---: |
| Luna | 50.7% | 58.1% | 94.9% | 49.1% |
| Qwen 3.5 4B | 54.0% | 58.2% | 85.1% | 49.1% |
| Qwen 3.5 9B | 56.6% | 67.9% | 91.2% | 46.2% |
| Qwen 3.6 27B | 51.7% | 67.0% | 90.9% | 50.1% |
| Gemma 4 E4B | 44.3% | 62.7% | 96.6% | 56.8% |
| Gemma 4 31B | 56.2% | 65.8% | **97.4%** | **45.2%** |
| Qwen 3.7 Flash | 56.2% | **73.5%** | 93.7% | 46.2% |

`Description findings consistency` mede a concordância entre conceitos
extraídos da descrição e findings estruturados pelo próprio modelo. Não mede
correção contra a referência. Por isso, uma consistency elevada pode coexistir
com um unsupported concept rate elevado.

### 4.4 Ranking e calibração

| Modelo | MRR | Macro-F1 Top-1 | Brier score | Expected calibration error | Casos calibráveis |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna | 47.5% | 37.0% | 26.2% | 25.7% | 91 |
| Qwen 3.5 4B | 49.2% | 31.8% | 40.7% | 42.7% | 99 |
| Qwen 3.5 9B | 47.2% | 32.5% | 39.5% | 43.7% | 100 |
| Qwen 3.6 27B | 55.0% | 39.0% | 37.3% | 38.7% | 100 |
| Gemma 4 E4B | 22.5% | 9.2% | 57.3% | 68.2% | 100 |
| Gemma 4 31B | **57.7%** | **46.6%** | 26.6% | **20.5%** | 100 |
| Qwen 3.7 Flash | 51.5% | 37.8% | 39.2% | 41.1% | 99 |

MRR e Macro-F1 mais elevados são melhores. Brier score e expected calibration
error mais baixos são melhores. Calibration usa apenas respostas que contêm
uma confidence numérica recuperável; por isso o denominador pode ser inferior
a 100.

### 4.5 Integridade das evidence links e do vocabulário

| Modelo | Valid evidence links | Broken references | Correct diagnosis with unsupported evidence | Invalid concept ID | Invalid disease ID | Forbidden description |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 100% | 0% | 2.9% | 0% | 0% | 0% |
| Qwen 3.5 4B | 100% | 0% | 5.6% | 6% | 3% | 4% |
| Qwen 3.5 9B | 100% | 0% | 2.9% | 7% | 1% | 3% |
| Qwen 3.6 27B | 100% | 0% | **2.4%** | 2% | 0% | 6% |
| Gemma 4 E4B | 100% | 0% | 7.7% | 7% | 12% | 2% |
| Gemma 4 31B | 100% | 0% | 4.4% | 0% | 0% | 0% |
| Qwen 3.7 Flash | 100% | 0% | **2.4%** | 1% | 0% | 4% |

Todos os modelos tiveram duplicate prediction rate de 0%. Duplicate finding
rate foi 0% em todos exceto Gemma 4 E4B, com 2%. O cohort contém 100 casos de
morfologia, 89 com description scoring, 100 com diagnosis scoring, 19 classes
diagnósticas e seis conceitos elegíveis para supported macro-F1.

## 5. Full disclosure de JSON, schema e parser

### 5.1 Visual Top-K

| Modelo | Raw JSON | Recoverable JSON | Strict schema | Canonical schema | Strict Top-1 | Canonical Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 95% | 95% | 95% | 95% | 39% | 39% |
| Qwen 3.5 4B | 100% | 100% | 100% | 100% | 34% | 34% |
| Qwen 3.5 9B | 100% | 100% | 100% | 100% | 42% | 42% |
| Qwen 3.6 27B | 100% | 100% | 100% | 100% | 44% | 44% |
| Gemma 4 E4B | 100% | 100% | 97% | 97% | 19% | 19% |
| Gemma 4 31B | 100% | 100% | 100% | 100% | 41% | 41% |
| Qwen 3.7 Flash | 100% | 100% | 100% | 100% | 42% | 42% |

As cinco respostas Luna em falta foram content-policy violations. Gemma E4B
repetiu disease IDs em três respostas; estas permanecem schema-invalid.

### 5.2 Visual Confusion Sets

| Modelo | Raw JSON | Recoverable JSON | Strict schema | Canonical schema | Strict Top-1 | Canonical Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 99% | 99% | 99% | 99% | 73% | 73% |
| Qwen 3.5 4B | 100% | 100% | 99.5% | 99.5% | 72% | 72% |
| Qwen 3.5 9B | 83.5% | 100% | 83.0% | 99.5% | 61.5% | 70.5% |
| Qwen 3.6 27B | 97% | 100% | 97% | 100% | 69% | 71% |
| Gemma 4 E4B | 96% | 100% | 96% | 100% | 56.5% | 59% |
| Gemma 4 31B | 0% | 100% | 0% | 100% | 0% | 75% |
| Qwen 3.7 Flash | 100% | 100% | 99.5% | 99.5% | 75% | 75% |

Gemma 31B colocou todas as 200 respostas em JSON fences. A capacidade de
ranking foi recuperada sem inferência semântica, mas a strict JSON validity é
zero e seria necessária uma camada de integração em produção.

### 5.3 Evidence-grounded diagnosis

| Modelo | Raw JSON | Recoverable JSON | Schema | Semantic | End-to-end `ok` | Clinical Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 91% | 91% | 91% | 78% | 78% | 35% |
| Qwen 3.5 4B | 50% | 99% | 90% | 34% | 16% | 36% |
| Qwen 3.5 9B | 72% | 100% | 91% | 42% | 31% | 34% |
| Qwen 3.6 27B | 95% | 100% | 98% | 44% | 42% | 42% |
| Gemma 4 E4B | 93% | 100% | 83% | 20% | 17% | 13% |
| Gemma 4 31B | 0% | 100% | 100% | 86% | 0% | 45% |
| Qwen 3.7 Flash | 98% | 99% | 98% | 21% | 20% | 41% |

`End-to-end ok` é a taxa que passou a precedência completa de formato, schema
e semântica. Não é uma accuracy diagnóstica. O caso extremo Gemma 31B mostra
isso: zero `ok` devido às 100 JSON fences, mas 45% Clinical Top-1 depois da
recuperação determinística e 86% semantic compliance.

### 5.4 Intervenções do parser por modelo

| Modelo | Confusion: JSON fences | Evidence: JSON fences | Unrecoverable JSON |
| --- | ---: | ---: | ---: |
| Luna | 0 | 0 | 0; as recusas não são parseadas |
| Qwen 3.5 4B | 0 | 49 | 1 Evidence |
| Qwen 3.5 9B | 33 | 28 | 0 |
| Qwen 3.6 27B | 6 | 5 | 0 |
| Gemma 4 E4B | 8 | 7 | 0 |
| Gemma 4 31B | 200 | 100 | 0 |
| Qwen 3.7 Flash | 0 | 1 | 0; existe uma falha de backend separada |

Não foi aplicado parser a Open-ended, porque a resposta pretendida é texto
clínico natural.

## 6. Open-ended judge: resultados, coverage e limitações

O judge principal foi Luna com reasoning high. Qwen 3.7 Flash foi chamado
apenas quando Luna devolveu `content_policy_violation`. Foi efetuada uma única
passagem de retry dirigida aos `judge_invalid`; judgments válidos não foram
repetidos.

| Modelo avaliado | Judge Top-1 | Judge Top-3 | Top-1/3 sobre 100 | Coverage | Invalid judge | Model failures | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 23.0% | 41.0% | 23 / 41% | 100% | 0 | 6 | 0 |
| Qwen 3.5 4B | 18.1% | 35.1% | 17 / 33% | 94% | 6 | 0 | 6 |
| Qwen 3.5 9B | 24.0% | 41.7% | 23 / 40% | 96% | 4 | 0 | 6 |
| Qwen 3.6 27B | 25.8% | 43.3% | 25 / 42% | 97% | 3 | 0 | 6 |
| Gemma 4 E4B | 13.0% | 31.0% | 13 / 31% | 100% | 0 | 0 | 6 |
| Gemma 4 31B | 29.3% | 47.5% | 29 / 47% | 99% | 1 | 0 | 6 |
| Qwen 3.7 Flash | **35.8%** | **53.7%** | **34 / 51%** | 95% | 5 | 1 | 6 |

O segundo par de valores usa sempre 100 como denominador e é a comparação
mais conservadora quando a judge coverage difere.

### 6.1 Scores do judge

| Modelo | Diagnosis | Findings | Grounding | Clinical rationale | Differential | Unsupported claim rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna | 1.44 | **3.07** | **3.30** | **2.52** | **2.57** | **27.0%** |
| Qwen 3.5 4B | 1.19 | 2.17 | 1.72 | 1.53 | 1.45 | 97.9% |
| Qwen 3.5 9B | 1.46 | 2.40 | 2.05 | 1.80 | 1.88 | 87.5% |
| Qwen 3.6 27B | 1.48 | 2.37 | 2.20 | 1.93 | 2.08 | 87.6% |
| Gemma 4 E4B | 1.01 | 2.47 | 2.41 | 1.68 | 1.63 | 74.0% |
| Gemma 4 31B | 1.65 | 2.78 | 2.57 | 2.20 | 2.28 | 74.7% |
| Qwen 3.7 Flash | **1.92** | 2.55 | 2.46 | 2.17 | 2.32 | 78.9% |

### 6.2 Self-judging condicional do Qwen 3.7 Flash

Em seis casos, Luna recusou a imagem como judge e o fallback foi o mesmo Qwen
3.7 Flash que estava a ser avaliado. Este self-judging não foi escondido:

- Luna julgou com sucesso 88 outputs: Top-1 35.2%, Top-3 54.5%;
- Qwen julgou seis outputs próprios: Top-1 50%, Top-3 50%;
- um caso foi model failure;
- cinco judgments permaneceram inválidos após uma passagem de retry.

O resultado agregado de 35.8/53.7% pode ter ligeiro viés favorável devido aos
seis casos de self-judging. A leitura Luna-only e o resultado conservador
34/51% sobre os 100 casos devem acompanhar qualquer comparação. Não se deve
selecionar o teacher apenas pelo resultado open-ended agregado.

## 7. Safety policy, erros e outputs ausentes

### 7.1 Luna como modelo avaliado

Luna devolveu `content_policy_violation` ao processar imagens dermatológicas:

| Task | Safety refusals | Requests |
| --- | ---: | ---: |
| Visual Top-K | 5 | 100 |
| Confusion Sets | 2 | 200 |
| Evidence-grounded diagnosis | 9 | 100 |
| Open-ended diagnosis | 6 | 100 |
| Total request-level | 22 | 500 |

Este total representa eventos por request, não 22 imagens necessariamente
distintas, porque algumas imagens reaparecem entre tasks. As recusas contam
como falha/ausência de prediction; não foram preenchidas por outro modelo nas
métricas determinísticas.

### 7.2 Luna como judge

O judge primário acionou content policy em seis casos para cada modelo local/API
não-Luna. O fallback foi usado apenas nesses casos. O judge safety refusal
final foi zero porque os casos chegaram ao fallback; outputs inválidos do
fallback permanecem `judge_invalid`.

### 7.3 Qwen 3.7 Flash

Não existiu safety refusal do Qwen como modelo avaliado. Existiram duas falhas
OpenRouter classificadas como backend/transport:

- um request de Evidence retornou HTTP 400 com mensagem genérica
  `Provider returned error`;
- um request Open-ended falhou da mesma forma.

Não há evidência suficiente para reclassificar estes erros como policy. Foram
mantidos como backend errors. Não foi repetida toda a execução para substituir
casos falhados, preservando o run observado.

### 7.4 Outros modelos e truncation

Os cinco modelos locais não tiveram backend errors nem safety refusals. Nenhum
dos sete modelos registou truncated output nas quatro tasks.

## 8. Demografia e subgroup accuracy

Os reference Parquet contêm tom de pele, idade, sexo/género e raça/etnia, mas
a cobertura é desigual. No cohort de 100:

| Task | Tom de pele | Idade agrupada | Sexo/género | Raça/etnia |
| --- | ---: | ---: | ---: | ---: |
| Visual Top-K | 95% | 28% | 46% | 21% |
| Confusion Sets | 94% | 37% | 48% | 22% |
| Evidence | 99% | 0% | 0% | 0% |
| Open-ended | 98% | 8% | 9% | 0% |

Visual Top-K já produz `by_skin_tone`, bands agregadas, Wilson confidence
intervals, worst-group accuracy e accuracy gap. Contudo, nenhum grupo exato
atinge neste cohort o mínimo pré-especificado de 30 leakage groups. Portanto,
não é correto interpretar estes 100 casos como uma fairness evaluation
conclusiva.

Além disso, a implementação atual de subgroup Top-K usa strict-schema
predictions em vez de canonical predictions. Isto pode penalizar modelos pela
formatação, em particular Gemma 31B. Deve ser corrigido antes da análise de
fairness no Validation completo ou no Internal Benchmark.

Fitzpatrick e Monk não devem ser fundidos como se fossem a mesma escala. Idade,
sexo/género e raça/etnia ficam exploratórios devido a missingness e possível
confounding por source e doença.

## 9. Limitações metodológicas completas

1. **Validation, não test final.** Estes dados foram usados para desenvolvimento
   e teacher screening; repetir decisões sobre eles aumenta risco de overfitting.
2. **Subsets pequenos e balanceados.** Cem casos dão sinal comparativo, mas não
   estimam prevalência clínica nem estabilizam diferenças pequenas.
3. **Dependência entre tasks.** As mesmas imagens podem aparecer em várias
   tasks; as 500 solicitações por modelo não são 500 casos independentes.
4. **Confusion emparelhado.** Existem 100 imagens, não 200 imagens únicas.
5. **Labels de origem.** O ground truth herda limitações, granularidade e
   possível ruído dos datasets originais.
6. **Image-only.** Não existe história clínica, sintomas, palpação, evolução,
   exames ou histopatologia; isto limita diagnósticos que dependem de contexto.
7. **Judge único.** Open-ended depende de uma única família de judge, prompt e
   rubrica. Não existe second judge nem revisão humana por decisão do protocolo.
8. **Fallback diferente do judge principal.** Seis casos usam Qwen em vez de
   Luna; para o Qwen avaliado isto produz self-judging condicional.
9. **Judge coverage desigual.** Accuracies condicionais têm denominadores
   diferentes; por isso também é reportado o resultado sobre 100.
10. **Safety policy provider-specific.** As recusas Luna medem simultaneamente
    capacidade e disponibilidade prática do provider.
11. **OpenRouter é provider-managed.** A rota e implementação subjacente podem
    mudar; não existe um immutable weight revision equivalente ao modelo local.
12. **Perfis de geração diferentes.** Temperature e sampling seguem perfis por
    família; os resultados não isolam apenas o efeito da arquitetura.
13. **Prompt-only structured output.** JSON validity mede também instruction
    following. Structured decoding poderia melhorar formato, mas seria outra
    condição experimental.
14. **Parser estreito mas necessário.** Canonical metrics medem conteúdo
    recuperável; strict metrics representam o custo real de integração sem
    pós-processamento.
15. **Evidence references incompletas.** Conceitos morfológicos são controlados
    e avaliados lexicalmente; F1 não equivale a uma revisão clínica humana.
16. **Fairness ainda não concluída.** O cohort de 100 não suporta conclusões
    robustas por tom de pele, idade, sexo/género ou raça/etnia.
17. **Sem confidence intervals globais nesta tabela.** Diferenças de poucos
    pontos percentuais não devem ser tratadas automaticamente como ranking
    estatisticamente estável.
18. **Thinking comparison incompleta.** Luna high e os restantes off não
    permitem concluir ainda se thinking melhora cada modelo.

## 10. Artefactos e reprodutibilidade

Raízes dos resultados:

- `outputs/validation_screening_v1/thinking_off/`;
- `outputs/validation_screening_v1/luna_high/`.

Run Qwen 3.7 Flash:

- Visual Top-K: `20260802T031406Z_84f40f19`;
- Confusion Sets: `20260802T031406Z_521d694e`;
- Evidence: `20260802T031406Z_d9214a4e`;
- Open-ended: `20260802T031406Z_c127fd3c`.

Cada run contém:

- `predictions.jsonl`, incluindo raw final text, parser audit, reasoning
  separado e usage;
- `metrics.json`;
- `report.html`;
- `run_manifest.yaml` e snapshots de configuração.

Open-ended acrescenta `judgments.jsonl`, `judge_metrics.json`,
`judge_manifest.yaml` e `judge_report.html`.

Auditoria final da fase:

- 28 `run_manifest.yaml`: sete modelos por quatro tasks;
- 28 `metrics.json`;
- sete `judge_metrics.json`;
- um único selection hash por benchmark entre os sete modelos;
- `168 passed`, `1 skipped`, quatro warnings conhecidos e nove subtests;
- `git diff --check` sem erros.

## 11. Decisão provisória e próximo gate

Nenhum teacher é congelado nesta fase. Os candidatos que justificam maior
atenção no A/B thinking-on são:

- Gemma 4 31B, melhor equilíbrio local e melhor Evidence;
- Qwen 3.6 27B, melhor Visual Top-1;
- Qwen 3.7 Flash, melhor Open-ended e melhor morphology F1, sujeito às
  limitações de provider e self-judging;
- Qwen 3.5 9B, candidato menor com morphology/description fortes.

O próximo gate é repetir os mesmos IDs com thinking enabled nos modelos
aplicáveis, preservar Luna high como referência e decidir se os líderes estão
suficientemente separados. Só se a conclusão permanecer instável se expande
Visual/Confusion para 200 e Evidence para os 137 casos completos.

A fase thinking-on usa atualmente um override de desenvolvimento
`max_output_tokens = 14336`. Depois dos smokes iniciais, foi acrescentado
`reasoning_max_tokens = 10240` a todos os modelos locais/OpenRouter com
thinking controlável. O primeiro valor é o limite total; o segundo limita
apenas o reasoning e deixa até 4096 tokens para a resposta final. No vLLM, o
campo é traduzido para
`thinking_token_budget`; no OpenRouter, para `reasoning.max_tokens`. Os
artefactos congelados permanecem em 8192. Como a fase thinking-off teve zero
truncated outputs, o limite anterior não foi vinculativo e esses runs não são
repetidos.

Antes da execução, foram concluídos 12 dry-runs: quatro tasks para Qwen 3.6
27B, Qwen 3.7 Flash e Gemma 4 E4B. Todos confirmaram os mesmos selection hashes
da fase anterior, `reasoning_enabled: true`, `generation_mode: enabled`, zero
chamadas de rede/modelo e `max_output_tokens: 10240`. Esses dry-runs ocorreram
antes da revisão final do protocolo para `14336/10240` e devem ser repetidos
estruturalmente antes da execução de 100 casos. Nenhum modelo controlável envia
um `reasoning_effort` discreto; Luna permanece a exceção com effort high.
