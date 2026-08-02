# Registo de decisão das prompts open-ended e da judge

Data: 2026-08-02

## 1. Objetivo desta nota

Esta nota reconstrói exatamente como foram decididas:

1. a prompt enviada ao modelo avaliado no benchmark
   `open_ended_diagnosis`;
2. a prompt usada pelo LLM-as-a-judge;
3. a forma de step reasoning introduzida, testada e rejeitada ou mantida;
4. as métricas, validações e estudos que sustentaram as decisões.

É importante separar as duas prompts:

```text
imagem + model prompt
  -> resposta clínica natural do modelo avaliado

imagem + referência + resposta clínica + judge prompt
  -> judgment JSON
```

A prompt do modelo determina o comportamento que está a ser medido. A judge
prompt não altera essa resposta; apenas a avalia depois de estar congelada.

## 2. Decisão final congelada

O protocolo final do ISEPDermaBench 1.5.0 usa:

| Componente | Versão final | Estado |
| --- | --- | --- |
| Prompt do modelo avaliado | 1.1.0 | congelada |
| Judge prompt | 1.2.0 | congelada |
| Judge schema | Draft 2020-12, estrito | congelado |
| Primary judge | `gpt_5_6_luna`, reasoning high | congelado |
| Fallback | Qwen 3.7 Flash apenas em content-policy violation | condicional |

Artefactos finais:

- `src/benchmark/resources/open_ended_diagnosis/model_prompt.yaml`;
- `src/benchmark/resources/open_ended_diagnosis/judge_prompt.yaml`;
- `src/benchmark/resources/open_ended_diagnosis/judge.schema.json`;
- cópias imutáveis em `data/benchmarks/ISEPDermaBench/artifacts/`.

O SHA-256 da judge prompt publicada é
`0662acfaa0b0eb1c4c7a68022c75be699d2193473662ef7a836ec2839c1a5a57`.
O hash é guardado em cada `judge_manifest.yaml`, juntamente com o hash do
schema, configuração do judge, predictions avaliadas, split e batch size.

## 3. Como a judge prompt foi desenvolvida

### 3.1 Judge 1.0.0 — primeira versão funcional

A primeira versão definia um único judge blinded, mandava inspecionar a imagem
e fornecia uma escala genérica de 0 a 4:

```text
You are the single blinded clinical judge for a dermatology benchmark.
Evaluate only the assistant response supplied for this case.
The reference diagnosis is correct for benchmark scoring. Inspect the image
yourself.
```

Incluía uma orientação geral para o ranking:

```text
Determine directly whether the reference diagnosis is ranked first, second,
third, or absent. Do not infer a hidden ranking when the response does not
make the order reasonably clear.
```

E a rubrica inicial:

```text
0 = absent or completely incorrect;
1 = mostly incorrect;
2 = partially correct, generic, or materially incomplete;
3 = mostly correct with minor limitations;
4 = correct, specific, coherent, and well supported.
```

Esta versão era suficiente para gerar o schema, mas deixava decisões
importantes implícitas:

- a relação exata entre rank, diagnosis score e verdict;
- o que fazer com um diagnóstico correto num diferencial sem ordem clara;
- quando um subtype ou synonym era equivalente;
- se a referência diagnóstica também validava automaticamente a morfologia;
- como penalizar diferentes gravidades de unsupported claims;
- como distinguir findings corretos de uma conclusão correta por acaso.

### 3.2 Judge 1.1.0 — ranking literal e self-check

Durante o desenvolvimento inicial foi identificado o risco de judgments
semanticamente contraditórios, por exemplo `rank = 0` acompanhado por
`diagnosis_correctness = 4` ou verdict `correct`. A versão 1.1 tornou o mapping
literal:

```text
- rank 1: the reference diagnosis is the assistant's primary diagnosis;
- rank 2: it is explicitly the second diagnosis;
- rank 3: it is explicitly the third diagnosis;
- rank 0: it is absent from the explicit top three or the response is unranked.
```

Também introduziu invariantes diretamente na prompt:

```text
Never combine rank 0 with a correctness score of 4 or an overall verdict of
correct. Never combine rank 1 with a diagnosis correctness score below 3.
```

E acrescentou um passo de verificação antes do JSON:

```text
Before returning JSON, verify that the rank, diagnosis correctness, verdict,
and judge summary agree with one another.
```

No user template foi introduzida uma sequência explícita:

```text
First locate the reference diagnosis or an accepted synonym in the
assistant's explicit ranking; then set reference_diagnosis_rank from that
position.
```

Este foi o primeiro step reasoning da judge: localizar primeiro a referência
na ordenação observável e só depois atribuir rank e scores. Não se pediu que o
judge revelasse chain-of-thought; pediu-se apenas uma ordem operacional antes
do output estruturado.

### 3.3 Judge 1.2.0 — rubrica clínica completa

A versão 1.2.0 substituiu regras implícitas por um protocolo explícito. As
principais alterações foram as seguintes.

#### Identidade do modelo escondida

```text
You are a model-identity-blinded clinical benchmark evaluator with expertise
in dermatology.
```

O request inclui apenas a resposta final visível. O nome do modelo, backend e
reasoning devolvido pelo provider não são enviados ao judge. Isto reduz
self-enhancement e preferências associadas à identidade, embora não elimine
todos os vieses de estilo.

#### Separação entre label e evidência visual

```text
The reference diagnosis is the benchmark ground truth for diagnosis matching.
Do not overturn it based on your interpretation of the image. Independently
inspect the supplied image to evaluate the assistant's visual findings,
clinical rationale, and factual support.
```

Esta regra evita dois erros opostos:

1. o judge substituir o label congelado pela sua própria impressão da imagem;
2. o judge assumir que qualquer finding referido pelo modelo é verdadeiro só
   porque o diagnóstico coincide com o label.

#### Hierarquia das fontes de evidência

```text
A patient-specific fact is supported only when it is:
1. visible in the supplied image; or
2. explicitly supplied as case context.
```

```text
The reference diagnosis supports diagnosis matching only. It does not prove
that a particular morphology, symptom, history, investigation, pathology
result, duration, or treatment response is present.
```

SKINCON e SkinCAP permanecem auxiliares. SKINCON fornece conceitos positivos
incompletos; SkinCAP pode ser diagnosis-conditioned ou conter elementos não
visuais. O judge recebe ambos quando existem, mas é instruído a inspecionar a
imagem e a não aceitar claims apenas por aparecerem numa descrição auxiliar.

#### Ranking antes dos scores

```text
Determine the position of the reference diagnosis in the assistant's explicit
diagnostic ordering.
```

O mapping congelado é:

| Rank da referência | `diagnosis_correctness` |
| --- | ---: |
| 1, equivalente e claro | 4; 3 apenas por imprecisão não material |
| 2 | 3 |
| 3 | 2 |
| 0, mas presente num diferencial atual não ordenado | até 2 |
| Ausente, negado ou não equivalente | 0 ou 1 |

Top-1, Top-3 e MRR são derivados de `reference_diagnosis_rank`. Não são
inferidos a partir do score subjetivo de diagnosis correctness.

#### Equivalência diagnóstica

```text
Accept standard synonyms, clinically interchangeable names, unambiguous
abbreviations, minor spelling errors with certain intent, and a compatible
specific subtype that is unambiguously an instance of the reference diagnosis.
```

Doenças relacionadas, manifestações, histologia, causas suspeitas ou famílias
demasiado amplas não são consideradas equivalentes. Synonyms repetidos ocupam
uma única posição.

#### Métricas independentes

```text
Score every 0-4 field independently.
```

As cinco dimensões foram definidas separadamente:

- `diagnosis_correctness`: correspondência entre ranking e label;
- `visual_findings_correctness`: correção e completude dos claims visuais;
- `evidence_grounding`: suporte de factos específicos na imagem ou contexto;
- `clinical_rationale_quality`: ligação clínica entre findings e ranking;
- `differential_quality`: unicidade, plausibilidade, ordem e discriminação.

Isto impede que um diagnóstico correto infle automaticamente morphology,
grounding ou rationale.

#### Unsupported claims proporcionais à gravidade

```text
A minor, non-consequential unsupported claim lowers the affected dimension by
one point.
```

```text
Invented pathology, investigation results, history, duration, treatment
response, or other central evidence caps the affected dimension at 1.
```

O rank diagnóstico não muda por causa de um claim inventado. A penalização é
aplicada às dimensões afetadas e ao verdict. Assim, accuracy de diagnóstico e
factual grounding permanecem métricas distintas.

#### Verdicts mantidos

A proposta intermédia considerava apenas `correct`, `partially_correct` e
`incorrect`. Foi decidido manter quatro níveis:

- `correct`;
- `mostly_correct`;
- `partially_correct`;
- `incorrect`.

Isto permite distinguir um Top-1 correto e bem fundamentado de um Top-1 com
imprecisão menor. Os verdicts não substituem as métricas quantitativas; são um
resumo clínico ordinal.

#### Consistency check final

```text
Before responding, verify:
- rank 0 is never paired with diagnosis_correctness above 2, correct, or
  mostly_correct;
- rank 2 has diagnosis_correctness 3;
- rank 3 has diagnosis_correctness 2;
- correct requires rank 1 and diagnosis_correctness 4.
```

A saída final continua a ser apenas o JSON, sem reasoning exposto.

## 4. Enforcement fora da prompt

A consistência não depende apenas de o LLM obedecer às instruções.

### 4.1 JSON Schema estrito

O schema usa:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "reference_diagnosis_rank",
    "diagnosis_correctness",
    "visual_findings_correctness",
    "evidence_grounding",
    "clinical_rationale_quality",
    "differential_quality",
    "unsupported_claim_count",
    "unsupported_claim_examples",
    "overall_verdict",
    "judge_summary"
  ]
}
```

O schema valida tipos e intervalos, mas não consegue expressar sozinho todas
as relações entre campos.

### 4.2 Validador semântico determinístico

`_validate_judgment_semantics()` rejeita, entre outras situações:

- rank 0 com diagnosis score superior a 2;
- rank 0 com verdict `correct` ou `mostly_correct`;
- rank 1 com diagnosis score inferior a 3;
- rank 2 com score diferente de 3;
- rank 3 com score diferente de 2;
- verdict `correct` sem rank 1 e score 4;
- contagem de unsupported claims inconsistente com os exemplos.

### 4.3 Retry corretivo

Existem até três tentativas. Quando JSON, schema ou semântica falham, o segundo
request inclui o erro e pede um judgment novo sobre a resposta original:

```text
Your previous JSON judgment was invalid: [erro]
Re-evaluate the original assistant response and return a fresh, internally
consistent JSON object.
```

Não se faz patch manual aos scores do judge. Se as três tentativas falharem, o
caso fica `judge_invalid` e é reportado separadamente.

### 4.4 Blinding implementado em teste

Os testes confirmam que o request inclui a resposta final visível, mas não
inclui:

- `model_id` do modelo avaliado;
- reasoning privado ou summary do provider;
- respostas de outros modelos.

O judge recebe exatamente uma resposta por caso. Não existe pairwise ranking,
votação nem segundo judge.

## 5. Como a judge 1.2.0 foi aceite

### 5.1 O que não foi feito

Não existiu um A/B clínico direto `judge 1.1 versus judge 1.2` com judgments
humanos como gold standard. Sem essa referência, comparar médias entre judges
não demonstraria qual judge estava clinicamente correto.

A judge 1.2.0 foi escolhida porque:

1. removeu ambiguidades formais da 1.0/1.1;
2. definiu uma hierarquia de evidência compatível com o benchmark image-only;
3. separou dimensões clínicas que não devem ser colapsadas;
4. produziu JSON e judgments semanticamente válidos sob validação automática;
5. manteve cobertura completa nos calibration runs;
6. permaneceu fixa em todos os A/B das prompts do modelo.

### 5.2 Evidência de calibração

Durante o desenvolvimento, a string de versão `1.2.0` foi usada numa candidata
antes do freeze. Os manifests permitem distinguir as duas revisões pelo hash:

- A/B inicial de dez casos: candidata 1.2.0 com SHA-256
  `ebd3b442ed1f72d44c6d73db1f76ec72cf3e961e4176e3ec3187c862dd46d20a`;
- acceptance run e A/B finais: judge 1.2.0 congelada com SHA-256
  `0662acfaa0b0eb1c4c7a68022c75be699d2193473662ef7a836ec2839c1a5a57`.

Isto aconteceu ainda durante Validation e antes do freeze. Depois da release
1.5.0, o hash, e não apenas a string de versão, passou a fazer parte da
identidade imutável do protocolo. A revisão preliminar não deve ser confundida
com a judge publicada.

| Execução | Casos | Resultado do modelo | Judge terminal | Retries observados |
| --- | ---: | ---: | ---: | ---: |
| Prompt A, exemplo; judge candidata | 10 | 10 válidos | 10/10, 0 invalid | 10 na primeira tentativa |
| Prompt B, sem exemplo; judge candidata | 10 | 10 válidos | 10/10, 0 invalid | 10 na primeira tentativa |
| Acceptance 1.2.0 | 20 | 20 válidos | 20/20, 0 invalid | 19 na primeira; 1 na terceira |
| A/B final, prompt 1.1 | 50 | 47 válidos, 3 safety failures | 50/50 terminal | 46 na primeira; 1 na terceira |
| A/B final, prompt 1.2.1 | 50 | 47 válidos, 3 safety failures | 50/50 terminal | 47 na primeira |
| Candidata 1.3 | 50 | 45 válidos, 5 failures | 50/50 terminal | 41 na primeira; 4 com retry |

`Terminal` inclui o judgment determinístico atribuído a um model failure. Entre
as respostas efetivamente julgadas não ficou nenhum `judge_invalid`. Os retries
mostram que o validador não era redundante: alguns JSON formalmente produzidos
precisaram de nova avaliação antes de serem aceites.

Os manifests e judgments encontram-se em:

- `outputs/prompt_ab/`;
- `outputs/prompt_freeze/`;
- `outputs/prompt_ab_final/`;
- `outputs/prompt_ab_dermatologist_vision/`.

## 6. Evolução da prompt do modelo avaliado

### 6.1 Prompt 1.0.0 — exemplo de estilo

A primeira prompt pedia findings, Top-3 e racionales, mas incluía um exemplo:

```text
The image shows [relevant morphology, colour, distribution, and other visible
features]. [First diagnosis] is the most likely diagnosis because [...].
[Second diagnosis] ranks second [...]. [Third diagnosis] ranks third [...].
```

Observação no dry run de dez casos: o modelo imitava a formulação do exemplo.

| Expressão do exemplo | Frequência com exemplo | Sem exemplo |
| --- | ---: | ---: |
| `The image shows` | 9/10 | 0/10 |
| `ranks second` | 3/10 | 0/10 |
| `ranks third` | 3/10 | 0/10 |

Por isso, o exemplo foi considerado um confounder de estilo.

### 6.2 Prompt 1.1.0 — sequência clínica curta

A versão 1.1 removeu o exemplo e manteve apenas uma sequência observável:

```text
Begin by describing the relevant visible findings. Then provide exactly three
unique diagnoses in an explicitly ordered differential.
```

```text
Briefly connect each diagnosis to the visible findings that support or weaken
it.
```

Isto é step reasoning ao nível do **output clínico**:

```text
visible findings -> Top-3 ordenado -> rationale baseada nos findings
```

Não é um pedido para revelar chain-of-thought privado. A system prompt diz
explicitamente:

```text
Do not reveal private chain-of-thought. Return only a concise clinical
assessment and its supporting visible evidence.
```

No A/B emparelhado de dez casos, remover o exemplo:

- manteve Top-3 em 60%;
- reduziu unsupported-claim rate de 60% para 30%;
- aumentou evidence grounding de 3,2 para 3,5;
- reduziu Top-1 de 40% para 20% e rationale de 2,9 para 2,5.

A amostra era pequena; a 1.1 avançou como candidata por remover style
anchoring, não por provar maior accuracy.

### 6.3 Prompt 1.2.0 — findings e ranking rigidamente estruturados

A candidata seguinte tornou a sequência mais prescritiva:

```text
Begin with the heading "Visible findings:" and describe only diagnostically
relevant features that can be assessed reliably.
```

```text
Most likely diagnosis: [one diagnosis] — [brief visible-evidence rationale]
Second most likely diagnosis: [one diagnosis] — [brief visible-evidence rationale]
Third most likely diagnosis: [one diagnosis] — [brief visible-evidence rationale]
```

Também distinguiu `not visible` de `clinically absent`, proibiu palpation-only
claims e restringiu dermoscopic terminology a imagens dermoscópicas.

No acceptance run de 20 casos:

- 20/20 respostas válidas;
- 0 truncamentos e 0 refusals;
- judge coverage 100%;
- evidence grounding 3,70/4;
- unsupported-claim rate 15%.

Contudo, `Image limitations` apareceu em 19/20 respostas. A prompt tinha
transformado uma salvaguarda útil num padrão quase obrigatório.

### 6.4 Prompt 1.2.1 — correção de Image limitations

A regra foi restringida:

```text
Add one final sentence beginning with "Image limitations:" only when a
case-specific problem with image quality or framing prevents assessment of a
diagnostically relevant visible feature.
```

```text
Do not add it for routine limitations inherent to photographs, such as
inability to palpate the lesion.
```

No A/B final de 50 casos, exatamente os mesmos task IDs receberam a prompt
1.1.0 ou 1.2.1. Modelo, seed, reasoning effort e judge 1.2.0 foram mantidos.

| Métrica | 1.1.0 | 1.2.1 |
| --- | ---: | ---: |
| Top-1 | 32% | 26% |
| Top-3 | 44% | 44% |
| MRR | 0,370 | 0,337 |
| Visual findings | 3,02 | 3,08 |
| Evidence grounding | 3,42 | 3,46 |
| Clinical rationale | 2,68 | 2,48 |
| Differential quality | 2,66 | 2,50 |
| Unsupported-claim rate | 16% | 16% |

A correção de limitations funcionou: apenas 4/47 respostas válidas usaram a
frase. Porém, a estrutura adicional não melhorou Top-3 e piorou Top-1,
rationale e differential quality. A prompt 1.1.0 foi congelada.

### 6.5 Prompt 1.3.0 — dois passes internos inspirados em dermatologia

A candidata 1.3 introduziu explicitamente dois passes silenciosos:

```text
Inspect the case silently in two complementary passes. First form a global
impression of the dominant cutaneous pattern [...]. Then verify that impression
against the most representative intact primary lesion [...].
```

E uma verificação de mimic:

```text
Before finalizing, compare the leading impression with its strongest plausible
visual mimic and check which visible finding best supports or weakens each.
```

O desenho tentou aproximar duas componentes da leitura dermatológica:

1. reconhecimento global do padrão;
2. verificação analítica de morfologia, configuração, distribuição e superfície.

No mesmo cohort de 50 casos:

| Métrica | 1.1.0 | 1.3 candidata |
| --- | ---: | ---: |
| Respostas válidas | 47 | 45 |
| Truncamentos | 0 | 2 |
| Top-1 | 32% | 28% |
| Top-3 | 44% | 46% |
| Visual findings | 3,02 | 2,78 |
| Evidence grounding | 3,42 | 3,20 |
| Clinical rationale | 2,68 | 2,56 |
| Unsupported-claim rate | 16% | 18% |
| Tokens totais médios | 1.562 | 3.223 |

O reasoning médio aumentou de 862 para 2.178 tokens. Nos dois truncamentos, o
reasoning consumiu os 8.192 tokens disponíveis. A melhoria de dois pontos em
Top-3 não compensou menor Top-1, grounding, findings e eficiência. A candidata
foi rejeitada.

## 7. Três significados diferentes de step reasoning

Para evitar ambiguidade, o projeto distingue:

### A. Sequência observável de resposta — mantida

```text
findings visíveis -> Top-3 explícito -> rationale clínica curta
```

É o comportamento da prompt 1.1.0. O conteúdo é visível, avaliável e não inclui
um monólogo interno.

### B. Dois passes internos numa só chamada — rejeitados

```text
gestalt interno -> verificação interna -> resposta final
```

Foi a candidata 1.3.0. Aumentou compute e truncamentos sem benefício agregado.

### C. Duas chamadas externas para dados sintéticos — futura ablation

```text
imagem -> morphology JSON
imagem + morphology JSON -> diagnóstico, evidência e ação
```

Esta opção é diferente da candidata 1.3: cada etapa tem um output verificável.
Está proposta apenas para teacher-data generation, não para alterar a prompt
congelada do benchmark.

## 8. Estudos que orientaram as decisões

### 8.1 Design do LLM-as-a-judge

**G-Eval — Liu et al., 2023.** Propôs critérios explícitos e um paradigma de
form filling para avaliação com LLMs, mas também alertou para viés a favor de
texto gerado por LLM. Aplicação no projeto: dimensões e schema explícitos;
limitação reconhecida: judge scores não são gold clínico humano.

<https://arxiv.org/abs/2303.16634>

**Judging LLM-as-a-Judge with MT-Bench — Zheng et al., 2023.** Identificou
position, verbosity e self-enhancement biases. Aplicação no projeto: uma única
resposta por request, identidade do modelo escondida, sem comparação pairwise,
judge fixo em todos os modelos e response length reportada.

<https://arxiv.org/abs/2306.05685>

**DermoGPT — Ru et al., 2026, preprint.** O benchmark separa Morphology,
Diagnosis, Reasoning e Fairness, usa judge prompts específicos e reporta human
sanity checks. Aplicação: separar scores clínicos em vez de usar um verdict
único.

<https://arxiv.org/abs/2601.01868>

### 8.2 Perceção dermatológica usada nas prompts candidatas

**Ko et al., 2019.** Descreve princípios de perceção, cognição e erro no
diagnóstico dermatológico. Motivou separar observação visual de conclusão.

<https://pubmed.ncbi.nlm.nih.gov/30797839/>

**Tourassi et al., 2022.** Comparou eye movements e narrativas de participantes
com diferentes níveis de expertise. Motivou pesquisa visual organizada, sem
assumir que uma checklist reproduz expertise.

<https://pubmed.ncbi.nlm.nih.gov/36046501/>

**Rimoin et al., 2015.** Estudou treino de morphology, configuration e
distribution. Motivou estas dimensões na candidata 1.3 e no futuro target de
morphology grounding.

<https://doi.org/10.1016/j.jaad.2014.11.016>

**Gachon et al., 2005.** Estudou o papel do padrão global no reconhecimento de
melanoma. Motivou o passo de global impression seguido de verificação.

<https://pubmed.ncbi.nlm.nih.gov/15837860/>

**Jain et al., 2024.** Avaliou diagnóstico diferencial em imagens clínicas,
incluindo diferenças por tom de pele e limitações do cenário com uma fotografia
isolada. Motivou Top-3, análise por skin tone e linguagem prudente sobre cor.

<https://www.nature.com/articles/s41591-023-02728-3>

### 8.3 Reasoning e prompting clínico

**DermPrompt — Vashisht et al., 2024.** Usou retrieval e re-ranking com
GPT-4V; naive CoT ajudou retrieval e guideline-grounded reasoning foi usado no
diagnóstico. Motivou a separação entre gerar candidatos e discriminá-los por
evidência, mas não justificou multi-agent reasoning como baseline do projeto.

<https://arxiv.org/abs/2404.17749>

**Structured clinical reasoning prompt — 2024.** Em 322 casos de radiologia,
organizar primeiro história e imaging findings e diagnosticar depois melhorou
primary accuracy de 56,5% para 60,6% e Top-3 de 66,5% para 70,5%. Motivou testar
etapas separadas, com a ressalva de não ser dermatologia image-only.

<https://pubmed.ncbi.nlm.nih.gov/39625594/>

**Diagnostic reasoning prompts — Savage et al., 2024.** CoT tradicional
melhorou GPT-3.5 face a non-CoT, mas prompts analíticas ou de differential
diagnosis foram piores; uma estratégia isolada funcionou melhor que combinar
várias. Isto ajuda a explicar porque a candidata 1.3, mais complexa, não foi
automaticamente superior.

<https://pmc.ncbi.nlm.nih.gov/articles/PMC10808088/>

**Zero-shot CoT em nonmelanoma skin cancer — O'Hagan et al., 2023.** A prompt
“Let's think step by step” não produziu ganho significativo: 4,92/5 sem CoT e
4,87/5 com CoT. Motivou não tratar CoT genérico como melhoria garantida.

<https://pmc.ncbi.nlm.nih.gov/articles/PMC10755659/>

**Two-stage verification — Shao e Zhang, 2025.** Diagnóstico inicial seguido de
verificação predefinida produziu ganhos máximos de 4,0 pontos sobre CoT, mas os
efeitos dependeram do modelo e dataset e menor incerteza significou por vezes
confiança errada. Motivou tornar verificação uma ablation controlada.

<https://www.nature.com/articles/s41746-025-02146-4>

**VL-MedGuide — Yu et al., 2025, preprint.** Separa concept perception de
explainable disease reasoning no Derm7pt. É a referência mais próxima das duas
chamadas propostas para dados sintéticos, mas não valida a mesma estratégia em
fotografias clínicas heterogéneas.

<https://arxiv.org/abs/2508.06624>

**Concept inconsistency in Derm7pt — Nápoles et al., 2026, preprint.** Mostra
que perfis de concepts iguais podem corresponder a labels diferentes. Por isso,
a etapa diagnóstica futura deve receber findings e imagem original, não apenas
um concept bottleneck rígido.

<https://arxiv.org/abs/2604.19323>

## 9. Observações e conclusão metodológica

1. Uma prompt pode melhorar formato e grounding sem melhorar diagnóstico.
2. Um exemplo few-shot pode criar style anchoring e afetar a imparcialidade do
   judge perante modelos com estilos diferentes.
3. Mais instruções clínicas aumentaram compliance, mas também verbosity,
   `Image limitations` mecânico e, na 1.3, overthinking.
4. A sequência curta `findings -> ranking -> rationale` foi mais robusta que
   uma simulação extensa do processo cognitivo de um dermatologista.
5. A judge prompt deve ser avaliada por consistência, cobertura, blinding e
   estabilidade; sem judgments humanos, não se deve afirmar que é clinicamente
   perfeita.
6. O validador determinístico e o retry corretivo são parte do protocolo da
   judge, não detalhes opcionais de parsing.
7. A judge 1.2.0 e a model prompt 1.1.0 devem permanecer congeladas durante a
   comparação de modelos. Alterá-las exigiria nova versão, novo cohort de
   Validation e repetição dos A/B.

## 10. Limitações assumidas

- O mesmo Luna gerou e julgou respostas em parte da calibração. O blinding
  reduz, mas não elimina, self-preference ou style bias.
- Não existiu adjudicação humana dos cinco scores em 50 casos; os scores são
  medidas judge-dependent, não verdade clínica independente.
- Um único judge reduz custo e ambiguidades de voting, mas não mede agreement
  inter-judge.
- O fallback Qwen melhora cobertura de safety refusals, introduzindo uma
  diferença de provider que deve ser reportada por caso e em agregado.
- Os estudos médicos e dermatológicos usam modelos, datasets e modalidades
  diferentes. Foram usados para formular hipóteses, não para substituir a
  evidência emparelhada do ISEPDermaBench.
