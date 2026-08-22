# Pipeline de Treino — Dermatology SLM/VLM
**Versão:** 22 de agosto de 2026

## Objetivo

Transformar um modelo que atualmente faz sobretudo:

```text
Imagem → classificação
```

num modelo capaz de:

```text
Imagem
  ↓
Morphology
  ↓
Differential Diagnosis
  ↓
Clinical Reasoning
  ↓
Diagnosis
```

A pipeline é sequencial, mas cada stage é independente e produz um checkpoint/dataset intermédio. Isto permite medir quanto cada componente acrescentou.

---

## 1. Visão geral da pipeline

```text
STAGE A — Morphology Dataset Generation
        ↓
STAGE B — Reasoning Dataset Generation
        ↓
STAGE C — E3 Supervised Fine-Tuning (SFT)
        ↓
STAGE D — E4 On-Policy Distillation (OPD, opcional)
        ↓
STAGE E — E5 RL / GRPO (opcional)
        ↓
FINAL MODEL
```

Conceptualmente:

```text
Stage A/B → construir e auditar dados grounded de alta qualidade
Stage C   → E3: hard distillation/SFT com justificação clínica estruturada
Stage D   → E4: experiência on-policy separada, se E3 justificar
Stage E   → E5: experiência GRPO separada, após auditoria dos rewards
```

Stage A/B, a auditoria e a materialização da release multitarefa primária estão
implementados. O runtime e a configuração reproduzível de Stage C também estão
implementados, mas nenhum checkpoint E3 é considerado treinado ou selecionado
até a execução e a avaliação generativa em `sft_dev` terminarem. Stage D e
Stage E permanecem contratos futuros: não existe ainda implementação de treino
OPD ou GRPO. Não é obrigatório executar D ou E para considerar E3 completo.

| Stage | Estado do código | Artefacto científico |
| --- | --- | --- |
| A/B | implementado e congelado | targets Teacher auditáveis |
| Materialização E3 | implementada e publicada | release multitarefa `e3_multitask_sft_v1` |
| C / E3 SFT | runner e configuração implementados; treino pendente | quatro checkpoints LoRA por época, depois seleção em `sft_dev` |
| D / E4 OPD | futuro, não implementado | checkpoint opcional derivado do E3 selecionado |
| E / E5 GRPO | futuro, não implementado | checkpoint opcional com parent declarado |

---

## 2. Stage A — Morphology Generation

### Input

- Imagem dermatológica
- O Teacher **não vê o label**

### Flow

```text
IMAGE
  ↓
TEACHER A
(no label)
  ↓
MORPHOLOGY
```

Exemplo:

```json
{
  "image_assessment": {
    "is_evaluable": true,
    "image_modality": "clinical",
    "views_available": ["close clinical view"],
    "quality_defects": ["tight_crop"],
    "has_anatomic_overview": false,
    "has_scale": false,
    "has_lateral_profile": false,
    "distribution_assessability": "partial",
    "color_reliability": "reliable"
  },
  "dominant_visual_pattern": "Solitary asymmetric pigmented macule.",
  "observations": [
    {
      "id": "obs_001",
      "concept_id": "lesion.primary",
      "value": "macule",
      "status": "present",
      "scope": "central lesion",
      "confidence": "high",
      "evidence_region": "central pigmented lesion"
    },
    {
      "id": "obs_002",
      "concept_id": "lesion.symmetry",
      "value": "asymmetric",
      "status": "present",
      "scope": "central lesion",
      "confidence": "high",
      "evidence_region": "overall lesion silhouette"
    },
    {
      "id": "obs_003",
      "concept_id": "lesion.color",
      "value": "brown and black",
      "status": "present",
      "scope": "central lesion",
      "confidence": "high",
      "evidence_region": "throughout the central lesion"
    },
    {
      "id": "obs_004",
      "concept_id": "lesion.border_regularity",
      "value": "irregular",
      "status": "present",
      "scope": "central lesion",
      "confidence": "high",
      "evidence_region": "peripheral lesion margin"
    }
  ],
  "not_assessable_features": ["full-body distribution"],
  "clinical_caption": "A solitary asymmetric brown-black macule has an irregular peripheral margin."
}
```

### Objetivo

Evitar que o Teacher veja primeiro `"melanoma"` e passe depois a procurar ou inventar features que confirmem melanoma.

Queremos uma descrição visual independente do diagnóstico.

---

## 3. Stage B — Reasoning Generation

Agora usamos:

- Imagem
- Morphology criada no Stage A
- Ground-truth label correto

### Flow

```text
IMAGE
+ MORPHOLOGY
+ CORRECT LABEL
      ↓
TEACHER B
      ↓
COMPACT GROUNDED FACTS
+ DIAGNOSIS-vs-ALTERNATIVE COMPARISONS
+ LIMITATIONS
+ RESPONSE POLICY
+ CLINICAL REASONING
```

Exemplo:

```json
{
  "anchor_evidence_status": "supported",
  "annotation_conflict": false,
  "annotation_conflict_reason": null,
  "diagnostic_confidence": "moderate",
  "diagnosis": "melanoma",
  "differential_comparisons": [
    {
      "alternative": "atypical nevus",
      "features_favoring_diagnosis": ["obs_002", "obs_003", "obs_004"],
      "features_favoring_alternative": ["obs_001"],
      "comparison": "Melanoma is favored over an atypical nevus by the marked asymmetry, irregular margin, and brown-black color variation."
    }
  ],
  "limitations": ["duration_and_evolution", "dermoscopy"],
  "response_policy": "ANSWER_DIFFERENTIAL",
  "non_evaluable_reason": null,
  "clinical_reasoning": "The visible asymmetric brown-black macule with an irregular margin supports melanoma with moderate confidence. Melanoma is favored over an atypical nevus because the asymmetry, border irregularity, and color variation are more concerning, although the macular and solitary presentation remains compatible with that alternative. Evolution and dermoscopic structures cannot be assessed from this image."
}
```

Stage B nunca reescreve Stage A. O JSON é um contrato interno compacto para
validação, proveniência e auditoria. Dentro desse contrato,
`clinical_reasoning` é o texto natural escrito diretamente pelo Teacher e
preservado verbatim como target do Student. Cada comparação tem de explicitar
as features que favorecem o gold e as que ainda tornam a alternativa plausível,
além de explicar por que são diferentes. O texto final deve sintetizar esses
factos sem expor raw chain-of-thought ou inventar observações.

O estado `supported` exige pelo menos um achado discriminativo que separe o
gold das alternativas; compatibilidade inespecífica fica em `weak`. O label
privado nunca pode aumentar artificialmente a força da evidência. Stage B não
introduz positivos ou negativos ausentes de Stage A, não infere exposição,
sintomas, palpação, evolução ou causalidade, e não produz recomendações de
tratamento, biópsia ou excisão. A confiança no texto deve coincidir com o campo
estruturado (`low`, `moderate` ou `high`) sem intervalos híbridos.

Para imagens avaliáveis, a política é sempre `ANSWER_DIFFERENTIAL`: informação
clínica, dermoscopia ou evolução ausentes podem ser registadas como limitações,
mas não originam perguntas nem recusa. `REQUEST_NEW_IMAGE` existe apenas quando
Stage A marcou `is_evaluable=false`; nesse caso não se gera um diagnóstico no
texto do Student. Um gold sem suporte numa imagem avaliável, ou um possível
conflito de anotação, é preservado para auditoria e excluído do target de treino
em vez de gerar evidência fictícia.

### Porque damos o label ao Teacher?

Porque o Teacher também pode classificar incorretamente.

```text
Ground truth = destino correto
Teacher      = ensina o caminho para chegar lá
```

Não usamos o Teacher como fonte de verdade do diagnóstico quando já possuímos
ground truth. Também não tratamos o ground truth como prova de que a evidência
está visível: `anchor_evidence_status` pode ser `supported`, `weak` ou
`unsupported`, sem obrigar o Teacher a inventar suporte.

---

## 4. Dataset final para SFT

```text
Tuas imagens (~6000)
        ↓
Stage A + Stage B
        ↓
Dataset enriquecido

        +

SkinCoT normalizado (braço auxiliar opcional)

        ↓
E3 SFT DATASET RELEASE
```

O dataset primário é A+B. SkinCoT não é importado como raw chain-of-thought:
exige auditoria de licença e overlap, revisão/normalização e proveniência
`target_source=human_reviewed_external`. Deve permanecer uma ablation separada
até demonstrar benefício.

### Separação entre representação interna e target do Student

Cada exemplo aceite preserva duas representações:

1. `stage_a` + `stage_b`: JSON interno, versionado e auditável;
2. `stage_b.clinical_reasoning`: resposta clínica natural escrita pelo Teacher.

Não existe renderer local nem banco de templates. O target é preservado
verbatim, enquanto os campos estruturados permitem verificar que o texto cita o
gold, cobre todas as alternativas e se mantém ligado aos IDs de evidência de
Stage A. A proveniência guarda modelo, provider, seed, limite de output, nível
de reasoning, política de exclusão do reasoning, prompt, schema e tentativa;
isto torna a origem auditável, embora não prometa reprodução byte a byte por
uma API externa. A geração principal via Vertex usa reasoning `medium`; os
pilotos anteriores em `high` são outro protocolo e não podem ser retomados no
mesmo output.

Para Vertex, cada tentativa preserva também tokens de input, output, total e
thinking quando o provider os disponibiliza. A interface calcula uma estimativa
acumulada em USD a partir de preços Standard/global fixados e datados no YAML;
essa estimativa é separada da faturação final e do saldo de créditos. Um limite
local opcional interrompe a campanha antes do pedido seguinte, enquanto o limite
efetivo deve ser configurado adicionalmente em Cloud Billing.

Exemplo de `clinical_reasoning`:

```text
The visible asymmetric brown-black macule with an irregular margin supports
melanoma with moderate confidence. Melanoma is favored over an atypical nevus
because the asymmetry, border irregularity, and color variation are more
concerning, although the macular and solitary presentation remains compatible
with that alternative. Evolution and dermoscopic structures cannot be assessed
from this image.
```

O Student não é treinado para repetir chaves JSON. A variação linguística vem do
Teacher, não de fórmulas locais; por isso, a release deve medir repetição,
comprimento e concentração estilística antes do treino e rejeitar padrões
degenerados.

### Fundamentação do formato híbrido

Esta separação é coerente com precedentes próximos, sem afirmar que exista um
único formato universalmente superior:

- SkinCaRe/SkinCoT constrói conteúdo clínico hierárquico em linguagem natural,
  embora use estrutura e normalização durante a criação dos dados
  ([SkinCaRe, 2024](https://arxiv.org/html/2405.18004v2)).
- SkinGPT-4 separa a descrição das features visuais do diagnóstico e mostra nas
  ablations que ensinar apenas features ou apenas diagnóstico perde parte do
  comportamento pretendido
  ([SkinGPT-4, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11226626/)).
- LLaVA-Med e BioMed-VITAL armazenam os exemplos em contentores estruturados,
  mas mantêm as respostas do Assistant em texto natural
  ([LLaVA-Med, 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf),
  [BioMed-VITAL, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aec33ab89b5986605cd7c331396e7e5c-Abstract-Datasets_and_Benchmarks_Track.html)).
- JSON continua adequado para interoperabilidade e extração de campos, como em
  geração estruturada de relatórios radiológicos, mas essa finalidade é distinta
  da aprendizagem de uma resposta clínica natural
  ([Adams et al., 2023](https://pubs.rsna.org/doi/10.1148/radiol.230725)).
- Resultados recentes em reasoning dermatológico também descrevem repetição e
  latência associadas a trajetórias longas, reforçando a opção por justificações
  curtas e verificáveis em vez de raw chain-of-thought
  ([SkinGPT-R1, 2025](https://arxiv.org/html/2511.15242v2)).

Logo, JSON é o plano de controlo e auditoria; `clinical_reasoning` é o alvo
principal de imitação. A linguagem é produzida pelo Teacher, mas continua
limitada por evidência estruturada e por gates verificáveis. Esta opção evita
ensinar ao Student a assinatura rígida de um renderer, em troca de menor
determinismo textual e da necessidade de auditar diversidade e repetição.

Cada exemplo A+B aceite contém semanticamente:

```text
Image
→ Morphology
→ Differential Diagnosis
→ Explicit diagnosis-vs-alternative comparison
→ Concise grounded natural-language justification
→ Diagnosis
```

---

## 5. Stage C — Supervised Fine-Tuning

### Flow

```text
Pinned E3 multitask train release
        +
Pinned human diagnosis sft_dev
        +
Qwen/Qwen3.5-4B official base
        ↓
Vision + language LoRA SFT
        ↓
epoch 1 / epoch 2 / epoch 3 / epoch 4 checkpoints
        ↓
generative selection on sft_dev only
        ↓
selected E3 SFT checkpoint
```

O braço primário E3 não carrega o adapter de E1 ou E2. Começa novamente na
revisão oficial fixada do Qwen, para que a diferença seja atribuível ao target
multitarefa E3 e não a treino cumulativo. SkinCoT continua uma ablation futura
separada e não está misturado na release primária.

### O que o SFT ensina?

Principalmente:

> "É assim que uma boa resposta dermatológica deve ser construída."

Ensina:

- linguagem morfológica;
- relação entre visual features e doença;
- estrutura de differential diagnosis;
- comparação explícita das features que favorecem o diagnóstico face às
  alternativas;
- justificação clínica curta e grounded;
- resposta natural sem obrigar o modelo a copiar uma família fixa de templates;
- diagnóstico final.

### Materialização multitarefa implementada

O Stage C não consome diretamente os JSONL extensos de auditoria. O comando
`isep-materialize-e3` volta a carregar o split `diagnosis` da revisão privada e
fixada de ISEPDistillDataset, reconcilia cada `sample_id` com os outputs Stage A
e Stage B e escreve um Parquet multimodal com uma conversa independente por
tarefa. A expansão é:

| Condição do exemplo | Tarefas materializadas | Origem do target |
| --- | --- | --- |
| Todos os exemplos da fonte | `diagnosis` | diagnóstico humano e prompt congelado |
| Stage A aceite | `morphology` | JSON canónico sem `clinical_caption` |
| Stage A aceite | `caption` | `clinical_caption` answer-blind |
| Stage B aceite e imagem avaliável | `grounded_differential` | `clinical_reasoning` preservado verbatim |
| Stage B aceite e imagem não avaliável | `request_new_image` | `clinical_reasoning` sem revelar o gold |

Os dois últimos comportamentos são mutuamente exclusivos. Assim, um exemplo
totalmente aceite gera quatro rows: diagnóstico, morfologia, caption e uma única
resposta clínica condicional. Esta decisão evita ensinar para a mesma imagem
dois comportamentos incompatíveis. Falhas do Teacher não eliminam o target
humano de diagnóstico; apenas reduzem as tarefas dependentes desse stage e são
registadas no manifest de cobertura.

O estado final de Stage B é rastreado por `sample_id` como `ok`, `rejected`,
`error`, `missing_attempt` ou `not_eligible_stage_a`, com precedência
`ok > rejected > error`. Um `rejected` é terminal e auditável: não entra como
target de Stage B e não volta a ser pedido automaticamente. Um `error` continua
retryable numa invocação posterior. O manifest conserva IDs e motivos das
rejeições, erros, IDs sem tentativa e contagens de tentativas duplicadas. Uma
imagem apenas com Stage A aceite mantém diagnóstico, morfologia e caption; a
ausência de Stage B deixa de ser um drop silencioso.

O release materializado inclui imagem, `messages`, IDs de tarefa, origem e hash
do target, referência e hash da imagem, grupo de leakage, IDs das tentativas A/B
e um manifest de integridade com contagens, bytes e SHA-256. A geração completa
é fail-closed por omissão; materialização parcial e overwrite requerem flags
explícitas.

A release privada publicada usada pelo treino está fixada em:

```text
repo:     danielfdias98/ISEPDistillDataset
revision: 4437aff671af4f4e32a2ebf006fdd3f4e72dea4f
config:   e3_multitask_sft_v1
split:    sft_train
rows:     25 084
```

As contagens esperadas são 6.312 `diagnosis`, 6.312 `morphology`, 6.312
`caption`, 6.127 `grounded_differential` e 21 `request_new_image`. O
materializador canónico é `project/pipeline/materialize_sft.py`; o antigo
`project/pipeline/sft.py` existe apenas como facade de compatibilidade e não é
um trainer.

O post-training do Student vive separadamente em `project/post_training` e é
invocado por `isep-post-train`. A recipe imutável do Student contém apenas o
base, precisão BF16 sem fallback, visão, LoRA e reprodutibilidade. Dataset,
optimizer, épocas, avaliação e outputs pertencem à configuração de Stage C em
`configs/training/e3_qwen3_5_4b_sft.yaml`.

```bash
uv run isep-post-train sft validate-config \
  --config configs/training/e3_qwen3_5_4b_sft.yaml
uv run isep-post-train sft smoke-test \
  --config configs/training/e3_qwen3_5_4b_sft.yaml
uv run isep-post-train sft run \
  --config configs/training/e3_qwen3_5_4b_sft.yaml
```

`validate-config` não carrega a GPU; `smoke-test` faz um fit técnico curto num
prefixo e pode alocar a GPU, mas não produz um checkpoint científico; só `run`
inicia a campanha completa.

O treino guarda um checkpoint LoRA no fim de cada uma das quatro épocas. A
seleção é um passo determinístico posterior sobre o split humano
`diagnosis/sft_dev` fixado na revisão
`b215f0474e4931b5951da768e79a0d579d26919d` (1.229 rows), usando
`macro_f1`, depois `balanced_accuracy`, `eval_loss` e a época mais antiga como
desempates. Enquanto essa avaliação não terminar, o estado correto é
`pending_sft_dev_generative_evaluation`: o runner não deve inventar nem inferir
um "best checkpoint" a partir de loss de treino.

Antes de reservar a GPU, o runner valida e persiste as contagens observadas e
esperadas das duas releases. Cada row E3 volta a verificar os hashes publicados
da imagem, prompt e target, além da igualdade entre `messages`, `prompt` e
`target_text`. Como o release histórico `sft_dev` não publicou
`target_sha256`, esse split usa explicitamente
`target_text == gold_diagnosis` e regista o digest calculado com o método
`gold_diagnosis_equality`, sem fingir que o hash veio do Hub. Depois de construir
o collator, a máscara assistant-only é testada numa row real de cada tarefa.

Uma execução full só pode terminar com exatamente os checkpoints íntegros das
épocas 1, 2, 3 e 4 (`checkpoint-3136`, `checkpoint-6272`, `checkpoint-9408` e
`checkpoint-12544`). A
retoma exige o `run_id` original, identidade imutável e o checkpoint mais
recente da própria run; não é permitido fazer rewind silencioso nem reescrever
manifestos com uma configuração diferente.

Como E1/E2 usaram três épocas, `checkpoint-9408` é preservado como o ponto E3
com orçamento de updates diretamente comparável. Os quatro checkpoints são
avaliados pelo mesmo protocolo generativo congelado em `sft_dev`, e o E3
selecionado pode ser a época 4. Se isso acontecer, a tese reporta separadamente
o resultado budget-matched da época 3 e o checkpoint selecionado da época 4,
marcando este último como comparação contextual com compute adicional face a
E1/E2.

### Métricas de treino e recursos

O namespace canónico é `project/metrics`. Durante o fit, o Stage C preserva:

- `loss`, `eval_loss`, learning rate, gradient norm, epoch e global step;
- segundos por step, exemplos/tokens/steps por segundo e duração total;
- utilização da GPU, VRAM usada, RAM do processo, potência e temperatura;
- energia integrada em Wh, GPU-hours, parâmetros treináveis e tamanho do
  checkpoint quando o checkpoint selecionado já estiver disponível;
- estado da run, sessões de retoma e identidade de configuração/dataset/modelo.

Estas métricas permitem apresentar curvas de otimização e, depois da avaliação
do checkpoint selecionado, fronteiras qualidade-versus-tempo, VRAM, energia e
custo. Os módulos históricos em `src.train` são apenas facades de
compatibilidade; métricas de qualidade de tarefas e eficiência de inferência
continuam nos respetivos módulos de avaliação/benchmark, porque têm protocolos
e denominadores diferentes.

### Limitação

O SFT é maioritariamente imitation learning: o Student vê exemplos bons já construídos pelo Teacher.

Ainda não estamos diretamente a ensinar o modelo a corrigir os seus próprios erros.

---

## 6. Stage D — On-Policy Distillation

Este stage futuro é OPD, não DPO. O dataset E3 não contém pares
`chosen`/`rejected`, por isso adicionar um `DPOTrainer` agora não corresponde ao
contrato dos dados nem ao objetivo on-policy. DPO só seria uma experiência nova
depois de criar e congelar um dataset de preferências próprio.

Aqui o Student já passou pelo SFT.

Agora o Student deixa de receber o label.

### Student

```text
IMAGE
  ↓
STUDENT
(no label)
  ↓
Morphology
+ DDx
+ Reasoning
+ Diagnosis
```

O Student pode estar errado.

### Teacher no OPD

O Teacher recebe:

```text
IMAGE
+ STUDENT TRAJECTORY
+ GROUND-TRUTH LABEL
        ↓
TEACHER
        ↓
Teacher token distribution / logits
```

### Flow completo

```text
             IMAGE
               │
               ▼
            STUDENT
         (sem label)
               │
               ▼
 morphology + DDx
 reasoning + diagnosis
               │
               ▼
            TEACHER
               ▲
               │
       ground-truth label
      (Student não vê)
               │
               ▼
        Teacher logits
               │
               ▼
        KL / JSD / ULD loss
               │
               ▼
         Student update
```

---

## 7. O que o OPD ensina?

SFT:

```text
"Imita estas demonstrações boas."
```

OPD:

```text
"Agora tenta sozinho.
O Teacher corrige exatamente os estados e erros
que tu próprio geraste."
```

O OPD pode ensinar o Student a:

- não saltar cedo demais para um diagnóstico;
- dar mais importância às features visuais relevantes;
- melhorar o ranking do DDx;
- corrigir reasoning que conduz a uma classe errada;
- manter consistência entre morphology → reasoning → diagnosis;
- chegar à resposta sem o ground-truth label como input.

---

## 8. Porque o Teacher continua a ver o label no OPD?

Para evitar:

```text
Student errado
    ↓
Teacher também errado
    ↓
KD ensina o erro
```

Em vez disso:

```text
Student não vê label
Teacher vê label
        ↓
Teacher está ancorado no target correto
```

Regra conceptual:

> **Ground truth diz para onde ir. O Teacher ensina como chegar lá.**

---

## 9. Stage E — RL / GRPO

Depois de SFT e OPD, podemos otimizar diretamente aquilo que queremos maximizar.

### Flow

```text
IMAGE
  ↓
STUDENT
  ↓
Structured output / JSON
  ↓
REWARD FUNCTIONS
  ↓
GRPO
  ↓
MODEL UPDATE
```

---

## 10. Output estruturado para RL

JSON não é obrigatório, mas torna os rewards muito mais simples e determinísticos.

```json
{
  "morphology": [
    "asymmetric pigmented lesion",
    "irregular border",
    "blue-white veil"
  ],
  "differential_diagnosis": [
    "melanoma",
    "atypical nevus",
    "pigmented BCC"
  ],
  "reasoning": "The asymmetry and blue-white veil favor melanoma over an atypical nevus.",
  "diagnosis": "melanoma"
}
```

---

## 11. Reward design

Não recomendamos apenas:

```text
correct diagnosis = +1
wrong diagnosis   = -1
```

Podemos criar uma reward composta:

```text
Total Reward =
    Diagnosis Reward
  + Diagnostic Hierarchy Reward
  + Morphology Reward
  + DDx Reward
  + Reasoning Consistency Reward
```

Exemplo inicial:

```text
Diagnosis correctness      0.50
Morphology                 0.20
Differential diagnosis     0.15
Diagnostic hierarchy       0.10
Reasoning consistency      0.05
```

Os pesos são hiperparâmetros.

### Diagnosis reward

```text
Ground truth: melanoma

Prediction: melanoma
→ reward alto

Prediction: atypical nevus
→ reward baixo

Prediction: psoriasis
→ reward muito baixo
```

### Hierarchical reward

Nem todos os erros são igualmente maus.

```text
Ground truth = melanoma

melanoma             → 1.0
atypical nevus       → 0.4
pigmented BCC        → 0.15
psoriasis            → 0.0
```

### Morphology reward

Comparar a morphology produzida com:

- annotations humanas;
- morphology criada no Stage A;
- atributos estruturados;
- opcionalmente um judge VLM.

### DDx reward

Pode considerar:

- diagnóstico correto presente no DDx;
- posição do diagnóstico correto;
- presença de alternativas clinicamente plausíveis;
- ausência de diagnósticos absurdos.

### Reasoning reward

Pode medir:

- consistência morphology → reasoning;
- consistência reasoning → diagnosis;
- ausência de contradições;
- groundedness visual.

Convém manter esta componente controlada porque rewards baseadas num LLM/VLM judge podem ser mais ruidosas e suscetíveis a reward hacking.

---

## 12. Diferença entre SFT, OPD e RL

### SFT

```text
Teacher demonstra
        ↓
Student imita
```

Ensina **como deve parecer uma boa resposta**.

### OPD

```text
Student tenta sozinho
        ↓
Teacher corrige a trajetória do Student
```

Ensina **como melhorar nos estados que o próprio Student visita**.

### RL / GRPO

```text
Student tenta várias respostas
        ↓
Reward mede quais são melhores
        ↓
Student aumenta a probabilidade dos comportamentos melhores
```

Ensina **quais comportamentos maximizam diretamente os nossos objetivos**.

---

## 13. Pipeline final

```text
STAGE A
Image
→ Teacher A sem label
→ Image assessment + atomic observations
        ↓
STAGE B
Image + Morphology + Ground Truth
→ Teacher B
→ Compact evidence + explicit differential comparisons + limitations
→ Teacher-generated clinical_reasoning preserved verbatim
        ↓
STAGE C / E3
Qwen/Qwen3.5-4B official base + pinned E3 multitask release
→ vision + language LoRA SFT
→ select only on frozen human sft_dev
→ E3 SFT Checkpoint
        ↓
STAGE D / E4 OPCIONAL
Student sem label gera trajetória
→ Teacher vê trajetória + ground truth
→ Teacher logits
→ On-Policy Distillation
→ OPD Checkpoint
        ↓
STAGE E / E5 OPCIONAL
Student gera structured output
→ Diagnosis/Morphology/DDx/Hierarchy rewards
→ GRPO
→ Final Checkpoint
```

---

## 14. Checkpoints e ablations

A pipeline preserva checkpoints independentes e comparáveis. E1, E2 e E3 são
braços paralelos que começam na mesma revisão oficial; não são uma cadeia de
continuação de adapters. E4/E5 só avançam depois de E3 ser congelado e
analisado:

```text
                         ┌→ E1 selected
Qwen3.5-4B official base├→ E2 selected
                         └→ E3 SFT selected
                                  └→ E4 OPD selected (opcional)

E5 GRPO (opcional) → parent checkpoint declarado antes da execução
```

Exemplo:

| Checkpoint | Internal Derm | DermoBench |
|---|---:|---:|
| Base | 70 | 45 |
| + SFT | 78 | 57 |
| + OPD | 82 | 64 |
| + GRPO | 84 | 68 |

Isto mostra ganho **incremental** sem apagar os baselines históricos. Para E3,
os quatro checkpoints de época são comparados exclusivamente na avaliação
generativa do `sft_dev` humano congelado; o `macro_f1` é o seletor primário. O
ISEPDermaBench, o DermoBench e qualquer benchmark externo só podem ser corridos
depois da seleção e do freeze e nunca selecionam epoch/checkpoint.

Para atribuição mais rigorosa, fazer também ablations:

```text
Base
E1 diagnosis-only desde o Base
E2 morphology/caption desde o Base
E3 multitask distilled SFT desde o Base
E3 + OPD (se executado)
E3 + GRPO ou E3 + OPD + GRPO, com parent pré-declarado (se executado)
```

---

# Frameworks Hugging Face — validação prática

## 15. Stage A/B — Dataset generation

Não é necessário um trainer especial.

Pode ser implementado com:

- `transformers`;
- `AutoProcessor`;
- `AutoModelForImageTextToText`;
- APIs externas, se o Teacher for fechado;
- vLLM quando o modelo for suportado.

**Estado:** fazível diretamente.

O runner operacional é `isep-generate-e3`. Ele fixa uma única coorte ordenada,
executa Stage A até cobertura aceite completa e só depois inicia Stage B sobre
os mesmos `sample_id`. A interface de terminal apresenta progresso, imagens em
falta, resultados aceites/rejeitados/falhados, amostra atual, ETA e custo
estimado acumulado quando a configuração fixa preços. Uma falha de provider ou
schema termina a campanha com código não-zero e impede a passagem
de A para B. O cliente Vertex usa uma única política Tenacity configurada e
limitada para erros HTTP transitórios (`408`, `429`, `500`, `502`, `503`,
`504`), com seis tentativas totais e backoff exponencial com jitter; o retry
implícito do SDK fica desativado para não multiplicar tentativas. Segurança e
schema nunca são regenerados automaticamente. Uma rejeição da validação clínica
em Stage B é terminal, fica excluída do SFT e não é repetida automaticamente;
apenas erros técnicos continuam elegíveis para retoma. Cada row
regista quantos pedidos físicos foram necessários. Uma nova invocação é uma
decisão explícita e retoma apenas erros técnicos ou IDs sem resultado terminal,
preservando
todas as tentativas no JSONL de auditoria. A retoma também verifica a identidade
do modelo, seed e hashes do prompt/schema; uma alteração de protocolo exige
novos outputs e nunca é misturada silenciosamente com registos anteriores.

### Freeze do protocolo de prompts E3 v1 — 2026-08-21

Os bytes usados nos dry runs ficam congelados e verificados no carregamento da
configuração:

| Stage | Versão | SHA-256 |
| --- | --- | --- |
| A | `e3_stage_a_v1` | `c28f6ff4f9a47ba23bc02f2a6d14541ee5afeeaf134bca5cf48936f150121a4f` |
| B | `e3_stage_b_v1` | `b8239b38c24eac6037c22bcfcbc3573deb37cd5011c97d234e4179f20718125e` |

Qualquer alteração semântica ou textual exige prompts v2, novos hashes, novos
outputs e novo pilot. Correções de retry, auditoria ou materialização podem ser
feitas sem quebrar este freeze desde que o request protocol do Teacher não
mude.

### Freeze do dataset Stage A E3 v1 — 2026-08-22

A geração Stage A está completa para os 6.312 IDs do `sft_train`. A release
accepted-only final é
`project/data/morphology/frozen/e3_stage_a_v1_20260822/stage_a.jsonl`, com
SHA-256
`1eefa665d791c5138ffc00d57c5d9161ab899985949d8d4c2f7e54d12db89bd2`.
O audit log com todas as tentativas, os outputs Batch em quarentena e os
backups de correções de metadata permanecem separados. O freeze preliminar de
2026-08-21 foi supersedido apenas porque o canário tinha custo Standard em vez
de Batch; os targets não mudaram. A auditoria e as limitações de aderência à
prompt estão documentadas em
`annotations/dataset_pipeline/14_e3_stage_a_teacher_dataset_freeze.md`.

---

## 16. SFT — TRL `SFTTrainer`

O `SFTTrainer` tem suporte explícito para Vision-Language Models.

O dataset pode conter:

```text
image / images
+
messages
```

Para VLMs, é importante evitar truncar image tokens. Uma configuração segura para começar é:

```python
SFTConfig(max_length=None)
```

**Estado:** suporte oficial e adequado ao Stage C. O adapter E3 para o backend,
a validação dos pins e o runner existem em `project/post_training`; esta nota
não significa que o treino E3 ou a seleção do checkpoint já tenham sido
executados.

---

## 17. OPD — TRL `GOLDTrainer`

Para este projeto multimodal, a melhor base atual é:

```text
trl.experimental.gold.GOLDTrainer
```

GOLD = **General Online Logit Distillation**.

O GOLDTrainer documenta explicitamente:

- on-policy distillation;
- VLM → VLM distillation;
- student-generated rollouts;
- generalized JSD;
- cross-tokenizer distillation;
- cross-family VLM distillation;
- LoRA;
- exemplos com Qwen3-VL.

### Mesmo model family

```text
Qwen3-VL Teacher
        ↓
Qwen3-VL Student
```

Podemos usar JSD/KL diretamente.

### Different model families/tokenizers

```text
Teacher family A
      ↓
Student family B
```

GOLD fornece **Universal Logit Distillation (ULD)** para alinhar tokenizações diferentes.

Isto é importante porque uma KL clássica entre logits de vocabulários diferentes não é diretamente válida.

---

## 18. Customização necessária no GOLD

O GOLDTrainer suporta o núcleo técnico de VLM on-policy distillation.

O nosso caso tem uma condição adicional:

```text
Student prompt:
IMAGE
(no label)

Teacher prompt:
IMAGE
+ Student trajectory
+ ground-truth label
```

O Teacher recebe **privileged information** que o Student não recebe.

Esta variante answer-aware/ground-truth-aware não aparece como workflow pronto a usar no GOLD.

Logo, o Stage D deve ser tratado como:

```text
GOLDTrainer
+
custom teacher prompt construction
```

Provavelmente será necessário subclassificar ou adaptar o Trainer para construir prompts diferentes para Student e Teacher.

O core de:

- rollout;
- logit extraction;
- JSD/ULD;
- gradient update;

já existe.

**Estado:** tecnicamente adequado e explicitamente VLM-compatible, mas a variante ground-truth-conditioned Teacher requer customização.

Além disso, GOLD continua em `trl.experimental`, portanto a API pode mudar.

---

## 19. `DistillationTrainer`

O TRL também possui:

```text
trl.experimental.distillation.DistillationTrainer
```

Suporta:

- fully on-policy distillation;
- forward KL;
- reverse KL;
- interpolated JSD;
- student generation;
- local teacher;
- external teacher via vLLM server.

No entanto, a documentação atual do `DistillationTrainer` é orientada para causal LMs textuais e não apresenta o mesmo suporte VLM explícito que GOLD.

Para este projeto multimodal:

```text
Preferência:
GOLDTrainer > DistillationTrainer
```

O `DistillationTrainer` continua útil como referência de arquitetura, especialmente para separar o Teacher num servidor externo.

---

## 20. RL — TRL `GRPOTrainer`

O `GRPOTrainer` tem suporte explícito para Vision-Language Models e suporta reward functions customizadas em Python.

Exemplo conceptual:

```python
def diagnosis_reward(completions, labels, **kwargs):
    ...

def morphology_reward(completions, morphology_targets, **kwargs):
    ...

def ddx_reward(completions, labels, **kwargs):
    ...

def hierarchy_reward(completions, labels, **kwargs):
    ...
```

Depois:

```python
GRPOTrainer(
    model=student,
    reward_funcs=[
        diagnosis_reward,
        morphology_reward,
        ddx_reward,
        hierarchy_reward,
    ],
)
```

**Estado:** suporte oficial VLM e adequado ao Stage E.

---

## 21. Stack recomendada

```text
Dataset generation:
Transformers / vLLM / Teacher APIs

SFT:
TRL SFTTrainer

On-Policy Distillation:
TRL experimental GOLDTrainer
+ custom ground-truth-conditioned teacher prompt

RL:
TRL GRPOTrainer

Efficiency:
PEFT / LoRA
Accelerate
DeepSpeed / FSDP, se necessário
vLLM para rollouts quando suportado
```

---

# GPU / Infraestrutura

## 22. Precisamos de dois GPUs?

**Não obrigatoriamente.**

Mas durante OPD, ter recursos separados para Teacher e Student é normalmente a arquitetura mais simples.

---

## 23. GPU requirements por stage

### Stage A/B — Dataset generation

```text
Teacher inference only
```

Pode ser apenas:

```text
1 GPU
```

e correr Teacher A e Teacher B sequencialmente.

Se usarmos Gemini/API:

```text
0 GPUs locais para o Teacher
```

Não precisamos de ter Teacher A e Teacher B carregados simultaneamente.

---

### Stage C — SFT

Só precisamos do Student:

```text
GPU(s)
  ↓
Student training
```

Uma GPU pode chegar se:

- Student for relativamente pequeno;
- LoRA/QLoRA;
- batch size pequeno;
- gradient accumulation.

Full fine-tuning ou Student maior pode exigir várias GPUs.

---

### Stage D — OPD

Aqui temos simultaneamente:

```text
STUDENT
- geração
- forward
- backward
- optimizer states

TEACHER
- inference
- logits
```

A configuração mais limpa é:

```text
GPU POOL A
Student training

        ↕ logits / requests

GPU POOL B
Teacher inference
```

Isto pode significar:

#### Opção 1 — Dois GPUs na mesma máquina

```text
GPU 0 → Student
GPU 1 → Teacher
```

#### Opção 2 — Dois servidores

```text
Server A → Student training
Server B → Teacher inference
```

#### Opção 3 — Mais de duas GPUs

Se o Teacher for muito grande:

```text
GPU 0-1 → Teacher tensor parallel
GPU 2   → Student
```

#### Opção 4 — Uma única GPU

Só é realista se:

```text
Teacher + Student + optimizer
```

couberem em VRAM, ou com offload/quantization agressiva.

Para um Teacher grande + Student em treino, geralmente não é a opção preferida.

---

## 24. Melhor forma de pensar na infraestrutura

Não é obrigatório serem literalmente “dois GPUs diferentes”.

É melhor pensar em:

```text
Teacher compute pool
+
Student compute pool
```

Podem estar:

- na mesma máquina;
- em máquinas diferentes;
- num único node multi-GPU;
- parcialmente via API.

O requisito é memória/compute suficiente.

---

## 25. GRPO — precisamos novamente do Teacher?

Não necessariamente.

Se os rewards forem calculados com:

- ground-truth label;
- diagnostic hierarchy;
- morphology annotations;
- regras determinísticas;

não precisamos de um Teacher grande.

```text
Student
  ↓
Rollouts
  ↓
Reward functions
  ↓
GRPO
```

Se usarmos um LLM/VLM judge para morphology/reasoning reward, então voltamos a introduzir um segundo modelo de inferência.

---

## 26. Infraestrutura recomendada inicialmente

```text
STAGE A/B
Teacher/API
→ dataset offline

STAGE C
1 Student training GPU
→ SFT checkpoint

STAGE D
GPU/Pool A → Student training
GPU/Pool B → Teacher inference
→ OPD checkpoint

STAGE E
Student training GPU(s)
→ GRPO
→ final checkpoint
```

A infraestrutura pode ser desligada entre stages. Não precisamos de pagar todos os GPUs durante a pipeline inteira.

---

## 27. Organização em código

```text
project/
├── configs/student_configs/qwen_3_5_4b.yaml  # recipe imutável
├── metrics/
│   ├── contracts.py                          # eventos/sinks escalares
│   ├── trainer_events.py                     # loss/LR/tempo por step
│   ├── resources.py                          # CPU/GPU/VRAM/potência
│   └── resource_metrics.py                   # agregação por run
└── pipeline/
    ├── materialize_sft.py                    # dataset A/B → SFT
    └── sft.py                                # facade legada

configs/training/
└── e3_qwen3_5_4b_sft.yaml                    # contrato Stage C

project/post_training/
├── cli.py                                    # isep-post-train
├── _availability.py                          # estado dos stages
├── common/
│   ├── config.py                             # validação + pins
│   └── data.py                               # adapter VLM/HF
├── sft/
│   └── runner.py                             # Stage C implementado
├── opd/__init__.py                           # contrato futuro; sem trainer
└── grpo/__init__.py                          # contrato futuro; sem trainer

outputs/training/post_training/
└── e3_qwen3_5_4b_sft/                        # manifests + LoRA checkpoints
```

Esta separação impede confundir criação de dados com otimização do
Student. O namespace `project.pipeline` não instancia `SFTTrainer`; o namespace
`project.post_training` usa a monitorização canónica de `project.metrics` e
reutiliza temporariamente o backend e os contratos gerais de execução maduros
de `src/train`. Os módulos OPD/GRPO expõem apenas o estado
"não implementado" e não simulam uma execução bem-sucedida.

---

## 28. Evaluation

Durante a seleção, avaliar os quatro checkpoints apenas no mesmo `sft_dev`
humano congelado:

```text
epoch 1 / epoch 2 / epoch 3 / epoch 4
                  ↓
       generative sft_dev
                  ↓
      selected checkpoint frozen
```

Só depois da seleção e do freeze se executam os benchmarks principais e de
retenção no checkpoint selecionado. MMLU Medical, MedQA e MedMCQA nunca são
usados para escolher a época e não são repetidos nos quatro checkpoints.

### Primary

```text
Internal Dermatology Benchmark
DermoBench
```

### Secondary / retention

```text
MedQA
MMLU Medical
MedMCQA
```

MedQA/MMLU não são o target principal desta pipeline. Servem inicialmente para verificar:

```text
Dermatology ↑
General medical capability ≈ preserved
```

---

## 29. Decisão técnica recomendada

```text
A. Teacher inference
   → morphology

B. Teacher inference + label
   → compact DDx evidence, pairwise comparisons, and clinical_reasoning

C. TRL SFTTrainer
   → Student SFT

D. TRL GOLDTrainer
   + custom label-aware teacher prompt
   → On-Policy VLM Distillation

E. TRL GRPOTrainer
   + deterministic dermatology rewards
   → final alignment
```

Guardar:

```text
checkpoint_base
checkpoint_sft
checkpoint_opd
checkpoint_grpo
```

para medir claramente o ganho incremental.

---

## 30. Estado das frameworks verificado

Verificado na documentação oficial Hugging Face / TRL em 18 de agosto de 2026:

- `SFTTrainer`: suporte explícito para treino de VLMs.
- `GRPOTrainer`: suporte explícito para treino de VLMs e custom reward functions.
- `GOLDTrainer`: suporte explícito para VLM→VLM on-policy/logit distillation, incluindo cross-tokenizer/cross-family via ULD.
- `DistillationTrainer`: on-policy KD e external teacher server, mas permanece experimental e a documentação atual não apresenta suporte VLM first-class equivalente ao GOLD.
- `GOLDTrainer` e `DistillationTrainer` pertencem atualmente ao namespace experimental do TRL; APIs podem mudar.

---

## Referências técnicas

- Hugging Face TRL — SFT Trainer  
  https://huggingface.co/docs/trl/sft_trainer

- Hugging Face TRL — GRPO Trainer  
  https://huggingface.co/docs/trl/grpo_trainer

- Hugging Face TRL — GOLD Trainer  
  https://huggingface.co/docs/trl/main/gold_trainer

- Hugging Face TRL — Distillation Trainer  
  https://huggingface.co/docs/trl/distillation_trainer

- Hugging Face TRL GitHub  
  https://github.com/huggingface/trl
