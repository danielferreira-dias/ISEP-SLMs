# Pipeline de Treino — Dermatology SLM/VLM
**Versão:** 18 de agosto de 2026

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
STAGE C — Supervised Fine-Tuning (SFT)
        ↓
STAGE D — On-Policy Distillation (OPD)
        ↓
STAGE E — RL / GRPO
        ↓
FINAL MODEL
```

Conceptualmente:

```text
Stage A/B → construir dados de alta qualidade
Stage C   → ensinar "como raciocinar"
Stage D   → ensinar a raciocinar autonomamente
Stage E   → otimizar diretamente o comportamento desejado
```

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
  "primary_lesion": "pigmented macule",
  "color": ["brown", "black"],
  "shape": "asymmetric",
  "border": "irregular",
  "surface": "flat",
  "distribution": "solitary",
  "additional_features": [
    "heterogeneous pigmentation",
    "blue-white veil"
  ]
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
DDx
+ REASONING
+ DIAGNOSIS
```

Exemplo:

```json
{
  "morphology": {
    "primary_lesion": "pigmented macule",
    "shape": "asymmetric",
    "border": "irregular",
    "additional_features": [
      "heterogeneous pigmentation",
      "blue-white veil"
    ]
  },
  "differential_diagnosis": [
    "melanoma",
    "atypical nevus",
    "pigmented basal cell carcinoma"
  ],
  "reasoning": "The marked asymmetry, irregular border and blue-white veil favor melanoma over an atypical nevus.",
  "diagnosis": "melanoma"
}
```

### Porque damos o label ao Teacher?

Porque o Teacher também pode classificar incorretamente.

```text
Ground truth = destino correto
Teacher      = ensina o caminho para chegar lá
```

Não usamos o Teacher como fonte de verdade do diagnóstico quando já possuímos ground truth.

---

## 4. Dataset final para SFT

```text
Tuas imagens (~6000)
        ↓
Stage A + Stage B
        ↓
Dataset enriquecido

        +

SkinCoT

        ↓
FINAL SFT DATASET
```

Cada exemplo idealmente contém:

```text
Image
→ Morphology
→ Differential Diagnosis
→ Reasoning
→ Diagnosis
```

---

## 5. Stage C — Supervised Fine-Tuning

### Flow

```text
Enriched Internal Dataset
        +
SkinCoT
        ↓
STUDENT
        ↓
SFT
        ↓
SFT CHECKPOINT
```

### O que o SFT ensina?

Principalmente:

> "É assim que uma boa resposta dermatológica deve ser construída."

Ensina:

- linguagem morfológica;
- relação entre visual features e doença;
- estrutura de differential diagnosis;
- estrutura de clinical reasoning;
- formato de output;
- diagnóstico final.

### Limitação

O SFT é maioritariamente imitation learning: o Student vê exemplos bons já construídos pelo Teacher.

Ainda não estamos diretamente a ensinar o modelo a corrigir os seus próprios erros.

---

## 6. Stage D — On-Policy Distillation

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
→ Morphology
        ↓
STAGE B
Image + Morphology + Ground Truth
→ Teacher B
→ DDx + Reasoning + Diagnosis
        ↓
STAGE C
Generated Dataset + SkinCoT
→ SFT
→ SFT Checkpoint
        ↓
STAGE D
Student sem label gera trajetória
→ Teacher vê trajetória + ground truth
→ Teacher logits
→ On-Policy Distillation
→ OPD Checkpoint
        ↓
STAGE E
Student gera structured output
→ Diagnosis/Morphology/DDx/Hierarchy rewards
→ GRPO
→ Final Checkpoint
```

---

## 14. Checkpoints e ablations

A pipeline é sequencial, mas devemos avaliar depois de cada stage:

```text
Base
  ↓ benchmark
SFT
  ↓ benchmark
SFT + OPD
  ↓ benchmark
SFT + OPD + GRPO
```

Exemplo:

| Checkpoint | Internal Derm | DermoBench |
|---|---:|---:|
| Base | 70 | 45 |
| + SFT | 78 | 57 |
| + OPD | 82 | 64 |
| + GRPO | 84 | 68 |

Isto mostra ganho **incremental**.

Para atribuição mais rigorosa, fazer também ablations:

```text
Base
Base + SFT
Base + OPD
Base + SFT + OPD
Base + SFT + GRPO
Base + SFT + OPD + GRPO
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

**Estado:** suporte oficial e adequado ao Stage C.

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
│
├── data/
│   ├── raw/
│   ├── morphology/
│   ├── reasoning/
│   └── sft/
│
├── stages/
│   ├── 01_generate_morphology.py
│   ├── 02_generate_reasoning.py
│   ├── 03_train_sft.py
│   ├── 04_train_opd_gold.py
│   ├── 05_train_grpo.py
│   └── 06_evaluate.py
│
├── rewards/
│   ├── diagnosis.py
│   ├── morphology.py
│   ├── differential.py
│   ├── hierarchy.py
│   └── reasoning.py
│
├── evaluation/
│   ├── internal_derm.py
│   ├── dermobench.py
│   ├── medqa.py
│   └── mmlu_medical.py
│
└── checkpoints/
    ├── base/
    ├── sft/
    ├── opd/
    └── grpo/
```

---

## 28. Evaluation

Avaliar exatamente o mesmo benchmark depois de cada checkpoint:

```text
BASE
 ↓
benchmark

SFT
 ↓
benchmark

OPD
 ↓
benchmark

GRPO
 ↓
benchmark
```

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
   → DDx/reasoning

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
