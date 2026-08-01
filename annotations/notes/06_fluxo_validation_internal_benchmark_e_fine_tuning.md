# Fluxo entre Validation, Internal Benchmark e fine-tuning

## 1. Distinção principal

A Validation não serve apenas para dry-run. É também o conjunto onde são
tomadas todas as decisões de desenvolvimento.

A palavra *benchmark* pode referir-se a duas coisas diferentes:

- **benchmark/tarefa**: Visual Top-K, Visual Confusion Sets ou
  Evidence-Grounded Diagnosis;
- **conjunto de avaliação**: Validation ou Internal Benchmark.

A mesma benchmark pode ser executada nos dois conjuntos:

```text
Visual Top-K
├── Validation
└── Internal Benchmark

Visual Confusion Sets
├── Validation
└── Internal Benchmark

Evidence-Grounded Diagnosis
├── Validation
└── Internal Benchmark
```

## 2. Flow completo

```text
1. VALIDATION — pequeno subconjunto
   10–20 imagens
   └── dry-run técnico
       imagem, API, prompt, JSON, parser, reasoning, HTML
                │
                ▼
2. VALIDATION — subconjuntos progressivos
   50–100 e depois 200–300 casos
   └── ajustar prompts, parsers e configurações
       comparar modelos e eliminar candidatos
                │
                ▼
3. VALIDATION — conjunto completo
   └── comparar os teachers finalistas
       escolher teacher, prompt, parâmetros e modo de output
                │
                ▼
4. CONGELAR O PROTOCOLO
   teacher + prompt + parser + schema + parâmetros
                │
                ▼
5. INTERNAL BENCHMARK — antes do treino
   └── guardar baseline do student
       guardar resultado do teacher
       não usar estes resultados para fazer alterações
                │
                ▼
6. TRAIN
   └── teacher gera dados sintéticos
       student é treinado
                │
                ▼
7. VALIDATION — durante/depois do treino
   └── escolher checkpoint
       verificar overfitting
       calibrar thresholds
                │
                ▼
8. INTERNAL BENCHMARK — depois do treino
   └── executar o student treinado
       comparar com o mesmo student antes do treino
```

## 3. Comparação principal da tese

| Modelo | Momento | Conjunto |
| --- | --- | --- |
| Student base | Antes do fine-tuning | Internal Benchmark |
| Teacher | Antes do fine-tuning | Internal Benchmark |
| Student treinado | Depois do fine-tuning | Internal Benchmark |

Esta comparação procura responder a três perguntas:

```text
Quanto melhorou o student?
O student treinado aproxima-se do teacher?
O fine-tuning aproximou um modelo pequeno de um modelo maior?
```

## 4. Regra de utilização dos conjuntos

```text
Validation
= local onde podemos experimentar e tomar decisões

Internal Benchmark
= local onde apenas medimos configurações já escolhidas
```

O student base pode ser executado no Internal Benchmark antes do treino e as
previsões podem ser guardadas. Contudo, os erros não devem ser consultados
para alterar o prompt, parser, teacher, parâmetros ou processo de treino.

Depois do fine-tuning, o student treinado é executado exatamente nos mesmos
casos. Isto permite uma comparação emparelhada entre o comportamento antes e
depois do treino.

## 5. SFT Dev não é a benchmark Validation

Durante o fine-tuning, o conjunto de treino deve ter a sua própria divisão:

```text
Train
├── SFT Train → participa nos gradientes
└── SFT Dev   → acompanha a loss e ajuda no early stopping
```

O `SFT Dev` acompanha a otimização do modelo e ajuda a detetar overfitting. A
benchmark Validation mede se um checkpoint é realmente melhor nas tarefas
dermatológicas pretendidas.

Assim, os dois conjuntos têm funções relacionadas, mas distintas:

| Conjunto | Função |
| --- | --- |
| SFT Dev | Monitorizar loss, overfitting e early stopping durante o treino |
| Benchmark Validation | Escolher teacher, protocolo, checkpoint final e thresholds com métricas dermatológicas |
| Internal Benchmark | Produzir a comparação final selada antes/depois |

## 6. Regra resumida

```text
Desenvolver na Validation
Treinar com SFT Train e monitorizar com SFT Dev
Congelar todas as decisões
Medir no Internal Benchmark
```
