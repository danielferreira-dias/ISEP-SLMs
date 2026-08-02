# Seleção do teacher, validação das benchmarks e dados sintéticos interativos

## 1. Benchmark e conjunto de avaliação não são a mesma coisa

Uma benchmark define o **contrato da tarefa**:

- prompt;
- schema de output;
- taxonomia;
- regras de parsing e validação;
- métricas;
- processamento da imagem.

Um evaluation set define **em que casos esse contrato é executado**. A mesma
benchmark Visual Top-K pode ser executada em:

```text
visual_top_k
├── validation
├── internal_benchmark_1000
├── internal_test
├── external_ddi
└── external_skindisnet
```

Portanto, selecionar o teacher na Validation continua a ser **correr a
benchmark**. Apenas se usam perguntas de desenvolvimento em vez das perguntas
seladas do exame final.

## 2. O que estaria errado no fluxo original

O fluxo inicialmente imaginado era:

1. correr as benchmarks finais em todos os modelos;
2. escolher o modelo com melhor accuracy como teacher;
3. usar esse teacher para gerar targets sobre Train;
4. treinar o student;
5. voltar a medir nos mesmos benchmarks.

Isto funciona tecnicamente, mas cria selection bias. Os resultados do teste
passam a influenciar a escolha do teacher e, por consequência, o dataset
sintético e o student:

```text
Casos do teste final
       ↓
Escolha do teacher
       ↓
Targets sintéticos
       ↓
Student treinado
```

Mesmo sem gradientes, houve otimização em função do teste. Se nove modelos
forem comparados e for escolhido o maior resultado, parte da vantagem pode
ser ruído específico desses casos. A accuracy final tende então a ser uma
estimativa otimista.

Não é proibido escolher o teacher dessa forma, mas nesse caso os conjuntos
consultados passam metodologicamente a development sets e seria necessário
criar outro teste final intocado.

## 3. Como foi criada a Validation atual

O split é definido em `configs/datasets/visual_top_k_split.yaml`.

As fontes internas originais são:

- Fitzpatrick17k-C;
- PAD-UFES-20;
- SCIN.

Antes do split, as imagens foram normalizadas e deduplicadas. O agrupamento é
feito por `leakage_group_id`, que agrega imagens relacionadas pelo mesmo
paciente/caso e relações de duplicidade. Um grupo nunca é dividido entre
Train, Validation e Internal Test.

O algoritmo congelado usa seed 42 e uma divisão aproximada de grupos:

```text
70% Train
15% Validation
15% Internal Test
```

A estratificação procura preservar:

- `disease_id`;
- dataset de origem;
- combinação de doença e dataset;
- secundariamente, suporte demográfico e de tom de pele.

O resultado original da partição, antes do rebalanceamento da release, era:

| Split | Imagens | Grupos independentes | Classes |
| --- | ---: | ---: | ---: |
| Train original | 6.417 | 4.962 | 21 |
| Validation | 1.683 | 1.063 | 21 |
| Internal Test | 1.722 | 1.063 | 21 |

A release operacional ISEPDermaBench 1.2.0 mantém 1.000 dessas imagens em 646
grupos completos. As restantes 683 imagens pertencem a 417 grupos completos
e passaram para ISEPDermData Train; nenhum grupo foi dividido.

## 4. Porque o Internal Benchmark tem exatamente 1.000 casos

O `internal_benchmark_1000.parquet` não corresponde aos 15% completos nem foi
criado em alternativa à Validation. É uma vista do Internal Test:

1. parte dos 1.063 grupos selados do Internal Test;
2. seleciona deterministicamente 1.000 grupos;
3. escolhe no máximo uma imagem por grupo;
4. preserva a distribuição de classes e datasets;
5. usa seed 1042.

Por isso:

| Ficheiro | Imagens | Grupos | Explicação |
| --- | ---: | ---: | --- |
| `visual_top_k/validation` | 1.000 | 646 | Desenvolvimento; grupos completos e proteção das outras tasks |
| `internal_test.parquet` | 1.722 | 1.063 | Teste interno completo e selado |
| `internal_benchmark_1000.parquet` | 1.000 | 1.000 | Um caso por grupo para comparação emparelhada |
| `internal_test_reserve.parquet` | 63 | 63 | Um representante dos grupos não escolhidos para os 1.000 |

Assim, executar Visual Top-K em Validation utiliza atualmente exatamente
1.000 imagens. Durante o screening inicial continua a ser possível usar um
`--limit` determinístico menor.

### 4.1 A confusão original entre Validation e Internal Benchmark

O fluxo inicialmente imaginado estava correto do ponto de vista operacional,
mas atribuía ao `internal_benchmark_1000` o papel que pertence à Validation:

```text
Ideia inicial
  Internal Benchmark
      -> experimentar modelos
      -> comparar potenciais teachers
      -> ajustar prompts e parsers
      -> escolher o teacher
      -> voltar a usar depois do fine-tuning

Fluxo metodologicamente protegido
  Validation
      -> experimentar modelos
      -> comparar potenciais teachers
      -> ajustar prompts e parsers
      -> escolher o teacher

  Internal Benchmark
      -> comparar configurações já congeladas
      -> medir base student, teacher e trained student
      -> produzir o resultado principal antes/depois
```

O problema não era executar a benchmark. Visual Top-K continua a ser a mesma
benchmark nos dois conjuntos. O problema era consultar as respostas do teste
selado e usar os seus resultados para tomar decisões. Nesse momento o teste
passaria a ser, na prática, um conjunto de desenvolvimento.

### 4.2 Função das três tarefas e dos seus conjuntos de avaliação

Existem duas dimensões diferentes: a **tarefa** que se pretende medir e o
**conjunto de casos** no qual essa tarefa é executada.

| Benchmark/tarefa | O que mede | Resultado principal |
| --- | --- | --- |
| Visual Top-K | Reconhecimento visual fechado nas 21 doenças | Top-1, Top-3, Top-6, MRR e métricas por classe/subgrupo |
| Visual Confusion Sets | Capacidade de distinguir doenças visualmente semelhantes | Accuracy por nível de confusabilidade e gap low/high |
| Evidence-Grounded Diagnosis | Morfologia, descrição clínica e ligação explícita entre achados e diferencial | Qualidade dos findings, descrição, diagnóstico e evidence links |

Cada uma destas tarefas pode depois usar conjuntos com funções diferentes:

| Conjunto | Pode influenciar decisões? | Função |
| --- | --- | --- |
| Smoke/development subset | Sim | Verificar integração, prompts, schema, parser, truncation e erros de output |
| Validation | Sim | Comparar teachers finalistas, modos de output e parâmetros; desenvolver e selecionar |
| Internal Benchmark | Não | Comparação principal selada e emparelhada antes/depois do fine-tuning |
| Internal Test completo | Não | Auditoria interna completa, incluindo casos fora da vista principal de 1.000 |
| External | Não | Medir generalização para outras instituições, populações e distribuições |

### 4.3 Fluxo progressivo dentro da Validation

Os 1.000 casos não devem ser executados depois de cada pequena alteração. A
Validation funciona como um pool de desenvolvimento com etapas progressivas:

```text
10-20 casos
  smoke test técnico
      |
      v
50-100 casos fixos
  inspeção do prompt, schema, parsing e comportamento
      |
      v
200-300 casos estratificados
  comparar configurações e eliminar candidatos claramente inferiores
      |
      v
Validation completa
  comparar apenas os dois ou três finalistas e escolher o teacher
      |
      v
congelar modelo + prompt + parser + parâmetros + preprocessamento
      |
      v
Internal Benchmark
  medição final sem novos ajustes
```

Assim, apenas a primeira etapa é um dry-run puro. As etapas intermédias
produzem estimativas de desenvolvimento; a execução completa da Validation é
uma confirmação final da escolha e não um ciclo repetido de prompt tuning.

### 4.4 Proposta para reduzir a Validation para 1.000 casos

> Estado: implementado em ISEPDermaBench 1.2.0 e ISEPDermData 1.3.0.
> O texto abaixo preserva o raciocínio anterior à materialização. A seleção
> final mantém 1.000 imagens de 646 grupos e transfere 683 imagens de 417
> grupos completos para Train. Os 504 grupos necessários às restantes tasks
> de Validation foram obrigatoriamente preservados.

É razoável reduzir a superfície operacional da Validation para 1.000 casos.
No entanto, a unidade de separação é o `leakage_group_id`, não a row individual.
A Validation atual contém 1.683 imagens, mas apenas 1.063 grupos. Muitas das
683 imagens aparentemente excedentes são vistas adicionais dos mesmos casos.

Por isso, não é seguro fazer simplesmente:

```text
1.683 imagens - 683 rows -> 1.000 imagens de Validation
683 rows removidas       -> Train
```

Se uma imagem de um caso permanecer em Validation e outra imagem do mesmo
caso entrar em Train, o student poderá aprender informação visual do caso que
mais tarde será usado para selecionar o teacher ou o checkpoint.

A opção recomendada é criar uma vista de desenvolvimento com:

```text
validation_benchmark_1000
  1.000 grupos independentes
  1 imagem representativa por grupo
  21 classes preservadas
  distribuição por fonte e subgrupos preservada
  casos raros de Evidence e Confusion protegidos durante a seleção
```

Dos 1.063 grupos atuais, 1.000 permaneceriam protegidos. Apenas os 63 grupos
não selecionados poderiam ser transferidos integralmente para Train. As
imagens adicionais pertencentes aos 1.000 grupos protegidos teriam de continuar
fora do treino, mesmo que não aparecessem na vista de 1.000 tarefas.

Uma alternativa seria reconstruir o split para manter aproximadamente 1.000
imagens distribuídas por cerca de 600-650 grupos e transferir os restantes
grupos para Train. Isso acrescentaria mais imagens ao treino, mas reduziria o
número efetivo de casos independentes usados na seleção do teacher. Para esta
tese, a primeira opção é metodologicamente mais forte: **1.000 casos
independentes**, e não apenas 1.000 rows.

Esta alteração deve ser feita antes das execuções completas e exige reconstruir
as releases de Validation de Visual Top-K, Visual Confusion Sets e
Evidence-Grounded Diagnosis, atualizar checksums e confirmar novamente zero
overlap por `leakage_group_id`.

## 5. Estado de cada benchmark para teacher selection

### 5.1 Visual Top-K

Esta benchmark já está preparada corretamente. O YAML declara os diferentes
evaluation sets e o CLI permite selecionar Validation:

```bash
uv run python -m src.benchmark.cli run \
  --model gpt_5_6_luna \
  --benchmark visual_top_k_closed_set \
  --evaluation-set validation
```

O ficheiro
`configs/experiments/teacher_selection_visual_validation_v1.yaml` regista
explicitamente que os resultados podem selecionar teacher, prompt, parser,
parâmetros de geração e thresholds.

### 5.2 Visual Confusion Sets

Existem agora duas releases separadas:

| Evaluation set | Imagens/grupos | Tarefas | Uso |
| --- | ---: | ---: | --- |
| `validation_paired_confusion_tasks` | 417 | 834 | Seleção do teacher e desenvolvimento |
| `paired_confusion_tasks` | 414 | 828 | Comparação interna final selada |

A release final continua derivada do `internal_benchmark_1000`. A nova
release de desenvolvimento parte exclusivamente de `validation.parquet`:

```text
data/benchmarks/derma_isep/visual_confusion_sets_v1/
└── datasets/
    ├── development/validation_confusion_tasks.parquet
    └── internal/confusion_tasks.parquet
```

A Validation tem 1.683 imagens em 1.063 grupos. Primeiro é escolhido um
representante determinístico de cada grupo com seed 3042. Depois aplica-se o
mesmo algoritmo de downsampling para a classe mais rara de cada confusion
set, a mesma seed de seleção 2042 e as mesmas regras de candidatos. O suporte
real originou 417 imagens e não 414; manter o resultado natural evita remover
casos apenas para forçar igualdade cosmética com o teste final.

Cada imagem recebe uma tarefa low-confusability e outra high-confusability,
originando 834 tarefas. A release cobre os mesmos cinco confusion sets e 15
doenças da versão final.

### 5.3 Evidence-Grounded Diagnosis

A release externa continua a usar DDI, com referências provenientes de
SKINCON e SkinCAP:

| Componente | Casos atuais |
| --- | ---: |
| Morfologia | 636 |
| Descrição | 635 |
| Diagnóstico/grounding | 294 |

DDI está reservado para avaliação externa. Os seus resultados não escolhem o
teacher, prompt ou thresholds.

Foi possível criar uma Validation interna sem gerar gold labels através dos
próprios candidatos. A interseção entre Fitzpatrick17k-C Validation, SKINCON e
SkinCAP produziu:

| Componente | Casos de Validation |
| --- | ---: |
| Morfologia SKINCON | 137 |
| Descrição SkinCAP | 124 |
| Diagnóstico/grounding | 137 |
| Grupos independentes | 137 |
| Doenças cobertas | 19 de 21 |

O manifest está em
`datasets/development/evidence_grounded_validation.parquet` e usa
`evaluation_origin=development_validation`. D002 (melanocytic nevus) e D006
(seborrheic keratosis) não têm casos anotados elegíveis nesta interseção.
Por isso, esta release pode selecionar o teacher para morfologia, descrição
e grounding, mas não substitui uma avaliação equilibrada das 21 classes.

As descrições SkinCAP continuam a ser referências secundárias: podem conter
diagnósticos, testes ou recomendações. As métricas de conceitos morfológicos
e consistência devem ter prioridade sobre simples semelhança textual.

## 6. Sequência experimental recomendada

```text
1. Smoke test técnico em 10-20 casos de desenvolvimento
2. Ajustar prompt, schema e parser em 50-100 casos fixos
3. Comparar configurações numa amostra estratificada de 200-300 casos
4. Correr apenas os teachers finalistas na Validation completa
5. Escolher teacher, modo de thinking, prompt, parser e sampling
6. Gerar targets apenas sobre Train
7. Filtrar e auditar os targets
8. Treinar o student e acompanhar o treino com SFT Dev
9. Escolher checkpoint e thresholds com as benchmarks de Validation
10. Congelar todas as decisões
11. Avaliar uma vez no Internal Benchmark, Internal Test e externos
```

As previsões baseline do student podem ser geradas antecipadamente sobre o
teste selado para permitir comparação emparelhada, desde que permaneçam sem
consulta e não influenciem nenhuma decisão. Operacionalmente é mais simples
executar base student, trained student e teacher apenas depois de congelar o
protocolo.

## 7. Distillation sintética: teacher e student

Exemplo conceptual:

```text
Teacher: Qwen 3.7 Flash
Student: Qwen 3.5 4B
```

O teacher recebe uma imagem de Train e pode produzir:

- output final estruturado;
- findings visuais;
- descrição clínica;
- diferencial e classificação;
- evidência associada;
- reasoning, quando o endpoint o disponibilizar.

Devem ser guardados separadamente:

```text
teacher_reasoning_raw       -> auditoria/ablation
teacher_reasoning_summary   -> resumo quando fornecido pela API
teacher_final_raw           -> resposta original
teacher_final_canonical     -> resposta depois do parser permitido
student_training_target     -> supervisão realmente usada no fine-tuning
```

O reasoning bruto não deve ser automaticamente o target principal. A
configuração recomendada ensina um rationale clínico curto, estruturado e
verificável. Raw reasoning pode ser testado numa ablation separada.

## 8. Quatro famílias de dados sintéticos

O corpus sintético não deve conter apenas classificação imediata. Deve conter
quatro comportamentos:

| Família | Ação | Quando |
| --- | --- | --- |
| Diagnóstico direto | `CLASSIFY` | Evidência visual suficiente e ranking estável |
| Pedido de contexto | `ASK_CONTEXT` | Duas ou mais hipóteses continuam plausíveis e uma resposta pode distingui-las |
| Pedido de nova imagem | `REQUEST_BETTER_IMAGE` | Problema visual concreto: blur, luz, enquadramento, distância ou oclusão |
| Fora do domínio | `ABSTAIN_OUT_OF_DOMAIN` | Imagem não dermatológica ou doença fora da capacidade/taxonomia pretendida |

O student precisa de exemplos positivos e negativos de cada ação. Se o
dataset contiver demasiados `ASK_CONTEXT`, o modelo aprenderá a perguntar
sempre. Se contiver apenas `CLASSIFY`, aprenderá a diagnosticar mesmo quando a
evidência é insuficiente.

## 9. Como gerar exemplos `ASK_CONTEXT`

O pedido de contexto deve acontecer apenas quando a informação tem valor
para resolver o diferencial, e não simplesmente porque a confiança declarada
pelo teacher é baixa.

Fluxo sintético recomendado:

```text
Turno 1
  input: imagem
  teacher: findings + diferencial provisório
  decision: ASK_CONTEXT
  question: uma pergunta discriminativa

Turno 2
  input: imagem + pergunta + resposta do utilizador
  teacher: atualiza diferencial e decide CLASSIFY/ASK/ABSTAIN
```

Exemplo:

```json
{
  "action": "ASK_CONTEXT",
  "findings": ["erythematous scaly plaque"],
  "provisional_differential": [
    {"disease_id": "D003", "rank": 1},
    {"disease_id": "D009", "rank": 2}
  ],
  "question": {
    "question_id": "NEW_PRODUCT_CONTACT",
    "text": "A lesão começou depois do contacto com algum produto novo nessa zona?",
    "distinguishes": ["D003", "D009"]
  }
}
```

Depois da resposta:

```json
{
  "context": {"NEW_PRODUCT_CONTACT": true},
  "action": "CLASSIFY",
  "updated_differential": [
    {"disease_id": "D009", "rank": 1},
    {"disease_id": "D003", "rank": 2}
  ]
}
```

As perguntas devem vir de um catálogo controlado, por exemplo:

- comichão;
- dor ou ardor;
- duração;
- progressão;
- recorrência;
- localização e distribuição;
- contacto com produto/alergénio;
- medicamentos recentes;
- sintomas sistémicos;
- exposição solar;
- história pessoal ou familiar, apenas quando clinicamente relevante para
  as hipóteses presentes.

Perguntar se existe alguém na família com determinada doença só é adequado
quando essa informação altera plausivelmente o ranking. Não deve ser uma
pergunta genérica em todos os casos.

## 10. Origem das respostas simuladas

Uma resposta sintética sobre sintomas ou história clínica não pode ser
apresentada como facto real do paciente se o dataset não a contiver.

Prioridade de fontes:

1. metadata real do caso, como SCIN, PAD-UFES-20 ou HIBA;
2. anotação humana adicional;
3. cenário explicitamente marcado como contrafactual/sintético;
4. nunca inventar contexto e guardá-lo como se fosse metadata observada.

SCIN e PAD-UFES-20 são especialmente úteis para conversas simuladas porque
possuem sintomas e informação clínica. Derm1M possui captions e alguns campos
textuais, mas a sua proveniência e qualidade devem ser consideradas antes de
transformar uma frase em resposta factual do utilizador.

## 11. Como decidir quando perguntar

Uma política futura pode combinar:

- diferença entre Top-1 e Top-2;
- estabilidade entre múltiplas gerações;
- calibração medida na Validation;
- existência de uma pergunta capaz de separar os candidatos;
- qualidade da imagem;
- deteção de out-of-domain.

A regra conceptual é:

```text
se imagem insuficiente:
    REQUEST_BETTER_IMAGE
senão se fora do domínio:
    ABSTAIN_OUT_OF_DOMAIN
senão se ranking estável e evidência suficiente:
    CLASSIFY
senão se existe pergunta com valor discriminativo:
    ASK_CONTEXT
senão:
    ABSTAIN_OUT_OF_DOMAIN
```

Os thresholds não devem ser escolhidos usando o Internal Benchmark. Devem ser
aprendidos/calibrados na Validation e congelados antes do teste final.

## 12. Avaliação futura da interação

Ensinar `ASK_CONTEXT` exige uma benchmark interativa separada. Em cada caso, o
avaliador esconde o contexto e responde apenas à pergunta escolhida pelo
modelo. Devem ser medidos:

- accuracy antes e depois do contexto;
- percentagem de perguntas que corrigem o Top-1;
- taxa de perguntas desnecessárias;
- relevância da pergunta para o diferencial;
- número médio de perguntas;
- repetição de perguntas;
- custo e latência adicionais;
- taxa de classificação correta sem pedir contexto;
- calibração e abstenção.

As comparações principais devem ser:

```text
imagem apenas
imagem + todo o contexto
imagem + contexto pedido adaptativamente
```

Isto permite demonstrar não apenas que contexto ajuda, mas que o small model
aprendeu **quando perguntar**, **o que perguntar** e **quando já possui
evidência suficiente para responder**.
