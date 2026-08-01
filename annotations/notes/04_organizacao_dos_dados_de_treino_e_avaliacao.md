# Organização dos dados de treino e avaliação

## Regra principal

Existem duas famílias de dados com funções diferentes:

1. **Dados de treino e desenvolvimento**, que podem influenciar o modelo.
2. **Dados de avaliação selada**, que servem apenas para medir o resultado.

Não se deve juntar tudo num único `train.parquet`, porque isso criaria
data leakage e tornaria impossível defender cientificamente a comparação
antes/depois do fine-tuning.

## Fluxo completo

```text
Fontes originais
    |
    +-- split interno já congelado -------------------------------+
    |       |                                                     |
    |       +-- train original: 6.417 imagens                     |
    |       +-- validation: 1.683 imagens                         |
    |       +-- internal test: 1.722 imagens (selado)             |
    |               +-- internal benchmark: 1.000 casos          |
    |               +-- reserve: 63 grupos restantes             |
    |                                                             |
    +-- Derm1M clínico + HIBA clínico                              |
            |                                                      |
            +-- filtro de modalidade clínica                       |
            +-- deduplicação contra avaliação                      |
            +-- treino aumentado: 81.787 imagens <-----------------+

Avaliação externa, sempre separada:
    +-- DDI: 300 imagens
    +-- SkinDisNet: 1.365 imagens
```

## O que significa cada conjunto

| Conjunto | Pode treinar o modelo? | Pode escolher prompts/modelo/thresholds? | Uso |
| --- | --- | --- | --- |
| `train_images.parquet` | Sim | Sim, indiretamente | Fine-tuning e geração de targets pelo teacher |
| `validation.parquet` | Não como dados de treino | Sim | Escolher teacher, prompt, hiperparâmetros, checkpoints e thresholds |
| `internal_test.parquet` | Não | Não | Avaliação interna completa depois de congelar as decisões |
| `internal_benchmark_1000.parquet` | Não | Não | Comparação final emparelhada antes/depois em 1.000 casos independentes |
| `internal_test_reserve.parquet` | Não | Não | Grupos do internal test fora dos 1.000 casos principais |
| `external_ddi.parquet` | Não | Não | Generalização para outra distribuição clínica |
| `external_skindisnet.parquet` | Não | Não | Generalização para smartphone/dados externos |

O `internal_benchmark_1000` não é um novo split independente: é uma vista
selada e balanceada do `internal_test`, com um caso por grupo selecionado. Por
isso, não se deve avaliar repetidamente nele enquanto se ajustam prompts ou
modelos.

## O novo treino aumentado

O ficheiro
`data/training/dermatology_multimodal_v1/train_images.parquet` contém:

- 6.417 imagens do treino interno original;
- 75.015 fotografias clínicas elegíveis do Derm1M;
- 355 fotografias clínicas do HIBA;
- total de 81.787 imagens.

Os novos dados nunca foram colocados em validation, internal test, internal
benchmark ou external. Antes de entrarem no treino, foram comparados com esses
conjuntos através de SHA-256 e perceptual hash.

Foram excluídos 2.869 candidatos Derm1M:

- 1.367 duplicados exatos já existentes no treino;
- 1.502 candidatos perceptualmente semelhantes a imagens protegidas de
  avaliação.

## Nem todas as 81.787 imagens têm o mesmo target

O campo `training_role` separa três utilizações:

| Papel | Imagens | Significado |
| --- | ---: | --- |
| `in_domain_diagnosis` | 18.914 | Diagnóstico mapeado diretamente para uma das 21 classes |
| `out_of_domain` | 45.355 | Diagnóstico dermatológico fora das 21 classes |
| `description_only` | 17.518 | Fotografia clínica com descrição, mas sem diagnóstico definitivo seguro |

Isto significa que não se deve usar as 81.787 imagens como se todas fossem
exemplos de classificação closed-set. Os três papéis permitem construir
targets diferentes:

- `in_domain_diagnosis`: descrição, findings e diagnóstico;
- `out_of_domain`: descrição e comportamento de fora do domínio/taxonomia;
- `description_only`: descrição visual e morphology grounding, sem inventar
  um diagnóstico.

`teacher_annotation_queue.parquet` ainda não é o dataset SFT final. É a fila
de imagens elegíveis que receberá targets estruturados depois de escolher e
validar o teacher.

## Sequência experimental recomendada

1. Avaliar os teachers/modelos candidatos em `validation.parquet`.
2. Escolher teacher, prompt, parâmetros de geração e parser.
3. Gerar targets para `teacher_annotation_queue.parquet`.
4. Validar automaticamente o schema e auditar manualmente uma amostra.
5. Fazer fine-tuning com os targets aceites.
6. Escolher o checkpoint e thresholds apenas com validation.
7. Congelar todas as decisões.
8. Medir uma única vez no `internal_benchmark_1000`.
9. Reportar também internal test completo e os externos DDI/SkinDisNet.
10. Comparar teacher e student nos mesmos casos, com as mesmas métricas e
    intervalos de confiança.

As diferentes benchmarks são vistas/tarefas sobre conjuntos protegidos, não
novos dados de treino. `visual_top_k`, `visual_confusion_sets` e
`evidence_grounded_diagnosis` medem capacidades diferentes e podem reutilizar
uma imagem sob contratos de scoring diferentes sem autorizar o seu uso no
fine-tuning.

A distinção entre benchmark e evaluation set, a construção das versões de
Validation e o plano de dados sintéticos interativos estão detalhados em
`annotations/notes/05_selecao_do_teacher_validacao_das_benchmarks_e_dados_interativos.md`.
