# MedSigLIP zero-shot na classificação da ISEPDermaBench

Data de conclusão: 17 de agosto de 2026  
Estado: **VERIFIED**  
Conclusão: **MedSigLIP é competitivo quando recebe três candidatos, mas fica
substancialmente abaixo dos students especializados no ranking global de 21
doenças**

## Material Passport

- **Origin Skill:** `experiment-agent`
- **Origin Mode:** `run`
- **Origin Date:** 2026-08-17
- **Verification Status:** **VERIFIED**
- **Version Label:** `medsiglip_internal_v1`
- **Modelo:** [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)
- **Revisão imutável:** `9cea28a1a1195f665105faa6e8544c112fd960a4`
- **Identidade local:** a cache do Hugging Face confirmou o repositório, a
  revisão e um snapshot de aproximadamente 3,5 GB.
- **Dados externos:** nenhum caso, referência ou output foi enviado para API,
  judge ou serviço externo.

## 1. Objetivo e âmbito

Esta experiência mede se um encoder imagem-texto médico, sem fine-tuning na
ISEPDermaBench, consegue executar apenas as duas tarefas determinísticas de
classificação de doença:

1. `visual_top_k_closed_set`: 1 000 imagens, ranking de seis entre 21 doenças;
2. `visual_disease_confusion_sets`: 828 tarefas sobre 414 imagens, ranking de
   três candidatos em condições emparelhadas de baixa e alta confusabilidade.

MedSigLIP não é um modelo generativo. A experiência não abrange descrição
morfológica, evidence-grounded diagnosis ou diagnóstico aberto, e não utiliza
LLM-as-a-judge. O objetivo é avaliar o valor do encoder como classificador
zero-shot ou componente auxiliar de retrieval/reranking, não como sistema
clínico autónomo.

## 2. Protocolo congelado

| Elemento | Valor |
|---|---|
| Arquitetura | SigLIP dual encoder |
| Precisão | FP32 |
| Resolução | 448 px |
| Dispositivo | Apple MPS |
| Batch | 2 |
| Seed registada | 42 |
| Prompt | `a clinical photograph of {display_name}` |
| Seleção do prompt | A priori; não otimizado na Internal Benchmark |
| Ranking | Similaridade cosseno entre embeddings L2-normalizados |
| Sampling | Não aplicável; ranking determinístico |
| Text length | 64 tokens |
| Imagens únicas | 1 000 |

Os nomes das 21 doenças foram obtidos da taxonomia congelada. O mesmo
embedding de cada imagem foi reutilizado nas duas tarefas quando aplicável. No
Top-K, todas as 21 labels foram pontuadas e foram devolvidos os seis maiores
scores. Nos confusion sets, o ranking foi restringido aos três candidatos
fornecidos por cada tarefa. Não houve prompt ensembling, calibração, answer
repair, regeneration ou escolha pós-hoc de template.

Comando da execução completa:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_medsiglip_internal_benchmark.py \
  --stage full --device mps --batch-size 2
```

## 3. Resultados principais

### 3.1 Ranking fechado de 21 doenças, n=1 000

| Métrica | Resultado |
|---|---:|
| Top-1 accuracy | **38,60%** |
| Top-3 accuracy | **68,10%** |
| Top-6 accuracy | **84,20%** |
| MRR | **54,58%** |
| Macro-F1 Top-1 | **35,70%** |

O Top-6 elevado mostra que a representação visual frequentemente coloca a
doença de referência numa shortlist útil. O Top-1 de 38,60% mostra, contudo,
que a similaridade direta com um único prompt de label não é suficiente para
resolver de forma robusta a taxonomia completa.

### 3.2 Confusion sets, n=828

| Métrica | Resultado |
|---|---:|
| Top-1 accuracy | **76,21%** |
| Top-2 accuracy | **91,43%** |
| MRR | **86,67%** |
| Macro-F1 Top-1 | **75,22%** |
| Macro set Top-1 | **75,99%** |
| Baixa confusabilidade Top-1 | **89,86%** |
| Alta confusabilidade Top-1 | **62,56%** |
| Gap low-high | **27,29 pp** |
| IC do gap | **22,22–32,37 pp** |

O modelo é muito mais competitivo quando recebe uma shortlist de três doenças.
O gap emparelhado de 27,29 pp confirma, porém, que a classificação se degrada
precisamente quando os candidatos foram construídos para serem visualmente
semelhantes.

Entre as doenças presentes nos confusion sets, os maiores Top-1 foram rosácea
(96,67%), urticária (93,48%) e melanoma (90,00%). Os menores foram drug
eruption (39,13%) e eczema (47,06%). Estes resultados por doença são
descritivos e dependem da composição dos respetivos confusion sets.

### 3.3 Sinal por tom de pele

- cobertura de metadata de tom de pele: 95,50%;
- Top-1 do pior grupo com suporte estatístico: 29,55%;
- gap Top-1 entre grupos suportados: 14,79 pp.

O gap é um sinal de heterogeneidade que requer auditoria adicional. Não permite
atribuir causalidade ao tom de pele, porque fonte, doença, sistema de anotação e
composição dos grupos podem atuar como confundidores.

## 4. Comparação com E1 e E2

As métricas de ranking são comparáveis ao nível dos mesmos cohorts, IDs,
candidatos e referências. O mecanismo de resposta é diferente: MedSigLIP
ordena scores de um dual encoder, enquanto os Qwen especializados geram o
output estruturado. Latência, energia e memória **não** são comparadas, porque
MedSigLIP foi executado em Apple MPS/FP32 e E1/E2 em NVIDIA L40S/BF16 com vLLM.

### 4.1 Visual Top-K

| Modelo | Top-1 | Top-3 | Top-6 | MRR |
|---|---:|---:|---:|---:|
| MedSigLIP zero-shot | 38,60% | 68,10% | 84,20% | 54,58% |
| E1 Frozen | 50,20% | 80,00% | 89,50% | 65,46% |
| **E1 Vision LoRA** | **56,20%** | **84,10%** | **92,60%** | **70,55%** |
| E2 Frozen | 49,30% | 77,40% | 88,80% | 64,39% |
| E2 Vision LoRA | 52,40% | 80,30% | 90,80% | 67,18% |

MedSigLIP ficou 11,60 pp abaixo da condição student mais fraca e 17,60 pp
abaixo do melhor student em Top-1. A especialização supervisionada continua a
ser importante quando o modelo precisa escolher globalmente entre 21 doenças.

### 4.2 Disease Confusion Sets

| Modelo | Top-1 | Top-2 | MRR |
|---|---:|---:|---:|
| MedSigLIP zero-shot | 76,21% | 91,43% | 86,67% |
| E1 Frozen | 76,57% | 89,86% | 85,47% |
| **E1 Vision LoRA** | **81,40%** | 91,67% | 88,71% |
| E2 Frozen | 80,07% | **93,24%** | **88,87%** |
| E2 Vision LoRA | 79,95% | 92,51% | 88,53% |

Aqui MedSigLIP aproximou-se do E1 Frozen: -0,36 pp em Top-1, +1,57 pp em
Top-2 e +1,20 pp em MRR. Isto sustenta o uso potencial do encoder para
shortlisting ou reranking, mas não demonstra equivalência clínica nem
substituição do modelo multimodal especializado.

## 5. Auditoria de possível sobreposição de fontes

O protocolo marca `SCIN` e `PAD_UFES_20` como fontes com sobreposição de
dataset conhecida ao nível da origem e usa `Fitzpatrick17k_C` como estrato de
menor sobreposição conhecida. Esta classificação não prova memorização de uma
imagem concreta, e o estrato inferior não é garantidamente contamination-free.

| Tarefa | Casos em fontes marcadas | Taxa | Estrato de menor sobreposição | Top-1 global | Top-1 estrato |
|---|---:|---:|---:|---:|---:|
| Visual Top-K | 455/1 000 | 45,50% | 545 | 38,60% | 37,61% |
| Confusion Sets | 346/828 | 41,79% | 482 | 76,21% | 76,14% |

A proximidade entre o resultado global e o estrato de menor sobreposição é
compatível com desempenho não explicado apenas pelas fontes marcadas. Não é,
por si só, uma prova de ausência de contaminação.

## 6. Execução, incidentes e validação

- CPU smoke: concluído;
- CPU gate: 10 Top-K e 20 confusion tasks, concluído;
- MPS smoke: rankings exatamente iguais ao CPU; diferença máxima de score
  entre `1,94e-7` e `3,80e-7`;
- MPS gate com batch 2: concluído;
- full MPS: concluído em 1 128,41 s, exit code 0;
- previsões: 1 000 Top-K e 828 confusion tasks, todas `status=ok`;
- pares: 414/414 com condições low/high completas;
- sem task IDs, candidatos ou ranks duplicados;
- referências sempre contidas nos candidatos;
- todos os scores finitos e os manifests/checksums válidos;
- testes finais: 22 testes e 2 subtests passaram; `ruff` sem erros.

Ocorreram três incidentes transparentemente preservados:

1. o primeiro smoke falhou antes da inferência por uma resolução online
   indevida de `adapter_config` apesar de `local_files_only`; o runner passou a
   resolver diretamente o snapshot imutável local;
2. o segundo smoke falhou antes da inferência de imagens porque a versão de
   Transformers devolveu `BaseModelOutputWithPooling`; foi adicionada extração
   explícita de `pooler_output`;
3. após o full, o relatório de sobreposição apresentou zero casos devido a uma
   diferença de capitalização (`SCIN` versus `scin`). A normalização foi
   corrigida e apenas os relatórios determinísticos foram recalculados a partir
   das previsões existentes. A inferência não foi repetida, as métricas
   primárias permaneceram idênticas e os relatórios anteriores foram
   preservados em `reporting_history/20260817T002710Z`.

Hashes SHA-256 das previsões finais:

- confusion sets: `069c326e55a7523ba8ca52a215d1d12199c20cecb524b4af2a8714b3786b4205`;
- Top-K: `45ba2fb8a06d090af6d9c7a6168134d9857698f783bdcadbd03c9f106753c73b`.

## 7. Interpretação defensável para a tese

### Evidência

- Um encoder médico zero-shot consegue produzir shortlists relevantes sem
  treino específico na taxonomia: 84,20% Top-6 no universo de 21 classes.
- A restrição do espaço de decisão para três candidatos eleva o Top-1 para
  76,21%, próximo do E1 Frozen.
- Todos os students especializados superam MedSigLIP no Top-1 global, mostrando
  valor adicional do treino supervisionado na tarefa-alvo.
- A queda low-to-high mostra que shortlist não elimina a dificuldade clínica
  das doenças visualmente semelhantes.

### Inferência

MedSigLIP é mais promissor como **componente auxiliar** do que como substituto
do student: pode recuperar candidatos, fornecer uma representação visual
médica ou atuar como sinal adicional para routing e abstention. Esta hipótese
requer uma experiência posterior que compare o student isolado com uma
pipeline híbrida, usando a mesma GPU e sem selecionar thresholds na test set.

## 8. Limitações

- Existe apenas uma execução e um único prompt template.
- Não foi testado prompt ensembling, sinónimos, calibração ou fine-tuning de
  MedSigLIP; fazê-lo após observar estes resultados exigiria um dev set
  separado.
- Os scores de cosseno não são probabilidades clínicas calibradas.
- O benchmark é interno e closed-set; não mede doenças fora da taxonomia,
  pedido de contexto, rationale, grounding textual ou segurança clínica.
- As métricas de schema/JSON são serialização interna determinística e não são
  comparáveis à fiabilidade de output de modelos generativos.
- O gap por tom de pele pode refletir confundimento por doença e fonte.
- A auditoria de sobreposição é ao nível da fonte, não uma pesquisa de
  duplicados exatos contra os dados de pretraining.
- A diferença para E1/E2 é descritiva; não foi calculado um teste pareado nem
  correção por múltiplas comparações nesta anotação.

## 9. Artefactos

- [Run completo](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/)
- [Manifest](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/campaign_manifest.json)
- [Métricas](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/metrics.json)
- [Auditoria de fontes](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/source_overlap_audit.json)
- [Previsões Top-K](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/predictions/visual_top_k.jsonl)
- [Previsões confusion sets](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/predictions/visual_confusion_sets.jsonl)
- [Checksums](../../outputs/medsiglip_internal_benchmark/full/20260817T000550Z/checksums.sha256)
- [Implementação](../../src/benchmark/medsiglip.py)
- [Runner e modo de rescore](../../scripts/run_medsiglip_internal_benchmark.py)
- [Testes](../../tests/test_medsiglip_benchmark.py)
- [Comparação E1 versus E2](14_e1_vs_e2_internal_benchmark.md)
