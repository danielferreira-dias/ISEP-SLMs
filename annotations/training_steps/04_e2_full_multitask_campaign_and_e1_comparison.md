# E2 completo: supervisão multitarefa e comparação com E1 label-only

Data de conclusão: 16 de agosto de 2026  
Estado: **ANALYZED**; campanha e preservação concluídas, uma seed por condição  
Decisão: **conservar E2 como resultado informativo, mas não o tratar como uma
melhoria global sobre E1; corrigir os targets e o contrato multitarefa antes de
E3**

## Material Passport

- **Objeto:** treino E2 do Qwen 3.5 4B com diagnóstico, morfologia SKINCON e
  descrições clínicas derivadas de SkinCAP.
- **Base comum:**
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Condições:** Frozen Vision e Vision LoRA, iniciadas independentemente da
  mesma base.
- **Dados:** `ISEPDermData/e1_label_v1` para diagnóstico e
  `ISEPDistillDataset/isep_distill_dataset_v0.4.1` na revisão
  `b215f0474e4931b5951da768e79a0d579d26919d` para morfologia e caption.
- **Manifest E2:** SHA-256
  `8638a3fea6ed49875359a9e8e1781f104e8350bae4edafb250ff58a4ef35ce13`.
- **Treino:** seed 42, três épocas, 4 557 updates, LR `1e-4`, BF16, LoRA
  `r=16`, `alpha=16`, microbatch 2, acumulação 4, sem augmentation e com
  thinking desativado.
- **Diferença controlada:** apenas a inclusão das camadas visuais nos targets
  LoRA; Frozen tem 32 464 896 parâmetros treináveis e Vision 38 756 352.
- **Seleção:** exclusivamente macro-F1 de diagnóstico em `sft_dev`, com
  balanced accuracy, eval loss e checkpoint mais cedo como desempates.
- **Verificação:** manifests `completed`, três checkpoints válidos por
  condição, seis uploads com commits e tree hashes no repositório privado
  `danielfdias98/ISEP-training-checkpoints`, métricas e predições preservadas
  localmente.
- **Limite de inferência:** E1 usou seed 3407 e E2 seed 42; as diferenças entre
  fases são descritivas e não identificam causalmente o efeito da supervisão
  multitarefa.

## 1. Pergunta experimental

E1 ensinou apenas a classe diagnóstica. E2 testou se adicionar supervisão
perceptual e descritiva produziria um student mais visualmente fundamentado sem
perder capacidade diagnóstica. O desenho comparou a mesma ablação arquitetural
de E1:

1. **Frozen Vision:** LoRA apenas fora das camadas visuais;
2. **Vision LoRA:** LoRA também nas camadas visuais.

E2 não é knowledge distillation de logits ou features. É SFT multitarefa com
targets humanos/derivados de datasets existentes e constitui uma etapa de
preparação para a distilação hard do E3.

## 2. Mistura de treino e avaliação

| Tarefa | `sft_train` | `sft_dev` | Fração do treino |
|---|---:|---:|---:|
| Diagnóstico, 21 classes | 6 312 | 1 229 | 52,0% |
| Morfologia, 48 conceitos | 3 068 | 527 | 25,3% |
| Caption clínica | 2 767 | 483 | 22,8% |
| **Total de linhas de treino** | **12 147** | — | **100%** |

A mistura quase duplicou o número de linhas/updates relativamente ao treino
label-only, mantendo LR e número de épocas. Cada tarefa tinha um contrato de
saída distinto: uma label para diagnóstico, JSON de conceitos para morfologia
e uma frase curta sem diagnóstico para caption.

## 3. Checkpoint selecionado em E2

O checkpoint da terceira época, `checkpoint-4557`, foi corretamente escolhido
em ambas as condições pelo protocolo pré-definido de `sft_dev`. Nenhum resultado
de ISEPDermaBench foi usado para escolher época ou condição.

| Métrica no checkpoint 4 557 | E2 Frozen | E2 Vision |
|---|---:|---:|
| Diagnóstico Top-1 | 59,24% | **59,89%** |
| Diagnóstico macro-F1 | 56,68% | **58,56%** |
| Diagnóstico balanced accuracy | 55,71% | **57,74%** |
| Eval loss | 0,5294 | **0,5060** |
| Morfologia micro-F1 | 73,23% | **73,61%** |
| Morfologia macro-F1 | 31,10% | **35,47%** |
| Morfologia exact match | **33,02%** | 31,88% |
| Caption task score | 21,45% | **21,83%** |
| Caption concept-F1 | 13,31% | **13,33%** |
| Caption reference similarity | 37,80% | **38,07%** |
| Caption clinical compliance | 13,25% | **14,08%** |
| Macro task score, não accuracy | 36,41% | **38,62%** |

Dentro de E2, Vision menos Frozen corresponde a +0,65 pp de Top-1, +1,88 pp
de macro-F1 e +2,04 pp de balanced accuracy. Esta direção é compatível com uma
vantagem da adaptação visual, mas uma única seed não permite uma afirmação
confirmatória.

## 4. Trajetória por época

| Condição | Época | Diagnóstico macro-F1 | Morfologia macro-F1 | Caption score |
|---|---:|---:|---:|---:|
| Frozen | 1 | 47,49% | 28,33% | 19,14% |
| Frozen | 2 | 53,88% | 29,92% | **21,93%** |
| Frozen | 3 | **56,68%** | **31,10%** | 21,45% |
| Vision | 1 | 44,39% | 26,42% | 17,67% |
| Vision | 2 | 56,13% | 33,25% | **22,10%** |
| Vision | 3 | **58,56%** | **35,47%** | 21,83% |

Diagnóstico e morfologia continuaram a melhorar até à terceira época. Caption
atingiu o máximo na segunda e desceu ligeiramente na terceira. Isto não torna a
seleção incorreta: o protocolo declarava diagnóstico macro-F1 como objetivo de
seleção. Mostra, sim, que uma única métrica de seleção não protege
automaticamente todos os comportamentos multitarefa.

## 5. Comparação descritiva E1–E2 em `sft_dev`

| Condição | Fase | Top-1 | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| Frozen | E1 label-only | **60,05%** | **58,07%** | **56,88%** |
| Frozen | E2 multitarefa | 59,24% | 56,68% | 55,71% |
| Vision | E1 label-only | **61,51%** | **62,14%** | **62,03%** |
| Vision | E2 multitarefa | 59,89% | 58,56% | 57,74% |

As diferenças E2 menos E1 foram:

- Frozen: -0,81 pp Top-1, -1,39 pp macro-F1 e -1,18 pp balanced accuracy;
- Vision: -1,63 pp Top-1, -3,58 pp macro-F1 e -4,28 pp balanced accuracy.

Estas comparações não isolam o efeito do E2 porque as seeds diferem e apenas
uma execução por fase foi realizada. Permitem afirmar que **não foi observada
uma melhoria diagnóstica de desenvolvimento** que compensasse, por si só, o
custo adicional.

## 6. Custo computacional

| Recurso | E1 Frozen | E2 Frozen | E1 Vision | E2 Vision |
|---|---:|---:|---:|---:|
| GPU-hours | 1,699 | 3,239 | 2,119 | 3,741 |
| Peak VRAM | 14,88 GiB | 17,97 GiB | 15,68 GiB | 18,03 GiB |
| Energia E2 | — | 710,97 Wh | — | 780,89 Wh |

E2 consumiu aproximadamente +90,7% GPU-hours em Frozen e +76,6% em Vision.
Peak VRAM aumentou cerca de 20,8% e 15,0%, respetivamente. Os recursos compram
duas capacidades que E1 não treinava — morfologia estruturada e descrição —,
mas não compraram melhor diagnóstico em `sft_dev` nesta execução.

## 7. O que funcionou

- A campanha foi tecnicamente estável: não houve OOM, NaN, interrupção ou
  fallback silencioso.
- O modelo base não dominava o schema de morfologia; E2 passou a produzir o
  formato e alcançou 73% de micro-F1.
- Vision LoRA terminou à frente de Frozen em macro-F1 diagnóstico,
  balanced accuracy, morfologia macro-F1 e score global.
- A seleção ficou isolada dos benchmarks finais, evitando leakage de model
  selection.
- Os seis checkpoints foram preservados com hashes e commits verificáveis.

## 8. O que provavelmente correu mal

### 8.1 Targets de caption incompletos

Uma auditoria posterior aos 3 250 captions de v0.4.1 encontrou 1 315 sem
pontuação terminal ou semanticamente incompletos (40,46%). Entre os sufixos
problemáticos estavam 234 casos terminados em `It is` e 374 em `which is`.
Exemplos aceites incluíam “Well-defined white patches. It is” e “This picture
does not allow for an accurate”.

O mecanismo mais plausível é o corte `skincap_observation_prefix_v1` numa
fronteira temática em vez de numa fronteira de frase. A associação é forte —
os outputs E2 reproduzem finais abruptos —, mas continua a ser uma hipótese de
mecanismo, não uma demonstração causal sem ablação limpa.

### 8.2 Interferência entre contratos de saída

Quase 48% das linhas ensinavam explicitamente a não diagnosticar: morfologia
pedia JSON observacional e caption pedia descrição curta. Nenhuma tarefa
ensinava conjuntamente descrição, diferencial ordenado e justificação. É
plausível que o student tenha aprendido “descrever e terminar” como padrão
forte, comportamento depois observado no benchmark aberto.

### 8.3 Dose de otimização

E2 realizou cerca de 1,93 vezes mais updates que E1, com a mesma LR inicial e
três épocas. Uma dose excessiva de SFT sobre targets heterogéneos é outra
explicação possível para especialização excessiva ou esquecimento. Não existe
uma ablação que separe quantidade de updates, composição da mistura e qualidade
dos captions.

## 9. Consequência para E3

E3 não deve simplesmente adicionar respostas de teacher à mistura E2 v0.4.1.
Antes do treino:

1. reconstruir captions apenas a partir de targets completos e validados;
2. usar tokens/prompts de tarefa inequivocamente distintos;
3. introduzir um target combinado de descrição + diferencial + evidência;
4. equilibrar por tokens/updates, não apenas por número de linhas;
5. selecionar por um painel `sft_dev` que inclua diagnóstico, grounding e
   completude do output, mantendo o benchmark externo fora da seleção;
6. incluir um teste open-ended pequeno e interno ao dev para detetar regressão
   de formato antes do full benchmark;
7. manter Frozen e Vision como ablação pareada até existir evidência multi-seed.

O desenho teacher em duas etapas — Stage A observacional e Stage B diferencial
fundamentado — permanece precisamente porque separa perceção de diagnóstico e
permite validar cada componente antes de o converter em targets do student.

## 10. Artefactos preservados

- [Relatório E2 Frozen](../../outputs/training/e2_skincon_skincap_frozen_vision/full-l40s-e2-skincap-frozen-seed42-20260816/report/thesis_summary.md)
- [Relatório E2 Vision](../../outputs/training/e2_skincon_skincap_unsloth_all/full-l40s-e2-skincap-vision-seed42-20260816/report/thesis_summary.md)
- [Manifest de uploads Frozen](../../outputs/training/e2_skincon_skincap_frozen_vision/full-l40s-e2-skincap-frozen-seed42-20260816/manifests/checkpoint_uploads.json)
- [Manifest de uploads Vision](../../outputs/training/e2_skincon_skincap_unsloth_all/full-l40s-e2-skincap-vision-seed42-20260816/manifests/checkpoint_uploads.json)
- [Comparação E1–E2 na ISEPDermaBench](../benchmarks/14_e1_vs_e2_internal_benchmark.md)

## 11. Limitações

- Uma seed por fase e seeds diferentes impedem inferência causal E1–E2.
- O caption score automático não mede por completo coerência clínica ou
  completude discursiva.
- O macro task score é uma média de métricas heterogéneas e não uma accuracy.
- O treino avalia comportamento em datasets internos; não estabelece segurança
  clínica, eficácia em doentes ou generalização externa.
- As hipóteses sobre truncagem, interferência e dose de treino foram formuladas
  após observar o benchmark e devem ser testadas prospectivamente em E3.
