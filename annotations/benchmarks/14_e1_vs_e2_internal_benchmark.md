# ISEPDermaBench: comparação E1 label-only versus E2 multitarefa

Data de conclusão: 16 de agosto de 2026  
Estado: **ANALYZED**; inferência e single-judge completos  
Conclusão: **E2 melhorou reconhecimento de achados visuais, mas regrediu em
diagnóstico aberto e não superou E1 como modelo final**

## Material Passport

- **Modelos:** E1 Frozen, E1 Vision LoRA, E2 Frozen e E2 Vision LoRA, todos
  derivados da mesma revisão do Qwen 3.5 4B.
- **Cohorts congelados por modelo:** Visual Top-K 1 000; Disease Confusion 828;
  Evidence-grounded Diagnosis 134; Open-ended Diagnosis 300; total 2 262.
- **Protocolo:** temperatura 0,6, `top_p=0.95`, `top_k=20`,
  `presence_penalty=1.5`, seed 42, thinking desativado, BF16, mesmos IDs,
  prompts, parsers e referências; inválidos no denominador, sem regeneration ou
  answer repair.
- **Hardware/backend E2:** uma NVIDIA L40S e vLLM local, condições executadas
  sequencialmente.
- **Judge aberto:** `gpt_5_6_luna`, Azure, batch 8, prompt e schema congelados,
  single judge, sem fallback; envio feito apenas após autorização explícita.
- **Cobertura E2 do judge:** 300/300 em cada condição, sem outputs inválidos,
  recusas ou falhas.
- **Checkpoint selection:** nenhum resultado desta página foi usado para
  selecionar checkpoints; os dois E2 já estavam fixados por `sft_dev`.
- **Verificação:** **ANALYZED**, não `VERIFIED`; não há replicação multi-seed nem
  segundo judge independente.

## 1. Resultado determinístico

### 1.1 Visual Top-K Closed Set, n=1 000

| Modelo | Top-1 | Top-3 | Top-6 | MRR |
|---|---:|---:|---:|---:|
| E1 Frozen | **50,20%** | **80,00%** | **89,50%** | **65,46%** |
| E2 Frozen | 49,30% | 77,40% | 88,80% | 64,39% |
| E1 Vision | **56,20%** | **84,10%** | **92,60%** | **70,55%** |
| E2 Vision | 52,40% | 80,30% | 90,80% | 67,18% |

E2 menos E1 foi -0,9 pp de Top-1 em Frozen e -3,8 pp em Vision. Na análise
pareada exploratória, a diferença Top-1 foi compatível com acaso em Frozen
(`p=0,543`) e mais difícil de explicar por acaso em Vision (`p=0,0062`). Estes
valores não foram corrigidos por múltiplas comparações.

### 1.2 Visual Disease Confusion Sets, n=828

| Modelo | Top-1 | Top-2 | MRR |
|---|---:|---:|---:|
| E1 Frozen | 76,57% | 89,86% | 85,47% |
| **E2 Frozen** | **80,07%** | **93,24%** | **88,87%** |
| **E1 Vision** | **81,40%** | 91,67% | **88,71%** |
| E2 Vision | 79,95% | **92,51%** | 88,53% |

Frozen melhorou +3,50 pp Top-1 e +3,38 pp Top-2; os testes pareados
exploratórios deram `p=0,0087` e `p=0,0018`. Vision perdeu -1,45 pp Top-1 e
ganhou +0,85 pp Top-2; nenhuma diferença foi conclusiva na análise exploratória.

### 1.3 Evidence-grounded Diagnosis, n=134

| Modelo | Diagnóstico Top-1 | Top-3 | Finding-F1 | Similaridade semântica |
|---|---:|---:|---:|---:|
| E1 Frozen | **55,97%** | **79,10%** | 44,21% | 32,09% |
| E2 Frozen | 42,54% | 70,15% | **65,48%** | **51,49%** |
| E1 Vision | **58,21%** | **81,34%** | 47,78% | 35,82% |
| E2 Vision | 46,27% | 71,64% | **67,91%** | **50,75%** |

Este é o trade-off central de E2. O diagnóstico Top-1 caiu -13,43 pp em
Frozen e -11,94 pp em Vision; os testes pareados exploratórios deram
`p=0,0021` e `p=0,0070`. Em simultâneo, finding-F1 aumentou +21,27 pp e
+20,13 pp, e a similaridade semântica dos achados aumentou +19,40 pp e
+14,93 pp. E2 aprendeu a mencionar melhor o que vê, mas ligou pior esses
achados à label pedida.

## 2. Open-ended Diagnosis, n=300

### 2.1 Resultado do judge

| Modelo | Cobertura | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|---:|
| E1 Frozen | 297/300 | 41,08% | **68,35%** | **53,25%** |
| E2 Frozen | 300/300 | 5,67% | 8,67% | 7,00% |
| E1 Vision | 297/300 | **41,75%** | 67,34% | 52,97% |
| E2 Vision | 300/300 | 10,33% | 17,33% | 13,61% |

A regressão não resultou de falhas do judge ou de cobertura: E2 teve 100% de
cobertura. O conteúdo preservado raramente apresentou os três diagnósticos
ordenados exigidos pelo protocolo.

### 2.2 Componentes do julgamento, escala 0–4 salvo taxa

| Modelo | Diagnóstico | Achados visuais | Grounding | Rationale | Diferencial | Unsupported claims |
|---|---:|---:|---:|---:|---:|---:|
| E1 Frozen | 2,374 | **2,596** | 2,559 | 1,869 | 1,835 | 72,39% |
| E2 Frozen | 0,307 | 2,300 | **2,893** | 0,207 | 0,247 | 42,67% |
| E1 Vision | 2,337 | **2,525** | 2,380 | 1,943 | 1,993 | 81,82% |
| E2 Vision | 0,607 | 2,317 | **2,880** | 0,377 | 0,393 | 42,33% |

Não se deve interpretar a menor taxa de unsupported claims como ganho de
segurança isolado. Os outputs E2 eram muito mais curtos e frequentemente não
chegavam a formular diagnóstico ou rationale; menos afirmações criam menos
oportunidades para uma afirmação não suportada.

## 3. Auditoria do comportamento de saída

| Modelo | Caracteres médios | Tokens médios | Três ranks presentes | Sem rank markers | Rank 0 |
|---|---:|---:|---:|---:|---:|
| E1 Frozen | 680,95 | 144,89 | 180/300 | 65/300 | 94/297 |
| E2 Frozen | 141,78 | 34,40 | 46/300 | 252/300 | 274/300 |
| E1 Vision | 770,06 | 163,10 | 191/300 | 55/300 | 97/297 |
| E2 Vision | 161,19 | 38,64 | 68/300 | 214/300 | 248/300 |

E2 Frozen terminou 61 respostas literalmente em “It is”; E2 Vision terminou
72 da mesma forma. Todos os `finish_reason` foram `stop` e não houve terminação
por limite de tokens. O padrão é, portanto, compatível com EOS aprendido a
partir de targets truncados, e não com truncagem do servidor durante o
benchmark.

## 4. Síntese: o que correu bem e mal

### Evidência favorável

- E2 aumentou fortemente finding-F1 e similaridade semântica na tarefa
  evidence-grounded.
- E2 Frozen melhorou no cohort específico de confusion sets.
- Vision continuou melhor que Frozen dentro de E2 no diagnóstico aberto e no
  `sft_dev`, sugerindo que adaptação visual ainda tem valor.
- A inferência teve cobertura completa, schema técnico válido e artefactos
  integrais; a regressão é comportamental, não uma falha operacional.

### Evidência desfavorável

- Ambos os E2 perderam Top-1 nas tarefas Visual Top-K e Evidence.
- O open-ended colapsou porque descrição curta substituiu o diferencial
  pedido; Vision atenuou, mas não resolveu, o problema.
- O treino nunca apresentou o contrato completo usado na pergunta aberta.
- Os captions incompletos de v0.4.1 oferecem um mecanismo plausível para o EOS
  prematuro.

A formulação defensável para a tese é: **supervisão perceptual adicional não
produz automaticamente raciocínio diagnóstico integrado**. E2 demonstrou uma
separação entre reconhecer/verbalizar achados e utilizá-los num diferencial
completo. Isso é um resultado negativo útil que motiva E3 com targets
teacher-generated, completos e explicitamente grounded.

## 5. Salvaguardas de interpretação estatística

1. **Simpson:** métricas agregadas podem esconder efeitos por classe, fonte ou
   tom de pele; essas estratificações continuam necessárias.
2. **Falácia ecológica:** desempenho médio não implica fiabilidade num caso
   clínico individual.
3. **Berkson/seleção:** os cohorts incluem apenas casos elegíveis e congelados;
   não representam toda a população dermatológica.
4. **Collider:** E1 teve três respostas inválidas; a comparação conserva-as no
   denominador em vez de condicionar apenas nos outputs julgáveis.
5. **Base rate:** Top-1 é acompanhado por macro-F1/balanced accuracy no dev e
   por métricas de ranking nos benchmarks.
6. **Regressão à média:** existe uma seed por fase e não se deve escolher uma
   narrativa a partir de flutuações de um checkpoint.
7. **Survivorship:** completar treino/gates prova integridade, não eficácia.
8. **Look-elsewhere:** foram observadas várias tarefas e métricas; os `p` são
   exploratórios e sem correção por multiplicidade.
9. **Forking paths:** a seleção de checkpoint foi pré-definida; a auditoria de
   captions e a hipótese de EOS foram análises pós-hoc.
10. **Correlação não é causalidade:** seeds, dose de treino, composição e
    qualidade dos targets mudaram entre E1 e E2.
11. **Causalidade inversa:** os outputs não demonstram por si só que um erro do
    teacher os causou; apenas mostram consistência com targets incompletos.

## 6. Limitações

- Não existe replicação multi-seed nem intervalo de confiança sobre a diferença
  entre fases.
- O single judge pode introduzir preferência de estilo; as métricas
  determinísticas são independentes dele, mas o open-ended não.
- Não se executou DermoBench nesta campanha, por desenho.
- E1 e E2 partilham arquitetura e cohorts, mas não a mesma seed de treino.
- O benchmark é interno e não valida uso clínico ou generalização externa.

## 7. Artefactos

- [Resumo do judge E2](../../outputs/e2_internal_benchmark_historical_t06/attempt-4/open_ended_judge_summary.json)
- [Relatório aberto E2 Frozen](../../outputs/e2_internal_benchmark_historical_t06/attempt-4/frozen/open_ended_diagnosis/qwen_3_5_4b_e2_frozen_vision_t06/20260816T145756Z_a4692e67/judge_report.html)
- [Relatório aberto E2 Vision](../../outputs/e2_internal_benchmark_historical_t06/attempt-4/vision/open_ended_diagnosis/qwen_3_5_4b_e2_vision_lora_t06/20260816T151709Z_dc985b50/judge_report.html)
- [Anotação do treino E2](../training_steps/04_e2_full_multitask_campaign_and_e1_comparison.md)
