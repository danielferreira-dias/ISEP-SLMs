# Comparação do dataset sintético e decisão de arquitetura

Data: 2026-08-02

## 1. Decisão executiva

A proposta das notas 11 e 13 é tecnicamente defensável, mas deve ser
apresentada como um **protocolo de geração e filtragem de supervisão**, não
como prova de que o teacher reproduz o raciocínio privado de um dermatologista.

O elemento mais distintivo não é usar duas etapas, porque já existem sistemas
com perceção seguida de diagnóstico. É combinar:

- uma Etapa A estritamente visual e sem acesso ao gold diagnosis;
- uma Etapa B que volta a receber a imagem e a Etapa A congelada, continuando
  sem acesso ao gold;
- observações curtas, estruturadas, com proveniência e escopo;
- diferencial ligado a evidência favorável, contrária e em falta;
- comparação emparelhada com a prompt direta congelada;
- comparação com o gold apenas depois da geração.

A recomendação arquitetural é **não substituir já o backbone nem introduzir
RL, MoE ou um sistema multi-agent**. Primeiro deve ser construído o corpus
canónico, executado SFT multitask com PEFT e medido onde ocorre o erro. A
principal alteração imediata é no schema, nos quality gates e nas ablations.

Há um bloqueio de dados anterior ao treino: o ISEPDermData 1.3.0 contém 7.541
imagens, mas a fila atual de anotação foi criada para o corpus antigo de
81.787 linhas. Apenas 6.735 IDs canónicos aparecem nessa fila e 806 estão em
falta. Nos 6.735 casos coincidentes, `caption`, `visual_concepts` e `symptoms`
estão vazios; `body_location` existe em apenas 309 casos. A fila deve ser
reconstruída a partir do release canónico antes de qualquer geração em escala.

## 2. O que existe atualmente no projeto

Esta distinção é importante para não confundir intenção, dados e código:

| Componente | Estado verificado | Consequência |
| --- | --- | --- |
| [ISEPDermData 1.3.0](../../data/training/ISEPDermData/README.md) | 7.541 imagens clínicas, 5.671 grupos leakage-safe, 21 classes e quatro sources; ainda é um pool `train` sem `sft_train/sft_dev` | É a fonte canónica para o próximo corpus |
| [Fila antiga](../../data/training/dermatology_multimodal_v1/teacher_annotation_queue.parquet) | 81.787 linhas; sobreposição parcial de 6.735 casos com o release atual | Não pode ser usada diretamente para gerar o SFT atual |
| Targets sintéticos | Os campos clínicos relevantes estão quase todos vazios | A proposta das notas 11/13 ainda não é um dataset materializado |
| Modelos | [Qwen 3.5 4B](../../configs/models/qwen_small_4b.yaml) foi selecionado como student oficial; Qwen 3.6 27B e os modelos API OpenRouter avançam para a Validation completa | A arquitetura do student está congelada, mas o trainer SFT ainda precisa de ser implementado e verificado |
| Código | Existe exportação, benchmark e inferência; não foi encontrado um trainer SFT executável em `src/` | É necessário implementar e verificar o caminho de PEFT antes de discutir alterações profundas |
| [ISEPDermaBench 1.5.0](../../data/benchmarks/ISEPDermaBench/README.md) | Validation, Internal Benchmark selado, DDI e SkinDisNet; tarefas Top-K, confusion sets, evidence-grounded e open-ended | Já fornece um protocolo local mais controlado do que comparar números isolados de papers |

O audit do caminho executável confirma ainda que não existe `src/training/`
nem formatter/collator multimodal, assistant-only loss, trainer ou manifest de
checkpoints. O [executor atual](../../src/benchmark/executor.py) faz uma chamada
por amostra, pelo que A -> B também ainda não está implementado. Afirmações no
[implementation log](../../doc/docs/src/implementation_log.md) sobre um antigo
`src/fine_tune/` não correspondem ao filesystem atual e não constituem
evidência de uma experiência executada.

O documento antigo `doc/docs/src/dataset_log.md` descreve uma experiência
histórica `derm-reasoning-think-v2`. Não deve ser citado como descrição do
corpus canónico atual sem ser explicitamente rotulado como histórico.

## 3. Comparação com datasets e modelos dermatológicos

As métricas abaixo só são comparáveis **dentro do mesmo paper**. Misturam
MCQA, balanced accuracy, zero-shot classification, retrieval, scores de um
LLM judge e ratings humanos. Não existe uma leaderboard comum que permita
afirmar que, por exemplo, 82% num dataset é superior a 70% noutro.

### 3.1 Datasets e MLLMs de reasoning

| Trabalho e força da evidência | Dados e modelo | O que foi feito para melhorar desempenho | Benchmark e resultado reportado | Leitura para o ISEP |
| --- | --- | --- | --- | --- |
| [DermoGPT](https://arxiv.org/abs/2601.01868), preprint | Qwen3-VL-8B; DermoInstruct com 211.243 imagens, 772.675 trajetórias e cerca de 646 mil exemplos de treino, agregados de 14 sources; ontologia de 9 superclasses e 325 subclasses | SFT multitask com LoRA; trajetórias `Observation -> Abstraction -> Deduction`; depois GRPO com reward MAVIC de diagnóstico, hierarquia, morfologia e formato; CCT com oito rollouts em teste | DermoBench: 12.371 imagens, 33.999 VQA e 11 tarefas. Média ID: base 52,44, SFT 77,25 e RL+CCT 78,04; OOD: SFT 64,19 e final 66,48 | A maior subida acontece no SFT, não no RL. O corpus usa gold/ontologia na síntese e no reward; não testa reasoning answer-blind. O agregado mistura accuracy e judge scores |
| [Skin-R1](https://arxiv.org/abs/2511.14900), ECCV 2026 | Qwen2.5-VL-7B; exemplares e conhecimento extraídos de Fitzpatrick 9e; grafo DDx de 211 nós/245 arestas, taxonomia de 458 nós; RL em 26.507 casos de seis datasets | SFT em trajetórias textbook-grounded, seguido de GRPO com reward hierárquico | Avaliação apenas MCQA. ID: base 43,67, SFT 47,43 e completo 63,85; OOD: 50,27, 51,53 e 71,71 | Bom antecedente para disease cards e hard negatives hierárquicos. Não valida rationales image-only nem abstention. Os livros não são automaticamente redistribuíveis |
| [SkinGPT-R1](https://arxiv.org/abs/2511.15242), preprint com versões materialmente diferentes | Vision-R1-7B congelado, PanDerm como teacher visual, adapter hierárquico e MoE de fairness; a versão atual descreve 310.830 casos Derm1M, 8.220 Fitzpatrick Black Skin Disease e 15.568 DermNet | Gemini descreve a imagem sem diagnóstico; Kimi recebe essa descrição **e o gold diagnosis**; DeepSeek normaliza o CoT; distillation visual e adaptação de cerca de 4,6 M parâmetros | Derm12345 82,50%, SkinCon 69,30%, Derm7pt 48,40%, PAD-UFES-20 42,40%; worst-group 41,40% em Fitz17k; cinco dermatologistas avaliaram 1.000 casos com média 3,6/5 | A Etapa A é próxima da proposta, mas a rationale é gold-conditioned. É uma comparação útil para medir hindsight bias, não um target fiel por defeito |
| [DermaSynth](https://arxiv.org/abs/2502.00196), preprint | 92.020 pares sintéticos a partir de 45.205 imagens de DERM12345, BCN20000, PAD-UFES-20, SCIN e HIBA; Gemini 2.0; fine-tuning preliminar de Llama-3.2-11B-Vision em 5.000 amostras | Self-instruct, prompts clínicos e inclusão de metadata para reduzir hallucination | O paper apresenta sobretudo a construção do recurso e exemplos preliminares; não fornece uma avaliação externa controlada suficiente para atribuir ganho de accuracy ao dataset | É o comparador de dataset mais direto e sobrepõe três sources do ISEP. Tem muito mais pares, mas menos isolamento entre observação, diagnóstico e suporte visual |
| [SkinGPT-4](https://www.nature.com/articles/s41467-024-50043-3), peer-reviewed | 52.929 imagens; SkinCon na fase de conceitos e 49.043 pares imagem-texto de DermNet e casos privados; ViT, Q-Former e Llama-2-13B congelados, com projeção treinável | Alinhamento em duas fases: conceitos visuais e depois diálogo/diagnóstico | Em 150 casos, 80,63% das respostas receberam `agree/strongly agree` dos dermatologistas | Mostra que alinhamento faseado pode funcionar com poucos parâmetros treináveis. O resultado é concordância subjetiva, não Top-1, e a avaliação é pequena |

O `DermBench` do SkinGPT-R1, o `DermoBench` do DermoGPT e o DermaBench
clinician-VQA de 2026 não são o mesmo benchmark. Os nomes não devem ser usados
indistintamente na dissertação.

### 3.2 Representação visual e alinhamento imagem-texto

| Trabalho | Dados e intervenção | Evidência reportada | Implicação |
| --- | --- | --- | --- |
| [PanDerm](https://www.nature.com/articles/s41591-025-03747-y), peer-reviewed | Pré-treino self-supervised de ViT-Large em 2.149.706 imagens de clínica, dermoscopia, total-body e dermatopatologia; masked latent modelling e alinhamento com teacher CLIP | 28 tarefas; ganhos médios reportados de 8,0 pp em imagem clínica e 5,1 pp em dermoscopia, com métricas diferentes por tarefa | Um encoder visual dermatológico pode valer mais do que texto de reasoning adicional. A escala é, contudo, quase 300 vezes a do ISEP e parte dos dados é privada |
| [Derm1M/DermLIP](https://arxiv.org/abs/2503.14911), ICCV 2025 | 1.029.761 pares, mais de 390 condições, quatro níveis de ontologia e 130 conceitos | O paper avalia classificação, retrieval e concept recognition em vários datasets | Útil como encoder/professor ou referência de ontologia. O audit local encontrou labels entity-linked e imagens dependentes de contexto, pelo que Derm1M foi corretamente excluído como supervisão diagnóstica direta do ISEP |
| [MAKE](https://arxiv.org/abs/2505.09372), MICCAI 2025 | 403.563 pares Derm1M; separa disease, concepts e subtexts; alinhamento global/local e weighting guiado pelo diagnóstico | Accuracy zero-shot média: 44,20 no baseline e 49,29 no modelo completo; DermNet 82,66 e Fitzpatrick17k 32,42; SkinCAP R@50 de 44,4/44,2 | Mais texto não basta. Targets curtos por aspeto e alinhamento fino deram +5,09 pp na ablation interna. O weighting é gold-conditioned e não equivale a rationale fiel |
| [VL-MedGuide](https://arxiv.org/abs/2508.06624), preprint | Primeira etapa prevê concepts; a segunda recebe concepts e a imagem novamente | Apenas Derm7pt: BACC/F1 83,55/80,12; sem concepts 81,23/77,56; sem CoT 82,05/78,89 | É o antecedente arquitetural mais próximo, mas o ganho é pequeno, num único dataset dermoscópico, sem código/splits/volume de treino suficientemente reproduzíveis |

### 3.3 Texto sintético ancorado e contexto interativo

O estudo peer-reviewed de [Marini et
al.](https://www.nature.com/articles/s41746-026-03040-3) é especialmente
relevante. Treinou uma arquitetura multimodal com aproximadamente 16 mil
imagens de seis datasets e avaliou retrieval e zero-shot em 15 datasets, seis
internos e nove externos. Comparou:

- `M`: template fixo baseado na metadata da lesão;
- `L`: nota natural gerada por LLM, condicionada pela metadata;
- `OL`: nota gerada pelo LLM sem essa ancoragem.

As notas `OL` introduziram hallucinations e pioraram generalização. A condição
`L` foi geralmente melhor porque combinou o label factual com descrição mais
rica. Contudo, a morfologia inventada pelo LLM não foi validada por médico por
imagem. Para o ISEP, a conclusão correta não é mostrar o gold à Etapa B
principal. É manter:

1. uma condição **answer-blind** para testar grounding e rationale;
2. uma condição **label-conditioned** separada para medir apenas utilidade de
   supervisão associativa/accuracy;
3. uma proibição explícita de chamar “observado” ao que foi apenas inferido a
   partir do label.

Dois sistemas ajudam a avaliar a política interativa:

- [DermPrompt](https://aclanthology.org/2024.clinicalnlp-1.50/) aumentou o
  hit-rate de retrieval de 59,57% image-only para 85,11% com imagem e contexto,
  e o re-ranking Top-1 de 42,55% para 53,19%. A amostra utilizável era apenas
  47 casos de validação; guidelines demasiado cedo reduziram o recall.
- [MINT](https://arxiv.org/abs/2401.12032) usou 17.454 casos privados e active
  feature acquisition para pedir imagem/contexto ou parar. Obteve Top-3 de
  61,2% usando menos inputs, face a 63,1% com todos os inputs. É evidência de
  que a ação deve ser avaliada pelo valor de informação e custo, não pela
  plausibilidade textual da pergunta.

## 4. O que parece realmente melhorar accuracy

A evidência conjunta sugere a seguinte ordem de importância:

1. **Representação visual de domínio.** PanDerm e a distillation visual do
   SkinGPT-R1 mostram que reasoning textual não corrige um encoder que não vê
   as características finas.
2. **SFT multitask bem ancorado.** No DermoGPT, o salto ID de 52,44 para 77,25
   ocorre antes de RL; RL+CCT acrescenta apenas 0,79 pp à média ID, embora ajude
   OOD.
3. **Labels, metadata e ontologia limpos.** DermaSynth e Marini reduzem
   hallucination ao ancorar geração; hard negatives dentro da mesma família
   tornam a tarefa clinicamente informativa.
4. **Targets granulares.** MAKE melhora quando separa aspetos e alinha texto
   localmente; uma caption longa não substitui morfologia, evidência e ação.
5. **Contexto real quando muda o diferencial.** DermPrompt e MINT mostram
   benefício, mas também que recolher tudo tem custo e que contexto errado
   pode prejudicar.
6. **Revisão clínica e avaliação externa.** Scores automáticos de rationale
   não provam grounding; é necessário auditar afirmações observáveis,
   subgroup performance e generalização.
7. **RL, self-consistency e MoE.** Podem acrescentar desempenho depois de uma
   boa fundação, mas os ganhos não justificam essa complexidade como primeiro
   passo num corpus de 7.541 imagens.

## 5. Processo clínico que o schema deve aproximar

Uma sequência defensável, sem afirmar que todos os especialistas pensam de
forma linear, é:

```text
qualidade, modalidade e cobertura
  -> descrição sem diagnóstico
  -> padrão global + características analíticas
  -> integração de história/exame realmente fornecidos
  -> diferencial com evidência favorável, contrária e em falta
  -> incerteza diagnóstica + risco se falhar
  -> diagnóstico provisório ou próxima ação confirmatória
```

Esta estrutura converge entre o [Clinical Methods da
NCBI](https://www.ncbi.nlm.nih.gov/books/NBK206/), a [nomenclatura
ILDS](https://academic.oup.com/bjd/article/174/6/1351/6617055), o [exame
dermatológico da DermNet](https://dermnetnz.org/cme/principles/examination-of-the-skin)
e a revisão cognitiva de [Ko et
al.](https://pubmed.ncbi.nlm.nih.gov/30797839/). O estudo de [Gachon et
al.](https://pubmed.ncbi.nlm.nih.gov/15837860/) também mostra que especialistas
usam reconhecimento global, comparação com outras lesões e evolução, não
apenas uma checklist. Por isso, `dominant_visual_pattern` deve coexistir com
campos analíticos.

Uma fotografia não permite afirmar tudo o que um exame permite:

| Estado | Exemplos | Regra |
| --- | --- | --- |
| Observável | forma, bordo, cor relativa, escala, crosta, configuração dentro do frame | Pode ser `present` com evidência regional |
| Condicional | tamanho, elevação aparente, eritema, distribuição, simetria | Só quando vista, escala, luz e cobertura o permitem |
| Não avaliável | consistência, induração, mobilidade, calor, dor à palpação, blanching, profundidade real | `not_assessable`, nunca inventar |
| Requer história | duração, evolução, prurido, fármacos, exposições, sintomas sistémicos | Só com `provenance=provided_history` |
| Não mostrado | unhas, mucosa, distribuição corporal, outras lesões | `not_shown`, não `absent` |
| Requer outra modalidade/teste | dermoscopia, KOH, cultura, histologia | Pedir modalidade/teste; não inferir o resultado |

Os termos ILDS `papule`, `plaque` e `nodule` incluem palpabilidade. Numa foto
única, o target deve preferir `apparent_papule`, `plaque_like` ou confiança
reduzida. Também não se deve inferir `generalized`, `symmetric`, `dermatomal`,
“sem envolvimento ungueal” ou “purpura não branqueável” quando a aquisição não
o permite.

## 6. Schema revisto

### 6.1 Etapa A: perceção answer-blind

```json
{
  "image_assessment": {
    "is_evaluable": true,
    "views_available": ["closeup"],
    "quality_defects": [],
    "has_anatomic_overview": false,
    "has_scale": false,
    "has_lateral_profile": false,
    "distribution_assessability": "within_frame_only",
    "color_reliability": "uncertain"
  },
  "dominant_visual_pattern": "...",
  "observations": [
    {
      "id": "obs_1",
      "concept": "apparent_plaque",
      "status": "present",
      "provenance": "clinical_photo",
      "scope": "index_lesion",
      "confidence": "moderate",
      "evidence_region": "central_lesion"
    }
  ],
  "associated_structures_visible": [],
  "not_assessable_features": ["palpable_consistency", "blanching"]
}
```

Estados permitidos para cada observation:

```text
present
absent_in_observed_scope
uncertain
not_assessable
not_shown
```

`absent` isolado deve ser inválido, porque transforma ausência no frame em
ausência clínica.

### 6.2 Etapa B: diferencial e ação

```json
{
  "stage_b_corrections": [],
  "differential": [
    {
      "rank": 1,
      "disease_id": "D004",
      "supporting_observation_ids": ["obs_1"],
      "contradicting_observation_ids": [],
      "missing_discriminators": [
        {
          "feature": "duration_and_evolution",
          "required_source": "history"
        }
      ],
      "diagnostic_confidence": "moderate",
      "clinical_risk_if_missed": "high"
    }
  ],
  "action": "REQUEST_CLINICAL_CONTEXT",
  "action_urgency": "prompt",
  "requested_information": "duration_and_recent_change",
  "concise_clinical_rationale": "..."
}
```

Se B discordar de A depois de voltar a observar a imagem, deve preencher
`stage_b_corrections` com o `observation_id`, valor revisto e justificação
visual. Uma correção silenciosa quebraria a auditabilidade; proibir B de
corrigir A criaria, por outro lado, um bottleneck rígido.

Separar `diagnostic_confidence`, `clinical_risk_if_missed` e
`action_urgency` evita tratar uma hipótese moderadamente provável mas perigosa
como um caso de baixo risco.

O vocabulário de ação deve ser mais específico do que
`CLASSIFY|ASK_CONTEXT|REQUEST_BETTER_IMAGE|ABSTAIN`:

```text
DIAGNOSE_PROVISIONALLY
REQUEST_OVERVIEW_IMAGE
REQUEST_CLOSEUP_IMAGE
REQUEST_SCALE_OR_PROFILE
REQUEST_CLINICAL_CONTEXT
REQUEST_DERMOSCOPY
REQUEST_IN_PERSON_EXAM
RECOMMEND_CONFIRMATORY_TEST
ABSTAIN_POOR_QUALITY
ABSTAIN_OUT_OF_DOMAIN
```

As recomendações de aquisição devem seguir qualidade fotográfica e
teledermatologia: overview, mid-range, close-up, orientação e escala quando
aplicável. Ver [ImageQX](https://pubmed.ncbi.nlm.nih.gov/36735575/) e as
[guidelines de teledermatologia](https://pmc.ncbi.nlm.nih.gov/articles/PMC10335147/).
Dermoscopia e biópsia devem ser ações clínicas, não resultados inventados; ver
[NICE](https://www.nice.org.uk/guidance/QS130/chapter/quality-statement-3-dermoscopy)
e [AAD](https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma/diagnose-treat).

### 6.3 Proveniência do gold

Manter um campo diferente do target sintético:

```text
clinical_diagnosis
dermoscopy_supported
histopathology_confirmed
microbiology_confirmed
expert_consensus
unknown_provenance
```

Uma label clínica e uma label histopatologicamente confirmada não devem ter o
mesmo peso de avaliação ou de filtragem.

## 7. Ablations que distinguem os mecanismos

### 7.1 Geração pelo teacher

Usar os mesmos task IDs, imagens e parâmetros de geração:

| Condição | Input e output | Pergunta respondida |
| --- | --- | --- |
| G0 — direta | imagem -> prompt 1.1.0 congelada | Qual é o baseline atual? |
| G1 — uma passagem | imagem -> observations + diferencial | Estrutura por si só ajuda? |
| G2 — bottleneck rígido | A vê imagem; B vê apenas A | Quanto se perde ao remover a imagem? |
| G3 — principal | A vê imagem; B vê imagem + A; ambas answer-blind | A decomposição auditável ajuda sem hindsight? |
| G4 — knowledge-assisted | G3 + disease cards leakage-safe | Conhecimento textual acrescenta discriminação? |
| G5 — gold-conditioned | imagem + gold/metadata -> nota | Qual o ganho associativo e qual o hindsight bias? |

G5 é uma ablation informativa à luz de Marini et al., mas não deve ser chamada
de reasoning fiel nem misturada silenciosamente com G3.

### 7.2 Treino do student

| Condição | Supervisão |
| --- | --- |
| S0 | imagem + label |
| S1 | S0 + quality/coverage + observations |
| S2 | S1 + diferencial, evidence links e rationale curta |
| S3 | S2 + política de próxima ação |
| S4, controlo de investigação | raw CoT do teacher |

Todos os modelos devem usar o mesmo backbone inicial, split por
`leakage_group_id`, seed, budget de tokens e política de sampling. Caso
contrário, não é possível atribuir o ganho aos targets sintéticos.

## 8. Decisão de arquitetura por fases

O pipeline de geração em duas chamadas não obriga o student a ser dois
modelos. Inicialmente, um único MLLM pode aprender tarefas diferentes por
instrução e produzir a resposta clínica final numa passagem.

### Fase A — baseline mínimo

```text
backbone multimodal pequeno
  + vision tower congelada
  + LoRA/QLoRA no projector e blocos de linguagem suportados
  + collator multimodal e assistant-only loss verificados
  + SFT multitask com sampling por classe/source/tarefa
```

Razão: 7.541 imagens são poucas para treinar uma vision tower completa sem
risco elevado de overfitting. DermoGPT treinou com cerca de 646 mil instruções
e PanDerm foi pré-treinado com mais de dois milhões de imagens.

### Fase B — adaptação visual condicionada a evidência

Só testar vision LoRA, últimos blocos visuais ou unfreezing parcial se:

- o treino aprende formato/labels mas morphology e grounding ficam perto do
  modelo base;
- os erros são visuais, não de taxonomia, mapping ou parser;
- há melhoria num linear probe ou numa avaliação de embeddings dermatológicos;
- a Validation não mostra overfitting por source.

Antes de fundir PanDerm/DermLIP com o MLLM, comparar embeddings congelados ou
um probe simples. Se um encoder dermatológico melhorar de forma consistente
morfologia e diagnóstico externo, então testar um projector/fusion adapter
como ablation. Substituir o encoder diretamente aumenta incompatibilidade de
resolução, tokens, licenças e custo de treino.

### Fase C — componentes opcionais

- tarefas auxiliares generativas de quality, modality, hierarchy e OOD antes
  de criar heads arquiteturais dedicadas;
- suporte multi-image apenas quando overview/close-up/dermoscopia pertencem de
  forma comprovada ao mesmo caso;
- hard negatives dentro do mesmo ramo taxonómico;
- CCT/self-consistency apenas em casos incertos e depois de medir custo;
- RL apenas se houver reward verificável e se SFT tiver estabelecido um ganho
  robusto.

### O que não introduzir no baseline

- concept bottleneck que retire a imagem da etapa diagnóstica;
- MoE por tom de pele sem volume e labels demográficos suficientes;
- RAG sobre Validation, Internal Benchmark ou sources com sobreposição não
  auditada;
- full fine-tuning visual em 7.541 imagens;
- multi-agent debate como substituto de um baseline reprodutível;
- contexto clínico sintético apresentado como facto do doente.

## 9. Quality gates do corpus sintético

1. Validar JSON, schema, enumerações e referências de observation IDs.
2. Rejeitar qualquer história, sintoma, palpação, dermoscopia ou teste sem
   proveniência real.
3. Exigir escopo para findings negativos e distinguir `not_shown` de
   `absent_in_observed_scope`.
4. Permitir que a Etapa B corrija explicitamente a Etapa A, registando a
   alteração; a imagem permanece disponível.
5. Comparar o Top-1 com o gold só depois da geração. Aceitar a componente
   diagnóstica apenas segundo a política predefinida, mas conservar Etapas A
   visualmente válidas mesmo quando o diagnóstico falha.
6. Rejeitar loops, truncamentos, duplicação mecânica e claims centrais sem
   evidência.
7. Medir acceptance rate por classe, source, diagnosis basis, qualidade e tom
   de pele quando disponível. Não filtrar até restarem apenas casos fáceis.
8. Auditar amostras aceites e rejeitadas por dermatologistas, com protocolo e
   critérios congelados; reportar acordo entre revisores quando houver dois.
9. Guardar teacher, versão, prompt, seed, parâmetros, raw response, parser,
   decisão de gate e hashes da imagem e do schema.
10. Nunca gerar targets de treino a partir de Validation, Internal Benchmark,
    DDI ou SkinDisNet.
11. Verificar que o gold não entra por nome de ficheiro, metadata oculta,
    retries condicionados pela resposta ou reescrita por um judge label-aware.

## 10. Plano de execução e decision gates

### Gate 0 — corpus canónico

1. Gerar `teacher_annotation_queue` exclusivamente a partir do ISEPDermData
   1.3.0 e confirmar 7.541/7.541 IDs.
2. Criar `sft_train` e `sft_dev` por `leakage_group_id`, visando 10–15% dos
   grupos para Dev mas escolhendo a proporção final após verificar equilíbrio
   de classe e source.
3. Reintegrar apenas metadata permitida, com proveniência de coluna e um valor
   explícito para missingness.
4. Congelar versões, hashes, seed e contagens.

### Gate 1 — validar o protocolo de geração

1. Smoke test de 20 casos para schema, loops, latência e custo.
2. Estudo emparelhado G0/G1/G2/G3 em pelo menos 100 casos estratificados da
   Validation; G4 e G5 apenas depois.
3. Avançar se G3 melhorar morphology/evidence grounding e não degradar Top-K,
   unsupported claims ou custo para além do limite predefinido.
4. Usar a Validation apenas para desenhar o protocolo; os seus outputs nunca
   entram no treino.

### Gate 2 — gerar e auditar Train

1. Gerar somente `sft_train` e, com política idêntica, `sft_dev`.
2. Aplicar gates determinísticos antes do judge ou review humano.
3. Auditar taxas de rejeição e calibrar sem consultar o Internal Benchmark.
4. Produzir um dataset card com exemplos, exclusões, limitações e licenças.

### Gate 3 — SFT e arquitetura

1. Treinar S0, S1 e S2 com PEFT e budget comparável.
2. Selecionar checkpoint com SFT Dev; usar Validation para métricas clínicas e
   threshold/calibration.
3. Só se morphology permanecer fraca, testar vision LoRA ou encoder de domínio.
4. Só se S2 melhorar de forma robusta, acrescentar S3; S4 permanece controlo.
5. Depois de congelar modelo, prompts e thresholds, executar uma vez o Internal
   Benchmark e os testes externos.

## 11. Métricas necessárias

### Diagnóstico e ranking

- Top-1, Top-3, MRR e macro-F1/balanced accuracy quando aplicável;
- calibração, selective accuracy, coverage e risk-coverage;
- OOD/abstention precision e recall.

### Perceção e grounding

- precisão/recall/F1 de concepts compatíveis com referências;
- unsupported finding rate;
- evidence-link validity e contradições;
- taxa de correção A -> B;
- performance por image quality e coverage.

### Raciocínio observável e ação

- qualidade do diferencial e hard-negative discrimination;
- evidência favorável, contrária e discriminadores em falta;
- afirmações não avaliáveis ou fora do frame;
- valor diagnóstico da pergunta/nova imagem, quando existe follow-up real;
- custo, tokens, latência e número de interações.

### Robustez e equidade

- classe, source, modalidade, diagnosis basis e duplicate/leakage group;
- tom de pele, sexo/género e idade apenas onde a metadata é real, licenciada e
  suficientemente completa;
- diferenças entre Validation interna, Internal Benchmark, DDI e SkinDisNet.

O [ISEPDermaBench](../../data/benchmarks/ISEPDermaBench/README.md) já oferece:

- `visual_top_k`: 1.000 casos em Validation, 1.000 no Internal Benchmark, 300
  DDI e 1.365 SkinDisNet;
- `visual_confusion_sets`: 834 tarefas/417 imagens em Validation e 828/414 no
  Internal Benchmark;
- `evidence_grounded_diagnosis`: 137/134/636 casos em Validation/Internal/DDI;
- `open_ended_diagnosis`: 100 Validation e 300 Internal.

Faltam sobretudo métricas formais de calibração/abstention, action utility e
uma auditoria clínica do novo schema.

## 12. Conclusão para a dissertação

Os trabalhos anteriores mostram que pré-treino visual de domínio,
decomposição percetual, conhecimento dermatológico, SFT multitask e contexto
podem melhorar resultados. Porém, vários datasets de reasoning criam a
explicação depois de fornecer o diagnóstico correto. Isso mede a capacidade
de produzir uma justificação plausível, não necessariamente de inferir uma
conclusão sustentada pela imagem.

O contributo experimental mais limpo do projeto é testar diretamente essa
diferença:

```text
reasoning answer-blind e visualmente auditável
versus
racionalização gold-conditioned
```

Antes de alterar a arquitetura, deve demonstrar-se que o novo contrato de
supervisão melhora perceção, grounding, ranking e calibração no mesmo student
e nos mesmos splits. Se o erro continuar a ser visual depois desse controlo,
então vision LoRA ou um encoder dermatológico passam a ser alterações
justificadas por evidência, em vez de complexidade adicionada por hipótese.

## 13. Fontes clínicas e condições de utilização

- [ILDS: revisão da nomenclatura de lesões
  cutâneas](https://academic.oup.com/bjd/article/174/6/1351/6617055)
- [NCBI Clinical Methods: overview da pele e
  anexos](https://www.ncbi.nlm.nih.gov/books/NBK206/)
- [DermNet: examination of the
  skin](https://dermnetnz.org/cme/principles/examination-of-the-skin)
- [DermNet: terminology in
  dermatology](https://dermnetnz.org/topics/terminology)
- [Ko et al.: perceção, cognição e erro no diagnóstico
  dermatológico](https://pubmed.ncbi.nlm.nih.gov/30797839/)
- [Gachon et al.: reconhecimento prospetivo de
  melanoma](https://pubmed.ncbi.nlm.nih.gov/15837860/)
- [Rimoin et al.: treino percetual de morfologia, configuração e
  distribuição](https://pubmed.ncbi.nlm.nih.gov/25592621/)
- [AAD: standards de
  teledermatologia](https://www.aad.org/member/practice/telederm/standards)

Estas fontes suportam vocabulário, processo e safety rules. Não concedem por
si só licença para copiar capítulos, definições extensas ou imagens. As
disease cards devem ser sínteses próprias, curtas, citadas e sujeitas a uma
auditoria de licença antes de serem distribuídas ou usadas em treino.
