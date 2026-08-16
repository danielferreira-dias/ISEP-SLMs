# Plano detalhado de construção do ISEPDistillDataset

| Campo | Valor |
| --- | --- |
| Data da decisão | 2026-08-08 |
| Estado | `diagnosis` e `morphology` v0.3.0; `caption` v0.4.1 executado em E2, mas rejeitado para reutilização direta em E3 após auditoria de completude |
| Student de referência | Qwen 3.5 4B multimodal |
| Objetivo | construir supervisão multimodal leakage-safe para especializar um modelo pequeno em perceção, descrição e diagnóstico dermatológico, mantendo cada afirmação clínica auditável |

## 1. Decisão executiva

O `ISEPDistillDataset` deve ser um repositório Hugging Face multimodal com várias configurações ligadas pelo mesmo `sample_id`. Cada imagem pode originar vários exemplos de treino, mas cada exemplo ensina uma tarefa curta e identificável: avaliar a qualidade da imagem, reconhecer morfologia, produzir uma descrição, ordenar diagnósticos, justificar esse diferencial com evidência visível ou decidir que informação adicional é necessária.

O dataset não deve ter como target principal uma chain-of-thought longa. A proposta é guardar conteúdo clínico canónico e verificável, com duas representações destinadas ao student:

- uma resposta estruturada em JSON, adequada a treino controlado, filtragem e avaliação por campo.
- uma resposta aberta, curta e natural, derivada dos mesmos factos clínicos e guardada num campo próprio.

O teacher gera a supervisão principal em duas chamadas. A Etapa A descreve apenas o que é observável na imagem, sem conhecer o diagnóstico correto. A Etapa B volta a receber a imagem e a saída congelada da Etapa A, constrói um diferencial e liga cada hipótese a observações concretas. O `gold_diagnosis` só é consultado depois da geração, para aceitar ou rejeitar o target diagnóstico. Este desenho reduz racionalização retrospetiva e permite conservar uma Etapa A válida mesmo quando a Etapa B erra.

O núcleo da primeira versão deve ser SFT multitask. Distilação de logits, estados internos ou features visuais é uma experiência posterior e deve usar artefactos separados do dataset clínico principal. Esta separação é necessária porque um bom corpus de respostas e uma loss de knowledge distillation são mecanismos distintos.

## 2. O que este dataset pretende ensinar

O dataset deve responder a cinco objetivos de aprendizagem:

1. **Ver antes de diagnosticar.** O student deve reconhecer qualidade, morfologia, cor relativa, superfície e distribuição dentro do enquadramento.
2. **Distinguir observação de conhecimento médico.** Uma característica só pode ser declarada como presente quando é visível ou fornecida por metadata real.
3. **Construir um diferencial verificável.** Cada hipótese deve indicar evidência favorável, evidência contrária e informação em falta.
4. **Comunicar em dois formatos.** O modelo deve conseguir devolver JSON estável e uma resposta clínica natural, sem contradições entre ambos.
5. **Saber quando não concluir.** Imagem insuficiente, caso fora de domínio ou falta de contexto devem conduzir a pedido de informação, nova imagem, exame presencial ou abstention.

O objetivo da tese não é demonstrar que o teacher reproduz o raciocínio privado de um dermatologista. O que pode ser medido é se uma supervisão observável, estruturada e ligada à imagem melhora o student em diagnóstico, morfologia, grounding, calibração e segurança.

## 3. Evidência científica que sustenta o desenho

### 3.1 Estudos dermatológicos

| Estudo | Resultado ou mecanismo relevante | Decisão que sustenta | Limite da evidência |
| --- | --- | --- | --- |
| [SkinGPT-4](https://doi.org/10.1038/s41467-024-50043-3), Nature Communications 2024 | Alinhamento em duas fases de um encoder visual e um LLM, usando 52.929 imagens, conceitos clínicos e notas médicas | Começar por alinhamento visual-conceptual antes de diálogo e diagnóstico | A avaliação clínica teve 150 casos e não equivale a accuracy Top-1 num benchmark externo |
| [SkinCaRe, incluindo SkinCAP e SkinCoT](https://arxiv.org/abs/2405.18004), versão atual de 2025 | Reúne 4.000 imagens com captions dermatológicas e 3.041 pares imagem-reasoning clínico | Captions clínicas podem ser uma fonte auxiliar quando licença, qualidade e leakage forem controlados | A publicação não transforma automaticamente todos os textos em targets livres de gold-conditioning, leakage ou restrições de redistribuição |
| [PanDerm](https://doi.org/10.1038/s41591-025-03747-y), Nature Medicine 2025 | Pré-treino de um ViT em mais de dois milhões de imagens, 11 instituições, quatro modalidades e avaliação em 28 benchmarks | Um encoder visual de domínio pode ser mais importante do que aumentar apenas o reasoning textual | A escala e os dados privados são muito superiores ao contexto da tese. Não é uma receita de compressão para Qwen |
| [MAKE](https://arxiv.org/abs/2505.09372), MICCAI 2025 | Divide descrições longas em aspetos clínicos e alinha subtextos a evidência visual | Criar targets curtos para morfologia, superfície, cor e diagnóstico em vez de uma única caption longa | Parte do weighting usa conhecimento diagnóstico, pelo que não prova fidelidade answer-blind |
| [DermoGPT](https://arxiv.org/abs/2601.01868), preprint 2026 | DermoInstruct contém 211.243 imagens e 772.675 trajetórias. O modelo usa supervisão morfológica, SFT e depois RL | Morfologia e diagnóstico devem ser tarefas separadas. O SFT é a primeira intervenção a testar | O corpus usa ontologia/gold durante parte da síntese e a contribuição adicional de RL é mais difícil de isolar |
| [SkinGPT-R1](https://arxiv.org/abs/2511.15242), preprint 2025 | Usa descrição inicial sem diagnóstico, seguida de rationale condicionada pelo gold, e distilação visual com PanDerm | Justifica comparar geração answer-blind com uma condição gold-conditioned e estudar um teacher visual especializado | A rationale condicionada pelo gold pode ser plausível sem representar a inferência original. O artigo tem versões materialmente diferentes |
| [SkinFlow](https://arxiv.org/abs/2601.09136), preprint 2026 | Combina captions dermatológicas antes de diagnóstico e um Dynamic Vision Encoder com `FDLinear` | Sustenta o currículo descrever-antes-de-diagnosticar e uma ablation visual futura | Não existe ablation `DVE-only`, baseline equivalente de Vision LoRA, código completo de `FDLinear` ou validação externa reproduzível |

O resultado mais útil de SkinFlow para este dataset é a Stage I de descrição. Na ablation reportada, adicionar essa fase aumentou o Top-1 de Fitzpatrick17k de 15,22% para 24,45%. Adicionar depois o DVE elevou-o para 29,19%. No conjunto interno, o ganho Top-1 atribuído ao DVE foi apenas 0,99 pontos. Isto indica que supervisão descritiva merece ser testada antes de alterar a arquitetura, mas não permite concluir que `FDLinear` é o melhor mecanismo visual.

### 3.2 Estudos de knowledge distillation multimodal

| Estudo | Contributo | Aplicação prudente no ISEPDistillDataset |
| --- | --- | --- |
| [Distilling Step-by-Step](https://doi.org/10.18653/v1/2023.findings-acl.507), Findings of ACL 2023 | Mostra que rationales do teacher podem melhorar eficiência de dados em tarefas textuais | Motiva targets explicativos curtos, mas não prova que uma rationale multimodal esteja visualmente fundamentada |
| [MiniLLM](https://arxiv.org/abs/2306.08543), ICLR 2024 | Usa reverse-KL e sequências amostradas pelo student para reduzir exposure bias | Motiva uma fase on-policy posterior, não a estrutura inicial do dataset estático |
| [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649), ICLR 2024 | O teacher avalia sequências produzidas pelo student | Sustenta guardar rollouts on-policy num artefacto opcional depois de existir um baseline SFT |
| [LLAVADI](https://arxiv.org/abs/2407.19409), 2024 | Estuda sistematicamente distilação de MLLMs e reporta benefício de alinhamento conjunto de tokens/logits e representações | Justifica testar KD multimodal seletiva. Não justifica distilar indiscriminadamente todas as layers ou attentions |
| [LLaVA-KD](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html), ICCV 2025 | Propõe distilled pre-training, SFT e distilled fine-tuning com transferência multimodal e relacional | Sustenta separar currículo, SFT e KD em fases mensuráveis |
| [Align-KD](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Align-KD_Distilling_Cross-Modal_Alignment_Knowledge_for_Mobile_Vision-Language_Large_Model_CVPR_2025_paper.html), CVPR 2025 | Alinha conhecimento cross-modal em layers selecionadas e usa distilação da projeção visual-textual | Motiva uma experiência dirigida a layers escolhidas. Alinhar uma representação inadequada pode degradar o student |
| [When Better Teachers Don't Make Better Students](https://arxiv.org/abs/2511.17886), preprint 2025 | Discute mismatch entre capacidade/representação do teacher e do student | O teacher com maior benchmark não deve ser escolhido automaticamente. É necessário medir teachability no Qwen 4B |

A literatura não identifica uma estratégia universalmente superior. A conclusão operacional é começar por dados supervisionados de alta qualidade e, depois, comparar feature distillation e on-policy distillation com o mesmo student, dados, compute e splits. O core dataset não deve armazenar dezenas de gigabytes de hidden states sem existir uma experiência que demonstre a sua necessidade.

### 3.3 Leitura conjunta da literatura

Os estudos convergem numa sequência simples:

```text
dados e labels limpos
  -> supervisão visual/conceptual
  -> descrição curta e auditável
  -> diagnóstico e diferencial
  -> SFT do student
  -> adaptação visual ou KD seletiva, se o erro continuar a ser visual
  -> RL/on-policy, apenas se acrescentar valor ao baseline anterior
```

Esta ordem é uma hipótese experimental apoiada por estudos diferentes, não uma receita já validada no conjunto exato `Qwen 3.6 27B -> Qwen 3.5 4B` e nos 21 diagnósticos do ISEP.

## 4. Fontes de dados e função de cada uma

### 4.1 ISEPDermData como corpus canónico

O release local atual do `ISEPDermData` contém 7.541 imagens de treino, 21 classes e quatro sources:

| Source | Imagens no pool atual |
| --- | ---: |
| Fitzpatrick17k-C | 3.226 |
| HIBA | 318 |
| PAD-UFES-20 | 1.629 |
| SCIN | 2.368 |
| **Total** | **7.541** |

Estas imagens fornecem a identidade canónica, a label de diagnóstico normalizada, a provenance da label, o source e o `leakage_group_id`. Antes da geração, o pool deve ser dividido em `sft_train` e `sft_dev` por grupo de leakage, nunca por linha individual. A divisão deve preservar, tanto quanto possível, classe, source e subgrupos demográficos disponíveis.

A utilização de quatro fontes não se justifica apenas pelo aumento do número de imagens. Cada source introduz um tipo diferente de cobertura e de incerteza, pelo que deve cumprir uma função explícita:

- **Fitzpatrick17k-C — cobertura de doenças e de fototipos.** O Fitzpatrick17k original reuniu 16.577 imagens clínicas com anotações de tipo de pele e mostrou que o desempenho dos modelos dependia da proximidade entre a distribuição de fototipos do treino e a população avaliada ([Groh et al., 2021](https://doi.org/10.1109/CVPRW53098.2021.00201)). Isto justifica a sua inclusão para estudar desempenho e erros entre tons de pele, mas não justifica aceitar o corpus original sem auditoria. Uma investigação posterior identificou duplicados, problemas de labels, risco de leakage e ausência de splits normalizados, propondo a versão corrigida Fitzpatrick17k-C ([Abhishek, Jain e Hamarneh, 2025](https://doi.org/10.1038/s41597-025-04382-5)). Por essa razão, o projeto usa a variante corrigida e volta a aplicar grouping e deduplicação próprios.
- **PAD-UFES-20 — variabilidade de aquisição por smartphone e metadata clínica.** O dataset original contém 2.298 imagens de 1.641 lesões em 1.373 doentes, captadas com smartphones, acompanhadas por dados clínicos; todos os casos de cancro de pele foram confirmados por biópsia ([Pacheco et al., 2020](https://doi.org/10.1016/j.dib.2020.106221)). A subset local filtrada é útil para aproximar o treino de imagens clínicas menos controladas e para explorar contexto com provenance real. A unidade de split deve ser o doente ou a lesão, e não a fotografia, porque uma lesão pode ter mais do que uma imagem.
- **SCIN — condições comuns, imagens submetidas pelos participantes e múltiplas vistas.** O estudo SCIN recolheu 10.408 imagens de 5.033 contribuições filtradas através de participação voluntária, incluindo uma a três imagens, sintomas e informação demográfica por caso; a coleção aumenta sobretudo a cobertura de condições alérgicas, inflamatórias e infeciosas, que estão frequentemente sub-representadas em datasets centrados em neoplasias ([Ward et al., 2024](https://doi.org/10.1001/jamanetworkopen.2024.46615)). Isto torna SCIN útil para diversidade clínica e robustez a aquisição real. Contudo, as labels retrospetivas e os dados autorreportados não têm a mesma certeza de uma confirmação histopatológica, pelo que `label_provenance` deve influenciar filtragem, ponderação e análise de erro.
- **HIBA — diversidade geográfica e imagens clínicas hospitalares.** O registo oficial do Hospital Italiano de Buenos Aires descreve 1.635 imagens de lesões cutâneas, clínicas e dermatoscópicas, recolhidas na Argentina ([Hospital Italiano de Buenos Aires, 2023](https://doi.org/10.34970/559884)). A subset clínica elegível acrescenta uma instituição e uma população diferentes das restantes fontes. No ISEPDistillDataset deve manter-se a modalidade explícita e nunca apresentar informação dermatoscópica como se tivesse sido inferida de uma fotografia clínica.

Assim, a combinação de sources procura reduzir dependência de um único hospital, mecanismo de aquisição ou tipo de patologia. Não garante generalização por si só: `source_dataset` deve permanecer em todas as tabelas de auditoria, e os resultados devem ser apresentados por source, classe e subgrupo sempre que o tamanho amostral o permita.

O `gold_diagnosis` pertence ao registo canónico e ao filtro pós-geração. Não pertence à allowlist de input da Etapa A ou da Etapa B principal.

### 4.2 SKINCON como supervisão humana de morfologia

O snapshot local de [SKINCON](../../configs/datasets/skincon/README.md) contém:

- 3.690 registos Fitzpatrick17k, dos quais 3.230 são utilizáveis.
- 656 registos DDI, dos quais 636 são utilizáveis.
- 48 conceitos binários de morfologia anotados por especialistas.

O estudo SKINCON foi concebido precisamente para fornecer conceitos clínicos finos, explicações baseadas em conceitos e análise de erros: dermatologistas anotaram 48 conceitos em 3.230 imagens Fitzpatrick17k e aplicaram a mesma ontologia a imagens DDI ([Daneshjou et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html)). Esta evidência sustenta usar SKINCON para `morphology`, concept grounding e avaliação por conceito. Não sustenta usá-lo como taxonomia diagnóstica principal, porque os conceitos descrevem sinais visuais e não substituem a label da doença.

Há também precedente arquitetural para esta separação: o SkinGPT-4 utilizou primeiro alinhamento imagem-conceito com SKINCON e só depois treino de interação imagem-texto ([Zhou et al., 2024](https://doi.org/10.1038/s41467-024-50043-3)). No ISEP, isto motiva uma fase visual-conceptual anterior ou uma tarefa de morfologia separada; não demonstra, por si só, que a mesma sequência será ótima no Qwen 3.5 4B.

O audit reproduzido em 14 de agosto de 2026 encontrou 606 imagens Fitzpatrick17k do pool de treino atual com conceitos SKINCON utilizáveis, 2.353 imagens Fitzpatrick17k adicionais elegíveis para uma configuração `morphology` sem target diagnóstico da taxonomia de 21 classes e 271 overlaps com o Internal Benchmark congelado. Estes 271 continuam excluídos para preservar a validade interna. O pool Fitzpatrick17k elegível fica, portanto, em 2.959 imagens.

Por decisão experimental registada em 14 de agosto de 2026, as 636 anotações SKINCON DDI utilizáveis passam a `sft_train`/`sft_dev`, sob o Research Use Agreement e apenas no repositório privado. Esta decisão eleva o pool de morfologia para 3.595 imagens e deixa de reservar linhas DDI para avaliação externa. A consequência deve ser explícita: depois de treinar com estas imagens ou anotações, DDI deixa de ser evidência independente de generalização e os seus resultados só podem ser descritos como in-domain ou contaminados. SkinDisNet e os restantes conjuntos não usados no treino mantêm o papel externo.

O builder materializado confirmou que todas as 3.866 anotações SKINCON utilizáveis têm uma label diagnóstica upstream. Depois de excluir os 271 overlaps reservados e aplicar os aliases e mappings source-specific versionados, 1.198 das 3.595 linhas publicadas mapeiam para as 21 classes atuais; as restantes 2.397 mantêm a label original, mas entram apenas como supervisão de morfologia para não expandir silenciosamente a taxonomia diagnóstica. O número preliminar de 1.171 foi corrigido porque não aplicava integralmente esses mappings revistos.

### 4.3 SkinCAP e SkinCoT como fontes condicionais

O snapshot local de [SkinCaRe](../../configs/datasets/skincare/README.md) inclui 4.000 imagens SkinCAP, 3.041 imagens SkinCoT e versões textuais em inglês e chinês. SkinCAP contém 187 strings de doença e deriva de 3.345 casos Fitzpatrick17k e 655 casos DDI. A versão atual do artigo SkinCaRe unifica SkinCAP e SkinCoT: apresenta as 4.000 captions como descrições escritas por dermatologistas e os 3.041 pares de reasoning como narrativas hierárquicas verificadas por clínicos ([Shen et al., 2025](https://arxiv.org/abs/2405.18004)). Isto justifica testar captions como supervisão auxiliar de descrição e reasoning como fonte condicional de estrutura clínica; não permite assumir que todas as frases constituem ground truth visual independente.

O snapshot standalone de [SkinCAP](../../configs/datasets/skincap/README.md), fixado à revisão Hub `4119044b3e14085d7439f88016d93376d433da5f`, foi auditado em 15 de agosto de 2026. Das 4.000 linhas, 439 têm `Do not consider this image = 1`; entre as 3.561 restantes, 124 intersectam Validation e 119 intersectam o Internal Benchmark congelado. Restam 3.318 candidatos técnicos, correspondentes a 3.317 grupos de leakage: 2.683 Fitzpatrick17k e 635 DDI. O transformador versionado `skincap_observation_prefix_v1` remove o sufixo desde o primeiro limite de diagnóstico, gold label, teste ou gestão e aplica guards de comprimento e leakage residual. Aceitou algoritmicamente 3.250 observações (2.649 Fitzpatrick17k e 601 DDI; mediana de 19 palavras) e rejeitou 68. Depois de o responsável pelo projeto atestar possuir autorização escrita para criar os derivados privados, estes 3.250 targets foram materializados. A v0.4.0 foi retirada ao serem encontrados 125 grupos partilhados com splits divergentes entre tarefas; a v0.4.1 corrigida herda sempre o split E1/morfologia e tem zero overlap cruzado. O documento de autorização não é armazenado no repositório.

Existe uma mudança bibliográfica que deve ficar explícita na dissertação: a versão 1 de `arXiv:2405.18004`, submetida em 2024, circulou como SkinCAP e tinha Zhou como primeiro autor; a versão 2, revista em 2025, chama-se SkinCaRe e tem Shen como primeiro autor. O plano e a bibliografia abaixo citam a versão atual. No snapshot local, SkinCoT usa 23 categorias amplas e narrativas hierárquicas.

O [dataset card de SkinCaRe](https://huggingface.co/datasets/yuhos16/SkinCaRe/blob/main/README.md), consultado em 8 de agosto de 2026, confirma a estrutura da release, mas apresenta termos que precisam de resolução antes do uso: a metadata indica `CC-BY-NC-SA-4.0`, o corpo menciona `CC-BY-4.0` e o acordo de acesso restringe distribuição, modificação e derivados. O artigo sustenta a motivação científica; o card e o acordo efetivamente aceite governam o acesso e a utilização dos ficheiros. Por isso, esta fonte permanece condicional mesmo que a sua metodologia seja relevante.

Estas fontes só devem entrar na construção após três verificações:

1. permissão escrita ou licença inequívoca para o uso pretendido e para a eventual distribuição de derivados.
2. remoção de todo o overlap com Validation, Internal Benchmark e restantes grupos selados. DDI deixa de ser um conjunto selado nesta experiência.
3. auditoria do processo de criação do target, porque uma explicação escrita depois de mostrar o gold é uma rationale condicionada pela resposta.

SkinCAP pode alimentar a tarefa `caption` quando o texto for compatível com a imagem e com a taxonomy. SkinCoT não deve ser importado como raw chain-of-thought principal: pode fornecer exemplos de estrutura clínica ou de descrição depois de revisão, normalização e rotulagem explícita de `target_source=human_reviewed_external`. O release local não expõe, por caso, todos os scores numéricos de revisão descritos no artigo. A ausência desses campos impede tratá-los como labels de qualidade individuais. Esta escolha preserva o benefício documentado — linguagem dermatológica mais rica — sem confundir uma rationale revista depois de conhecida a resposta com evidência answer-blind.

### 4.4 Teacher generativo

O teacher selecionado deve gerar apenas targets que não existem em dados humanos. Para cada geração são obrigatórios o nome e revisão do modelo, prompt e respetivo hash, parâmetros de sampling, timestamp, output bruto, output parsed e decisão dos filtros.

O Qwen 3.6 27B é um candidato natural por partilhar a família arquitetural com o Qwen 3.5 4B, mas a seleção final deve resultar do screening já definido na Validation. A literatura de distilação alerta que um teacher melhor no seu próprio benchmark não produz necessariamente o melhor student quando existe mismatch de capacidade ou representação ([When Better Teachers Don't Make Better Students, 2025](https://arxiv.org/abs/2511.17886)). Deve, portanto, comparar-se a utilidade pedagógica dos targets e não apenas a accuracy isolada do teacher.

Um segundo teacher visual, como PanDerm, pode ser usado numa experiência de feature distillation. O PanDerm é relevante porque foi pré-treinado em mais de dois milhões de imagens dermatológicas de várias modalidades e avaliado em 28 benchmarks ([A multimodal vision foundation model for clinical dermatology, 2025](https://doi.org/10.1038/s41591-025-03747-y)). Ainda assim, não deve substituir automaticamente a label humana de SKINCON nem validar texto clínico por si só: a sua evidência sustenta o papel de teacher de representação visual, não o de árbitro universal de cada target textual.

## 5. Regra de leakage e elegibilidade

O treino só pode usar amostras cujo `leakage_group_id` não ocorra em nenhum conjunto reservado. O registo de exclusão deve incluir Validation, Internal Benchmark selado, SkinDisNet e qualquer benchmark futuro. DDI fica fora desta lista a partir da decisão experimental de 14 de agosto de 2026.

O audit local identificou 931 imagens Fitzpatrick17k únicas em grupos reservados e 271 overlaps utilizáveis com SKINCON. Estes 271 casos podem servir para avaliar morfologia, mas não podem produzir targets de treino. A regra aplica-se à imagem exata, a cópias redimensionadas, crops, versões recomprimidas e imagens com o mesmo patient/lesion/source lineage.

Cada amostra passa pela seguinte decisão:

```text
licença permite o uso?
  -> não: excluir
  -> sim: tem identificador ou hash associado a um grupo reservado?
       -> sim: manter apenas no benchmark correspondente
       -> não: existe conflito de label ou duplicado perceptual não resolvido?
            -> sim: revisão manual ou exclusão
            -> não: elegível para sft_train/sft_dev
```

O protocolo de benchmark e isolamento de referências está documentado em [ISEPDermaBench Hugging Face release](../dataset_pipeline/13_isep_dermabench_huggingface_release.md). A publicação do dataset de treino não deve conter tabelas de referência, respostas corretas ou artefactos privados dos benchmarks.

## 6. Modelo lógico: um caso, várias imagens possíveis e várias tarefas

O desenho recomendado mantém uma imagem por linha na primeira release, mas distingue explicitamente o caso clínico do asset visual. Um caso, lesão ou doente pode ter várias fotografias; cada fotografia conserva o seu `sample_id` e `image_asset_id`, enquanto todas partilham o mesmo `case_id` e `leakage_group_id`. Desta forma, as imagens continuam simples para o trainer sem serem incorretamente tratadas como observações clínicas independentes durante os splits.

A mesma imagem pode aparecer visualmente em várias linhas do Dataset Viewer quando essas linhas representam perguntas diferentes. Fotografias diferentes do mesmo caso também podem originar linhas próprias, desde que a relação entre elas permaneça explícita.

Esta separação por capacidade é sustentada por resultados convergentes, embora nenhum estudo imponha este schema de Hugging Face. O SkinGPT-4 separou alinhamento de conceitos e diálogo clínico ([Zhou et al., 2024](https://doi.org/10.1038/s41467-024-50043-3)); MAKE decompôs texto dermatológico em aspetos clínicos alinháveis à imagem ([Yan et al., 2025](https://arxiv.org/abs/2505.09372)); e DermoGPT criou trajetórias distintas para morfologia e diagnóstico ([Ru et al., 2026](https://arxiv.org/abs/2601.01868)). O plano traduz esse princípio numa decisão de dados: uma imagem pode ensinar várias capacidades, mas cada linha deve deixar claro qual capacidade está a ser supervisionada. A forma exata das configurações abaixo é uma opção de engenharia da tese.

```text
canonical_sample (privado/controlo)
  sample_id
  case_id
  image_asset_id
  view_type
  source + source_id
  leakage_group_id
  gold_diagnosis + provenance
  license + hashes
        |
        +-- diagnosis       imagem -> label/Top-K
        +-- morphology      imagem -> conceitos visíveis
        +-- caption         imagem -> descrição curta
        +-- structured      imagem -> observações + diferencial + ação
        +-- open_response   imagem -> resposta clínica natural e curta
        +-- preferences     resposta aceite/rejeitada, opcional
```

Isto não significa armazenar cinco cópias do mesmo ficheiro original. Cada fotografia distinta tem um único `image_asset_id` e um único ficheiro canónico sempre que o formato de publicação o permitir. Os exports Parquet que materializam bytes podem aumentar o tamanho. Esse trade-off deve ser medido no builder em vez de assumido.

### 6.1 Configurações Hugging Face

| Configuração | Unidade de linha | Target visível ao student | Origem principal |
| --- | --- | --- | --- |
| `diagnosis` | uma pergunta diagnóstica por imagem | classe ou diferencial Top-K | gold normalizado. Teacher apenas para ranking auxiliar |
| `morphology` | uma pergunta de perceção por imagem | conceitos e limitações observáveis | SKINCON, Etapa A aceite, revisão humana |
| `caption` | uma pergunta de descrição por imagem | descrição clínica curta | SkinCAP autorizado e filtrado; Etapa A renderizada em releases futuras |
| `structured` | uma pergunta clínica completa por imagem | JSON de observações, diferencial, evidência, incerteza e ação | Etapas A e B aceites |
| `open_response` | uma pergunta aberta por imagem | resposta natural curta | rendering consistente do target canónico |
| `preferences` | um par de respostas para a mesma pergunta | `chosen` e `rejected` | apenas depois de existir revisão/critério fiável |

A primeira release pode publicar apenas as cinco configurações supervisionadas. `preferences` e `opd_rollouts` devem ser adicionadas apenas quando houver uma experiência DPO/on-policy concreta.

### 6.2 Chaves e cardinalidade

- `sample_id` identifica uma fotografia canónica e é igual em todas as configurações derivadas dessa fotografia.
- `case_id` identifica o caso clínico, lesão ou contribuição de origem e pode ligar várias fotografias.
- `task_id` identifica uma pergunta concreta para essa fotografia ou caso.
- `generation_id` identifica uma execução do teacher.
- `image_asset_id` identifica o ficheiro de imagem canónico.
- `sample_id + task_id + target_variant` deve ser único.
- Um `sample_id` pode ter muitas tarefas e várias gerações candidatas, mas apenas uma versão aceite de cada target entra numa release congelada.
- Todos os `sample_id` associados ao mesmo `case_id`, doente ou lesão devem partilhar uma unidade de split indivisível através de `leakage_group_id`.

### 6.3 Casos com várias fotografias

A release v1 não necessita de colocar várias imagens na mesma linha. A representação principal deve continuar a ser uma imagem por exemplo:

| `sample_id` | `case_id` | `image_asset_id` | `view_type` | `leakage_group_id` |
| --- | --- | --- | --- | --- |
| `SCIN_123_IMG_1` | `SCIN_123` | `overview.jpg` | `overview` | `SCIN_123` |
| `SCIN_123_IMG_2` | `SCIN_123` | `closeup.jpg` | `close_up` | `SCIN_123` |
| `SCIN_123_IMG_3` | `SCIN_123` | `angled.jpg` | `oblique` | `SCIN_123` |

Esta decisão permite usar todas as fotografias como exemplos single-image, mas impede que vistas da mesma lesão sejam divididas entre treino, desenvolvimento e avaliação. Quando a aplicação suportar uma conversa em que o modelo pede uma segunda fotografia e a recebe num turno posterior, pode ser adicionada uma configuração opcional `multi_view` ou `interactive_follow_up`. Nessa configuração, uma linha pode conter `image_asset_ids[]`, `view_types[]` e vários turnos multimodais. Essa extensão não deve ser requisito para o primeiro SFT.

## 7. Schema canónico

O registo canónico deve conter mais informação do que aquela mostrada ao modelo. Uma allowlist explícita decide os campos de input de cada tarefa. Tudo o resto serve para controlo, filtragem e auditoria.

### 7.1 Identidade, provenance e split

| Campo | Conteúdo |
| --- | --- |
| `sample_id` | identificador estável, sem semântica clínica |
| `case_id` | identificador que liga fotografias do mesmo caso, lesão ou contribuição |
| `image_asset_id` | ligação ao asset canónico |
| `view_type` | `overview`, `close_up`, `oblique`, `dermoscopic`, `unknown` ou outro valor controlado |
| `image_sha256` | deteção exata de duplicados e integridade |
| `perceptual_group_id` | grupo de imagens visualmente equivalentes |
| `leakage_group_id` | unidade indivisível de split |
| `source_dataset` | Fitzpatrick17k-C, HIBA, PAD-UFES-20, SCIN ou outra fonte elegível |
| `source_sample_id` | ID original para rastreabilidade |
| `license_id` | licença/termos aplicáveis à imagem e ao texto |
| `split` | `sft_train`, `sft_dev` ou `excluded` |
| `exclusion_reasons` | motivos estruturados quando a amostra não é usada |

### 7.2 Gold diagnóstico

| Campo | Conteúdo |
| --- | --- |
| `disease_id` | ID da taxonomy ISEP |
| `gold_diagnosis` | label canónica legível |
| `gold_provenance` | `clinical_diagnosis`, `dermoscopy_supported`, `histopathology_confirmed`, `microbiology_confirmed`, `expert_consensus` ou `unknown_provenance` |
| `taxonomy_version` | versão da mapping table usada |
| `label_conflict_status` | estado de conflitos entre fontes |

`gold_diagnosis` não é sinónimo de observação visual. Serve para target de classificação e para filtro pós-geração, nunca para transformar conhecimento típico da doença num achado supostamente visto no doente.

### 7.3 Etapa A: perceção visual

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
  "dominant_visual_pattern": "pigmented asymmetric lesion",
  "observations": [
    {
      "id": "obs_1",
      "concept": "irregular_border",
      "status": "present",
      "provenance": "clinical_photo",
      "scope": "index_lesion",
      "confidence": "moderate",
      "evidence_region": "lesion_periphery"
    }
  ],
  "not_assessable_features": ["palpable_consistency", "blanching"]
}
```

Estados permitidos para uma observação:

- `present`.
- `absent_in_observed_scope`.
- `uncertain`.
- `not_assessable`.
- `not_shown`.

`absent` isolado deve ser inválido. Uma estrutura fora do enquadramento não foi observada como normal. Simplesmente não foi mostrada.

### 7.4 Etapa B: diferencial, grounding e ação

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
  "concise_clinical_rationale": "O bordo irregular sustenta uma lesão melanocítica atípica, mas a evolução não é observável na fotografia."
}
```

Cada `supporting_observation_id` e `contradicting_observation_id` deve existir na Etapa A. Se a Etapa B discordar de uma observação depois de rever a imagem, a alteração deve ser declarada em `stage_b_corrections`. Uma correção silenciosa quebra a rastreabilidade.

As ações permitidas devem ser fechadas e clinicamente interpretáveis:

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

#### 7.4.1 Supervisão das decisões adaptativas

As ações acima não serão aprendidas apenas com exemplos em que toda a imagem termina numa das 21 doenças. O dataset deve conter tarefas com um target de ação explícito e apenas a informação necessária para justificar essa ação:

| Situação observada | Target principal | Regra de construção |
| --- | --- | --- |
| imagem avaliável e evidência suficientemente discriminativa | `DIAGNOSE_PROVISIONALLY` | usar diagnóstico gold e rationale visual aceite |
| hipóteses visualmente próximas que dependem de história | `REQUEST_CLINICAL_CONTEXT` | pedir uma única informação que discrimine o diferencial; preferir contexto real de SCIN/PAD-UFES |
| desfoque, distância, corte, oclusão ou iluminação inadequada | `REQUEST_OVERVIEW_IMAGE`, `REQUEST_CLOSEUP_IMAGE` ou `ABSTAIN_POOR_QUALITY` | usar imagens de qualidade insuficiente ou degradações controladas; manter original e derivadas no mesmo `leakage_group_id` |
| fotografia sem lesão dermatológica avaliável ou condição fora do âmbito suportado | `ABSTAIN_OUT_OF_DOMAIN` | distinguir não dermatológico de doença dermatológica fora das 21 classes |
| fotografia e contexto insuficientes para uma conclusão segura | `REQUEST_IN_PERSON_EXAM`, `REQUEST_DERMOSCOPY` ou `RECOMMEND_CONFIRMATORY_TEST` | não forçar uma label apenas porque existe no registo privado |

Uma row adaptativa deve ensinar uma ação principal, uma pergunta curta quando aplicável e uma rationale verificável; não deve conter uma chain-of-thought longa. Respostas clínicas reais podem ser usadas como contexto quando a provenance o permite. Respostas simuladas do utilizador devem ser marcadas como sintéticas e nunca apresentadas como história real. A proporção entre diagnóstico, pedido de contexto, pedido de imagem e abstention deve ser escolhida através de um pilot na Validation, medindo tanto diagnóstico excessivo como abstention excessiva.

### 7.5 Resposta estruturada e resposta aberta

O mesmo caso pode ter estas duas colunas:

| Campo | Finalidade |
| --- | --- |
| `structured_response` | JSON canónico usado para parsing, métricas por campo e treino de output estável |
| `open_response` | resposta natural curta usada para comunicação clínica e avaliação open-ended |

A resposta aberta não deve ser uma segunda opinião independente. Deve ser um rendering dos mesmos factos ou uma geração submetida a uma verificação de consistência com o JSON. Durante o treino, é preferível criar dois `task_id` diferentes, um que pede JSON e outro que pede linguagem natural. Mostrar as duas respostas simultaneamente em todos os exemplos desperdiça tokens e incentiva repetição.

Exemplo de resposta aberta:

> A imagem mostra uma lesão pigmentada assimétrica com bordo irregular. Melanoma deve ser considerado, seguido de nevo melanocítico atípico e queratose seborreica. A duração e a mudança recente não podem ser avaliadas na fotografia. É necessário obter esse contexto e considerar avaliação presencial/dermatoscópica.

### 7.6 Provenance da geração e validação

| Grupo | Campos mínimos |
| --- | --- |
| Teacher | `teacher_model`, `teacher_revision`, `prompt_id`, `prompt_sha256`, `generation_parameters` |
| Execução | `generation_id`, `created_at`, `raw_response_uri`, `parsed_response` |
| Validação automática | `schema_valid`, `vocabulary_valid`, `evidence_links_valid`, `unsupported_claim_flags`, `no_image_dependency_result` |
| Comparação com gold | `top1_matches_gold`, `gold_in_topk`, `diagnosis_acceptance` |
| Revisão | `quality_status`, `rejection_reasons`, `reviewer_id`, `review_timestamp`, `review_notes` |

O `raw_response_uri` deve apontar para um store privado de auditoria. O raw reasoning não precisa de ser distribuído nem usado como target.

## 8. Aspeto de uma linha de treino Hugging Face

Cada configuração é exportada para o formato conversacional esperado pelo trainer. Uma linha `structured` pode ter este aspeto simplificado:

```json
{
  "sample_id": "FITZ_000123",
  "task_id": "structured_diagnosis_v1",
  "image": "images/FITZ_000123.jpg",
  "source_dataset": "fitzpatrick17k_c",
  "disease_id": "D004",
  "target_variant": "structured_json",
  "teacher_model": "Qwen/Qwen3.6-27B",
  "teacher_revision": "<frozen_hf_revision>",
  "prompt_id": "structured_two_stage_v1",
  "schema_version": "1.0.0",
  "quality_status": "accepted",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {
          "type": "text",
          "text": "Avalia a imagem, descreve apenas achados visíveis e devolve o diferencial no schema solicitado."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"image_assessment\":{...},\"observations\":[...],\"differential\":[...],\"action\":\"REQUEST_CLINICAL_CONTEXT\"}"
        }
      ]
    }
  ]
}
```

Uma segunda linha com o mesmo `sample_id` pode ter `task_id=open_diagnosis_v1` e `target_variant=open_response`. A imagem é a mesma, mas a pergunta e o target mudam. Esta é a interpretação correta de “várias respostas para a mesma imagem”: várias tarefas de supervisão, não várias respostas contraditórias à mesma pergunta.

### 8.1 Colunas públicas e colunas privadas

Nem toda a informação canónica deve ser publicada no Hub.

**Publicável, quando a licença permitir:**

- `sample_id`, imagem ou referência autorizada.
- configuração, `task_id`, `messages` e target aceite.
- conceitos, resposta estruturada, resposta aberta.
- provenance não sensível do source e da geração.
- estado de qualidade e versão do schema.

**Privado ou apenas para auditoria:**

- identificadores potencialmente reidentificáveis.
- raw outputs extensos e reasoning interno do teacher.
- dados clínicos não necessários ao target.
- comentários livres dos revisores.
- respostas/referências dos benchmarks selados.
- tokens, logits e hidden states que não pertençam a uma release experimental específica.

### 8.2 Estrutura física proposta

```text
ISEPDistillDataset/
  README.md                       # dataset card, licenças, schema e métricas
  data/
    diagnosis/
      train-00000-of-00001.parquet
      dev-00000-of-00001.parquet
    morphology/
      train-00000-of-00001.parquet
      dev-00000-of-00001.parquet
    caption/
      train-00000-of-00001.parquet
      dev-00000-of-00001.parquet
    structured/
      train-00000-of-00001.parquet
      dev-00000-of-00001.parquet
    open_response/
      train-00000-of-00001.parquet
      dev-00000-of-00001.parquet
  images/                         # apenas assets cuja redistribuição é permitida
  metadata/
    taxonomy.json
    schema_versions.json
    prompt_registry.json
    source_licenses.json
    quality_summary.json
```

O store de auditoria de gerações rejeitadas deve ficar fora do repositório público. O dataset card deve declarar quais configurações contêm imagens, quais contêm apenas referências e que termos upstream continuam aplicáveis.

## 9. Pipeline de construção

### Fase 0: congelar inputs e contratos

Antes de chamar o teacher:

1. congelar a versão do `ISEPDermData`, da taxonomy e do ISEPDermaBench.
2. reconstruir a fila de anotação a partir das 7.541 imagens atuais, porque a fila antiga pertence a um corpus diferente.
3. resolver licenças de cada source e de cada target textual externo.
4. executar exact hash, perceptual hash e grouping por source/patient/lesion.
5. criar `sft_train` e `sft_dev` por `leakage_group_id`.
6. congelar os JSON Schemas, vocabulários, prompts e parsers.
7. registar uma allowlist de campos para cada chamada ao teacher.

**Artefacto de saída:** `canonical_samples.parquet`, manifest de exclusões e relatório de leakage com hashes.

### Fase 1: integrar supervisão humana existente

O builder faz joins semânticos e por ID com SKINCON, SkinCAP e outras fontes autorizadas. Cada target recebe `target_source`:

```text
human_expert_annotation
human_reviewed_external
gold_label
teacher_generated
teacher_generated_human_reviewed
derived_rendering
```

SKINCON deve ter prioridade sobre um conceito equivalente gerado pelo teacher. Quando existem divergências, os dois valores permanecem no store de auditoria, mas o target de treino fica marcado para revisão em vez de aceitar silenciosamente o output sintético.

**Artefacto de saída:** targets humanos normalizados e tabela de conflitos.

### Fase 2: pilotar o teacher em 100 casos

O pilot deve ser estratificado por classe, source, dificuldade, qualidade de imagem e skin tone quando disponível. Em cada imagem devem ser comparadas, com os mesmos parâmetros:

| Condição | Input da geração | Pergunta experimental |
| --- | --- | --- |
| `G0_direct` | imagem + prompt direta congelada | qual é o baseline atual? |
| `G1_single_structured` | imagem -> observações e diagnóstico numa chamada | o schema, por si só, melhora qualidade? |
| `G2_text_bottleneck` | A vê imagem. B recebe apenas A | quanto se perde quando B não volta a ver a imagem? |
| `G3_answer_blind_two_stage` | A vê imagem. B vê imagem + A. Nenhuma vê gold | a decomposição auditável melhora grounding sem hindsight? |
| `G4_gold_conditioned` | A answer-blind. B recebe gold | quanto da melhoria é racionalização condicionada pela resposta? |

`G3` é a proposta principal. `G4` deve ser guardada como braço experimental separado, nunca misturada silenciosamente com o corpus answer-blind.

A comparação `G3` versus `G4` existe porque os dois processos respondem a perguntas diferentes. SkinGPT-R1 usa uma descrição inicial sem diagnóstico e uma rationale posterior condicionada pelo gold ([Shen et al., 2025](https://arxiv.org/abs/2511.15242)); esse desenho pode produzir explicações pedagogicamente úteis, mas também pode racionalizar retrospetivamente a resposta. SkinFlow fornece outro precedente para descrever antes de diagnosticar ([Liu et al., 2026](https://arxiv.org/abs/2601.09136)). A proposta ISEP mantém a descrição inicial, mas esconde o gold também da inferência principal, permitindo medir separadamente o benefício de decomposição e o benefício artificial de conhecer a resposta.

O pilot deve medir acceptance rate, erros de schema, unsupported claims, concordância com conceitos humanos, diagnóstico, custo, tokens, latência, truncamentos e diferença entre subgrupos. Só depois desta análise se justifica gerar todo o `sft_train`.

### Fase 3: gerar Etapa A answer-blind

O input permitido contém imagem, modalidade conhecida e instruções de output. Não contém `gold_diagnosis`, caption que revele a doença, disease-specific metadata, texto de benchmark ou respostas anteriores.

A geração deve ser determinista ou de baixa variabilidade na primeira release. Se forem produzidos vários candidatos, o número e a estratégia de seleção devem ser congelados antes de olhar para o resultado final. Não se deve escolher retrospetivamente a resposta que coincide com o gold sem registar esse processo.

Cada output passa por parsing e validação imediata. Os casos não avaliáveis ou fora do domínio podem ser targets úteis para tarefas de qualidade/abstention, mas não produzem diagnóstico afirmativo.

### Fase 4: gerar Etapa B

O input contém:

- a imagem original.
- a Etapa A congelada.
- taxonomy ou lista fechada apenas quando a tarefa o exige.
- contexto clínico exclusivamente quando existe e tem provenance real.

O input continua sem `gold_diagnosis` na condição principal. O teacher pode usar conhecimento médico para comparar hipóteses, mas não pode declarar como observados sinais que só conhece por associação com uma doença.

Se o output pedir contexto, o dataset treina a decisão de perguntar. Uma resposta simulada só pode ser criada a partir de metadata real. Quando essa metadata não existe, não se inventa duração, prurido, exposição, medicação ou sintomas sistémicos.

### Fase 5: comparar com gold e aplicar quality gates

O gold entra pela primeira vez nesta fase. A comparação deve permitir acceptance parcial:

- Etapa A válida e Etapa B correta: aceitar ambas.
- Etapa A válida e Etapa B incorreta: aceitar apenas as tarefas de perceção/descrição.
- Etapa A contém afirmações inventadas: rejeitar A e todos os targets dependentes.
- diagnóstico correto com evidência inválida: aceitar, no máximo, o target simples de classificação. Rejeitar rationale/grounding.
- imagem não avaliável com diagnóstico assertivo: rejeitar diagnóstico e considerar target de abstention.

Isto evita descartar supervisão visual útil e, simultaneamente, evita ensinar justificações incorretas apenas porque o nome final da doença coincidiu.

### Fase 6: criar renderings e exemplos de treino

A representação estruturada aceite é a fonte canónica. A partir dela são criados exemplos curtos:

| Exemplo | Input | Target |
| --- | --- | --- |
| qualidade | imagem | avaliável, defeitos e limitações |
| morfologia | imagem | conceitos observáveis normalizados |
| descrição | imagem | caption clínica curta |
| diagnóstico fechado | imagem + 21 classes | Top-K canónico |
| diagnóstico aberto | imagem | Top-3 em linguagem natural |
| grounding | imagem + hipóteses | evidence links por hipótese |
| ação interativa | imagem + estado | classificar, pedir informação/imagem/exame ou abster |
| resposta estruturada | imagem | JSON compacto completo |

Uma amostra pode contribuir para algumas tarefas e não para outras. Por exemplo, uma imagem sem scale pode ensinar `has_scale=false`, morfologia e pedido de nova imagem, mesmo que não seja segura para um target diagnóstico.

A criação de renderings curtos segue a motivação de MAKE — alinhar aspetos clínicos específicos em vez de depender de uma caption monolítica ([Yan et al., 2025](https://arxiv.org/abs/2505.09372)) — e a separação explícita entre supervisão morfológica e diagnóstica usada em DermoGPT ([Ru et al., 2026](https://arxiv.org/abs/2601.01868)). A literatura sustenta a decomposição; a decisão de materializar cada capacidade como uma configuração separada continua a ser uma hipótese de implementação a testar pelas ablations D0–D4.

### Fase 7: revisão humana e freeze

O pilot de 100 casos deve ter revisão especializada sempre que possível. Na geração em escala, a primeira política proposta é rever:

- 100% dos casos com conflito entre gold e teacher.
- 100% dos casos com `clinical_risk_if_missed=high` e baixa confiança.
- 100% das correções de Etapa B e dos flags de unsupported claims.
- uma amostra estratificada dos casos aceites por classe, source e skin tone.
- classes raras e subgrupos com acceptance rate anormal.

A percentagem final de revisão dos casos aparentemente normais deve ser definida com o orientador e, idealmente, com disponibilidade clínica. Não existe base suficiente para apresentar um valor arbitrário como padrão científico universal.

Depois da revisão, a release recebe version tag, manifest de hashes, estatísticas, prompts, revisão dos modelos, relatório de rejeições e dataset card.

## 10. Quality gates

### Gate 1: legalidade e provenance

- licença e termos registados.
- source e ID original resolvidos.
- ausência de PHI ou identificadores não necessários.
- permissão para redistribuir imagem e/ou target derivado.

Falha neste gate exclui a amostra da publicação e, quando os termos também proibirem treino, do próprio corpus.

### Gate 2: integridade e leakage

- hash exato calculado.
- grupo perceptual resolvido.
- `leakage_group_id` não pertence a nenhum conjunto reservado.
- conflito de label resolvido.
- um único split por grupo.

### Gate 3: schema e vocabulário

- JSON válido.
- todos os campos obrigatórios presentes.
- enums dentro do vocabulário.
- ranks únicos e contíguos.
- disease IDs válidos na taxonomy congelada.
- nenhuma referência a observation ID inexistente.

### Gate 4: grounding visual

- toda a evidência diagnóstica aponta para uma observação da Etapa A.
- nenhum achado requer história, palpação, dermoscopia ou histologia quando essa fonte não foi fornecida.
- `not_shown` não é convertido em `absent`.
- cor, escala e distribuição respeitam as limitações da aquisição.
- as correções da Etapa B são explícitas.

### Gate 5: dependência da imagem

Uma amostra de outputs deve ser comparada com condições sem imagem, imagem desfocada ou crop irrelevante. Uma resposta que permanece específica e confiante sem informação visual suficiente é suspeita de prior linguístico ou label leakage.

Este teste não decide sozinho se uma linha é correta, mas fornece flags para auditoria e mede se o dataset ensina efetivamente vision grounding.

### Gate 6: consistência clínica e gold

- Top-1 equivalente ao gold para aceitar o target diagnóstico principal.
- Top-K sem duplicados e clinicamente plausível.
- rationale curta consistente com observações e ranking.
- incerteza e ação compatíveis com qualidade da imagem e missing discriminators.
- nenhuma recomendação clínica afirma resultados de testes ainda não realizados.

Quando o Top-1 não coincide, os targets de morfologia podem ser aceites separadamente. Esta decisão deve ficar registada em `accepted_components`.

### Gate 7: consistência entre JSON e resposta aberta

- mesmo Top-1 e mesma ordem de diferenciais, salvo instrução explícita.
- nenhum achado novo na resposta aberta.
- mesma incerteza e mesma ação.
- ausência de tratamento ou prognóstico não presente no target canónico.
- comprimento abaixo do limite definido para a tarefa.

### Gate 8: equilíbrio do corpus

O relatório de qualidade deve apresentar, por classe, source, modalidade e skin tone quando disponível:

- número elegível, gerado, aceite e rejeitado.
- acceptance rate.
- motivos de rejeição.
- cobertura de conceitos.
- comprimento e tokens dos targets.
- origem humana ou sintética.
- proporção de ações de abstention/pedido de contexto.

Se o teacher só produzir respostas aceites nos casos fáceis, o dataset amplifica esse viés. A solução não é baixar os filtros. É rever amostragem, prompt, teacher ou cobertura humana.

## 11. Estados de processamento

Cada geração deve atravessar estados explícitos:

```text
queued
  -> generated
  -> parsed | parse_failed
  -> schema_valid | schema_rejected
  -> grounding_valid | grounding_rejected
  -> gold_checked
  -> accepted | partially_accepted | rejected | needs_human_review
  -> frozen_in_release
```

Nenhum script deve inferir `accepted` apenas porque o parser conseguiu ler JSON. Validade sintática, grounding e correção clínica são verificações diferentes.

## 12. Mistura de treino

### 12.1 Princípio

O student não deve ver sempre o mesmo template completo. Uma mistura de tarefas força a reutilização de capacidades e permite diagnosticar onde surgiu o ganho.

A proporção exata de cada configuração deve ser tratada como hiperparâmetro. Como ponto de partida de pilot, a resposta estruturada pode ser dominante e a resposta aberta representar uma fração minoritária, por exemplo 10% a 20% dos exemplos clínicos completos. Este intervalo é uma decisão de engenharia a validar por ablation, não um resultado publicado da literatura.

### 12.2 Controlo de repetição por imagem

Quando uma imagem origina muitas tarefas, um sampler ingênuo pode sobrepesar esse caso. O loader deve controlar o número de exemplos por `sample_id` em cada epoch ou usar sampling por configuração e depois por caso. A contagem principal de cobertura deve ser apresentada tanto em linhas como em imagens únicas.

### 12.3 Currículo proposto

1. **Alinhamento visual básico:** qualidade, morfologia e caption curta.
2. **Diagnóstico supervisionado:** label e Top-K.
3. **Grounding clínico:** diferencial, evidence links, incerteza e ação.
4. **Formatos de resposta:** alternância controlada entre JSON e open response.
5. **KD opcional:** features/logits/on-policy apenas depois de congelar o baseline SFT.

Esta progressão combina o alinhamento visual-conceptual do [SkinGPT-4](https://doi.org/10.1038/s41467-024-50043-3), a decomposição por aspetos do [MAKE](https://arxiv.org/abs/2505.09372), a separação morfologia-diagnóstico do [DermoGPT](https://arxiv.org/abs/2601.01868), a fase descritiva do [SkinFlow](https://arxiv.org/abs/2601.09136) e o treino faseado do [LLaVA-KD](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html). Nenhum desses estudos testa a sequência exata proposta nesta população. Por isso, o currículo deve ser comparado com treino misturado desde o início, mantendo exemplos, compute e avaliação comparáveis, para separar o efeito da ordem do efeito de mais passos de otimização.

## 13. Knowledge distillation para o Qwen 3.5 4B

### 13.1 O que o dataset estático consegue transferir

O dataset transfere comportamento observável:

- vocabulário morfológico.
- descrições visuais.
- diferenciais e ranking.
- ligação entre evidência e hipótese.
- incerteza, pedidos de contexto e abstention.
- estilo de resposta e formato JSON.

Isto é response distillation por SFT. Pode melhorar a utilização da visão do student, mas não torna automaticamente o encoder de 4B igual ao encoder do Qwen 3.6 27B.

### 13.2 O que exige treino model-side

Para aproximar representações visuais são necessários objetivos adicionais:

- feature matching entre tokens visuais do teacher e do student, com projection quando as dimensões diferem.
- relation distillation entre pares de tokens/regiões.
- alinhamento seletivo de text-to-vision attention.
- reverse-KL ou on-policy distillation sobre respostas do student.

Estas opções correspondem a mecanismos já estudados, mas com conclusões condicionais. LLAVADI reporta vantagens em combinar transferência de logits/tokens e representações ([Xu et al., 2024](https://arxiv.org/abs/2407.19409)); LLaVA-KD organiza a transferência multimodal e relacional em fases ([Cai et al., 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html)); e Align-KD seleciona layers e conhecimento de alinhamento cross-modal em vez de tentar copiar indiscriminadamente todo o teacher ([Feng et al., 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Align-KD_Distilling_Cross-Modal_Alignment_Knowledge_for_Mobile_Vision-Language_Large_Model_CVPR_2025_paper.html)). Isto sustenta testar feature/KD seletiva, não afirmar antecipadamente que igualará a visão do Qwen 27B.

Qwen 3.5 4B e Qwen 3.6 27B têm geometrias de output visual e larguras textuais diferentes. Uma troca direta da torre visual não é plug-and-play. Feature distillation com adapter é mais fácil de isolar e reverter.

### 13.3 Artefactos opcionais, separados do core

```text
distillation/
  response_targets/          # já representados nas configs SFT
  teacher_features/          # apenas IDs + shards comprimidos, se a experiência avançar
  teacher_logits/            # tokens selecionados, não vocabulário completo por defeito
  opd_rollouts/              # resposta do student + avaliação/distribuição do teacher
  manifests/                 # modelo, layer mapping, projection, hashes e losses
```

Os artefactos devem ser ligados por `sample_id`, `task_id` e `generation_id`, mas versionados separadamente. Assim, o dataset clínico continua legível e reutilizável sem obrigar qualquer utilizador a descarregar estados internos de um modelo específico.

### 13.4 Ordem recomendada das experiências de distilação

1. Avaliação do student original, sem fine-tuning.
2. SFT apenas com `diagnosis`; quando o target vem do ground truth, esta condição é supervisão convencional e não distilação.
3. SFT multitask com `morphology`, `caption`, `structured` e `open_response`.
4. Hard response distillation com respostas filtradas do teacher, mantendo a arquitetura do student inalterada.
5. A condição 4 mais soft distillation sobre probabilidades das classes canónicas.
6. SFT multitask mais feature distillation visual seletiva, apenas se a análise de erros indicar um bottleneck visual.
7. SFT multitask mais on-policy/reverse-KL, se a distilação estática saturar.
8. combinação das condições avançadas apenas quando cada componente melhorar isoladamente.

Por defeito, a soft distillation deve usar `class_logprobs` sobre a taxonomia canónica, e não logits completos sobre todo o vocabulário em cada token. Os nomes de doenças podem ocupar vários tokens; por isso, cada score deve corresponder à probabilidade condicional da label completa, calculada com o mesmo prompt e template. Logits token-level Top-K podem ser guardados em shards separados como experiência posterior. O vocabulário completo não deve ser materializado como uma coluna do dataset core.

O maior teacher deve competir com pelo menos um teacher alternativo ou com targets humanos onde existam. O critério é quanto o student aprende, não apenas a accuracy isolada do teacher.

Se a resposta distillation estática saturar, reverse-KL e geração on-policy podem ser avaliadas porque MiniLLM e Generalized Knowledge Distillation procuram reduzir o mismatch entre sequências do teacher e estados realmente visitados pelo student ([Gu et al., 2024](https://arxiv.org/abs/2306.08543); [Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)). Estes estudos são sobretudo de linguagem; no ISEP, o ganho multimodal e o grounding visual terão de ser demonstrados, não presumidos.

## 14. Estratégia para melhorar a visão do student

O dataset contribui para a visão através de supervisão explícita, mas a investigação arquitetural deve seguir um percurso de risco crescente:

| Braço | Alteração | Pergunta |
| --- | --- | --- |
| `V0` | Qwen 3.5 4B sem alteração | baseline reproduzível |
| `V1` | `V0` + SFT de morfologia/descrição | os targets visuais já resolvem parte do bottleneck? |
| `V2` | `V1` + projector ou LoRA em late vision blocks | uma adaptação pequena melhora features? |
| `V3` | `V1` + resolução/crop/multi-scale com compute equivalente | o problema é detalhe/localização? |
| `V4` | `V1` + feature distillation de Qwen 27B ou PanDerm | um teacher visual transfere informação adicional? |
| `V5` | `V1` + piloto `FDLinear` | o mecanismo de SkinFlow acrescenta valor além de V2–V4? |

`FDLinear` não deve entrar no baseline oficial. SkinFlow não compara esse operador com as alternativas simples acima e não fornece detalhe suficiente para uma reprodução inequívoca. Se V1–V4 não resolverem um erro visual persistente, V5 torna-se uma contribuição experimental defensável.

A ordem V1–V5 coloca primeiro a supervisão de conceitos porque SKINCON foi criado para explicações e debugging baseado em conceitos ([Daneshjou et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html)). Só depois introduz um teacher visual de domínio, apoiado pela escala e resultados de PanDerm ([A multimodal vision foundation model for clinical dermatology, 2025](https://doi.org/10.1038/s41591-025-03747-y)), e finalmente o operador experimental de SkinFlow ([Liu et al., 2026](https://arxiv.org/abs/2601.09136)). Deste modo, se houver melhoria, a tese consegue distinguir o efeito de melhores targets visuais do efeito de alterar a arquitetura.

### 14.1 Análise visual before/after do student

Além das métricas de benchmark, a tese deve incluir uma análise interpretativa reproduzível de como a atribuição visual do Qwen 3.5 4B muda ao longo do treino. Esta análise não é uma nova benchmark principal e não deve ser apresentada como prova de raciocínio interno. O objetivo é produzir uma vista programática, comparável e auditável das regiões da imagem associadas a um output clínico específico.

O conjunto principal deve conter 21 casos fixos da Validation, idealmente um por classe, escolhidos antes de observar os heatmaps e equilibrados tanto quanto possível por source e tom de pele. Os mesmos bytes, prompt e target devem ser usados em todos os checkpoints:

```text
E0_base
  -> E1_label
  -> E2_structured
  -> E3_hard_kd
  -> E5_vision, se existir
  -> E6_final
```

Para cada caso e checkpoint devem ser geradas duas atribuições:

1. uma atribuição para a log-probability da mesma label gold, permitindo comparação direta entre checkpoints;
2. uma atribuição para a label efetivamente prevista pelo checkpoint, mostrando a evidência associada ao seu comportamento real.

Comparar apenas o segundo mapa pode ser enganador quando o modelo base prevê eczema e o modelo final prevê psoriasis, porque os heatmaps estariam condicionados por targets diferentes.

#### Execução programática

A análise deve usar PyTorch/Transformers diretamente, com o modelo em `eval()` mas gradients ativados. Não deve usar a API OpenAI-compatible do vLLM como mecanismo principal: o endpoint é adequado para geração rápida, mas não expõe de forma estável os gradients e ativações necessários. A implementação deve ser um módulo separado da pipeline de benchmark, com aproximadamente esta responsabilidade:

```text
src/vision_analysis/
  cli.py                 seleção de casos, checkpoints e métodos
  model_adapter.py       carregamento e hooks específicos do Qwen
  targets.py             score teacher-forced da label gold/predita
  attribution.py         gradient × activation e occlusion
  layers.py              early/middle/late vision e fusão multimodal
  render.py              overlays e painéis comparativos
  report.py              HTML e manifest auditável
```

O fluxo por caso é:

```text
imagem + prompt congelado
  -> forward teacher-forced da label alvo
  -> score = soma da log-probability dos tokens da label
  -> gradient do score relativamente aos visual patch embeddings
  -> importância por patch
  -> reshape para a grelha visual
  -> interpolação para a resolução original
  -> raw map + overlay + metadata
```

Os hooks devem observar posições normalizadas da componente visual — aproximadamente 25%, 50%, 75% e 100% dos blocos — e a representação depois da fusão visual-textual, em vez de depender de nomes frágeis de layers. A disponibilidade exata de attentions e hidden states deve ser verificada contra a implementação Transformers da revisão congelada do Qwen. O método primário deve ser `gradient × activation` sobre visual tokens, acompanhado por patch occlusion como teste de sensibilidade. Attention rollout e relevance propagation podem ser visualizações secundárias.

O script deve processar um checkpoint de cada vez com batch size 1, gravar cada caso atomicamente e só depois libertar o modelo e carregar o checkpoint seguinte. Um GPU de 24 GB pode ser suficiente para o Qwen 4B e attribution simples; 48 GB oferece margem mais segura para eager attention, alta resolução e layers intermédias. Não é necessário um H100 para 21 casos. Num RunPod, os resultados devem ser sincronizados periodicamente para o workspace local, seguindo o mesmo princípio de checkpoint durável usado nas benchmarks.

Outputs mínimos:

```text
outputs/vision_analysis/<analysis_id>/
  manifest.json
  cases/<case_id>/<checkpoint>/<target>/
    attribution.npy
    overlay.png
    metadata.json
  report.html
```

`metadata.json` deve registar checkpoint hash, model revision, processor revision, prompt hash, target textual, tokens do target, score, layer, método, normalização, resolução e seed quando aplicável. O relatório deve mostrar a imagem original ao lado dos checkpoints; o overlay nunca deve ser a única vista porque pode ocultar a própria lesão.

#### Interpretação e limites

[Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) estabeleceu o uso de gradients para localizar regiões associadas a um conceito; para Transformers multimodais, [Chefer, Gur e Wolf (2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html) propõem relevance propagation para arquiteturas bi-modais e encoder-decoder. [Attention rollout](https://aclanthology.org/2020.acl-main.385/) permite acompanhar fluxo de informação através das layers, mas atenção bruta não deve ser confundida com uma explicação causal: [Jain e Wallace (2019)](https://aclanthology.org/N19-1357/) mostram que distribuições de atenção diferentes podem produzir previsões equivalentes, e [Adebayo et al. (2018)](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) demonstram que alguns mapas visualmente plausíveis são insensíveis ao modelo ou aos dados. Patch occlusion e sanity checks são, portanto, obrigatórios para qualquer interpretação forte.

Há evidência dermatológica direta para estudar alinhamento visual. [SKINCON](https://papers.nips.cc/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html) fornece conceitos dermatológicos para debugging e explicações baseadas em conceitos. Um sistema alinhado com regiões e características selecionadas por dermatologistas aumentou a confiança diagnóstica e a confiança no suporte, embora o primeiro estudo não tenha demonstrado ganho adicional significativo de accuracy sobre AI sem explicações ([Chanda et al., 2024](https://www.nature.com/articles/s41467-023-43095-4)); um estudo posterior com eye-tracking reportou mais 2,8 pontos percentuais de balanced accuracy com XAI ([Chanda et al., 2025](https://www.nature.com/articles/s41467-025-59532-5)). Uma comparação de 2026 encontrou correlação mediana de 0,540 entre mapas da AI e dermatologistas, face a 0,591 entre dermatologistas e 0,434 no controlo não homólogo ([Kremer et al., 2026](https://pubmed.ncbi.nlm.nih.gov/41490767/)). Estes resultados justificam a análise, mas não transformam um heatmap isolado em validação clínica.

Na primeira versão, os 21 painéis são uma case study qualitativa. Se posteriormente existirem lesion masks ou regiões de conceitos anotadas, a mesma pipeline pode calcular energia dentro da lesão, pointing-game accuracy, IoU/Dice e queda do score após remoção das regiões mais salientes. Só essa versão congelada e quantitativa deve ser descrita como avaliação de localização auxiliar. A mudança de representações entre layers também pode ser estudada com CKA ([Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html)), sem assumir que semelhança representacional equivale a grounding clínico.

## 15. Ablations obrigatórias

### 15.1 Conteúdo do dataset

| Variante | Supervisão |
| --- | --- |
| `D0_label` | imagem + label |
| `D1_concepts` | D0 + morfologia/qualidade |
| `D2_grounded` | D1 + diferencial, evidence links e rationale curta |
| `D3_open` | D2 + resposta aberta consistente |
| `D4_adaptive` | D3 + pedir contexto/imagem/exame ou abster |
| `D5_raw_cot` | raw CoT do teacher, apenas braço de investigação |

`D5_raw_cot` não deve ser a variante principal. A comparação serve para testar se texto longo acrescenta valor ou apenas tokens, truncamentos e afirmações não suportadas.

### 15.2 Protocolo do teacher

Comparar `G0` a `G4`, definidos na Fase 2, com os mesmos casos, modelo, temperatura, limite de tokens e parser. Isto separa o efeito de duas chamadas, schema, acesso à imagem e acesso ao gold.

### 15.3 Arquitetura e distilação

Comparar `V0` a `V5` e as fases de KD aplicáveis com budget de compute, resolução, número de epochs, seed e dados controlados. Uma arquitetura com mais treino ou mais pixels não deve ser apresentada como superior sem essa diferença estar declarada.

### 15.4 Sequência experimental principal da tese

As experiências não devem começar com todos os tipos de supervisão e todas as alterações arquiteturais ativos. A sequência principal deve introduzir uma alteração de cada vez, mantendo splits, student base, protocolo de avaliação e, tanto quanto possível, budget de otimização comparáveis. Esta organização permite atribuir um eventual ganho ao componente que foi acrescentado.

| Experiência | Condição | Alteração introduzida | Pergunta principal |
| --- | --- | --- | --- |
| `E0_base` | student original, sem treino | nenhuma | qual é o desempenho inicial do Qwen 4B? |
| `E0_vision` | classificador visual convencional ou linear probe, se viável | baseline não generativo | a componente multimodal generativa acrescenta valor ao diagnóstico? |
| `E1_label` | `D0_label` | imagem → diagnóstico gold | a supervisão diagnóstica simples é suficiente? |
| `E2_structured` | `D1_concepts` e depois `D2_grounded`/`D3_open` | conceitos, evidência, diferenciais e rationale curta | supervisão clínica estruturada melhora diagnóstico e grounding? |
| `E3_hard_kd` | melhor variante de E2 + targets do teacher | response distillation | as respostas filtradas do teacher acrescentam conhecimento transferível? |
| `E4_soft_kd` | E3 + `class_logprobs` | distribuição do teacher sobre as classes | a incerteza do teacher melhora ranking e calibração? |
| `E5_vision` | melhor variante de E1–E4 + um braço V2–V5 | adaptação ou distilação visual | o erro residual provém de uma limitação da visão? |
| `E6_final` | apenas componentes vencedores | combinação final | qual é a melhor solução completa sob o mesmo protocolo? |

`E0_vision` é um controlo recomendado, não uma substituição do student. Permite saber se um classificador de imagem especializado resolve a classificação melhor do que o MLLM e ajuda a separar o valor diagnóstico do valor adicional de gerar explicações, diferenciais e ações.

Em `E2_structured`, os subcomponentes devem continuar identificáveis. Se `D2_grounded` e `D3_open` forem adicionados simultaneamente, a tese só poderá concluir sobre o pacote completo. Quando o budget permitir, a comparação incremental `D0 → D1 → D2 → D3` é preferível.

Em `E3_hard_kd`, o raw CoT não deve ser necessário para declarar response distillation. O braço principal usa uma `clinical_rationale` curta, verificável e ligada à evidência. `teacher_reasoning_trace` fica privado, marcado com `thinking_enabled`, `gold_visible_to_teacher` e provenance, e entra apenas através de `D5_raw_cot`. Uma explicação pedida com thinking desligado é `clinical_rationale`, não um reasoning interno recuperado.

O avanço entre experiências deve ser decidido na Validation, com métricas primárias congeladas e análise de variância. O Internal Benchmark permanece selado até `E6_final`. Cada comparação deve conservar os mesmos IDs e reportar, no mínimo, qualidade diagnóstica, grounding, calibração, subgrupos e eficiência. Quando não for possível repetir várias seeds, a execução única e a incerteza resultante devem ser declaradas como limitação, não omitidas.

Esta sequência é uma matriz de ablação de alto nível. Os identificadores `D*`, `G*` e `V*` continuam a representar, respetivamente, conteúdo do dataset, protocolo do teacher e alteração visual. Assim, por exemplo, `E3_hard_kd` pode declarar explicitamente que utilizou `D2 + G3 + V1`, evitando que o nome da fase esconda a configuração real.

## 16. Avaliação

### 16.1 Diagnóstico

- Top-1, Top-3 e Top-6 canónicos.
- MRR e posição do gold.
- matriz de confusão por classe.
- calibração/confiança.
- cobertura e accuracy quando o modelo decide diagnosticar.
- desempenho open-ended com juiz cego e regras congeladas.

### 16.2 Perceção e grounding

- F1 por conceito e macro-F1 em SKINCON elegível.
- precisão/recall de achados visuais revistos.
- validade dos evidence links.
- unsupported-finding rate.
- diferenças entre com imagem, sem imagem e imagem degradada.
- localization metric num subset com masks, se vier a existir.

Attention maps podem ser exploradas, mas não provam causalmente que o modelo usou a lesão correta. SkinFlow é uma razão adicional para medir localização, não para tratar heatmaps como validação clínica.

### 16.3 Ação e segurança

- decisão correta entre diagnosticar, pedir informação, pedir imagem, pedir exame ou abster.
- adequação do discriminating question.
- taxa de diagnóstico assertivo em imagem não avaliável.
- false reassurance em casos de risco.
- consistência entre confiança diagnóstica, risco e urgência.

### 16.4 Equidade e generalização

- resultados por skin tone quando disponível.
- resultados por source e tipo/modalidade de imagem.
- classes inflamatórias, infeciosas e neoplásicas.
- SkinDisNet e outros conjuntos não usados no treino como avaliação externa. DDI é apenas uma análise in-domain/contaminada depois da integração SKINCON.
- diferença entre Validation de desenvolvimento e Internal Benchmark selado.

### 16.5 Eficiência

- tokens e latência do teacher por amostra.
- acceptance rate por milhar de gerações.
- tempo e custo de revisão humana.
- parâmetros treináveis, VRAM, throughput e latência do student.
- tamanho das configurações e artefactos de distilação.

### 16.6 Retenção de conhecimento médico e geral

Além do ISEPDermaBench, DermoBench e SkinDisNet, a avaliação final pode
incluir **MedQA**, **PubMedQA** e os domínios médicos de **MMLU**. Estes
benchmarks têm uma função diferente: não medem perceção dermatológica nem
grounding visual; medem conhecimento médico textual, resolução de perguntas
determinísticas e eventual perda de capacidades após especialização ou
distilação.

O MMLU original reúne 57 tarefas académicas e profissionais e foi proposto
para medir simultaneamente conhecimento e problem solving
([Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300)). Para esta tese, o
núcleo relevante deve ser pré-registado como `anatomy`, `clinical_knowledge`,
`college_medicine`, `medical_genetics` e `professional_medicine`. O MMLU
completo pode ser executado como análise secundária de retenção geral, mas não
deve ser agregado às métricas dermatológicas.

O MedQA foi construído a partir de exames médicos profissionais e inclui
perguntas em inglês, chinês simplificado e chinês tradicional
([Jin et al., 2020](https://arxiv.org/abs/2009.13081)). A condição principal
deve congelar antecipadamente a variante linguística, release, split, número
de opções, prompt e regime zero-shot/few-shot. A versão inglesa é o candidato
natural para comparação com os restantes benchmarks, mas esta escolha só deve
ser formalizada depois de verificar licença, formato e overlap conhecido.

O PubMedQA avalia perguntas biomédicas `yes/no/maybe` a partir do abstract do
artigo correspondente. O dataset original contém 1.000 instâncias anotadas por
especialistas, além de subconjuntos não anotados e artificialmente gerados
([Jin et al., 2019](https://arxiv.org/abs/1909.06146)). A avaliação da tese deve
usar apenas o subconjunto expert-labeled (`PQA-L`) e congelar a release, o split,
o prompt e o parser; os subconjuntos sintéticos não devem ser misturados na
métrica principal.

O fluxo correto distingue seleção interna de checkpoint e avaliação externa:

```text
Training objective
        ↓
Small medical MLLM
        ↓
SFT Dev determinístico
(label accuracy, macro-F1, balanced accuracy e eval loss)
        ↓
checkpoint selection
        ↓
checkpoint de fase congelado
        ↓
┌──────────────────────────────────────┐
│ Avaliações determinísticas externas  │
│                                      │
│ MedQA                                │
│ MMLU Clinical Knowledge              │
│ MMLU Professional Medicine           │
│ MMLU Medical Genetics                │
│ MMLU Anatomy / College Medicine      │
│ PubMedQA PQA-L                       │
└──────────────────────────────────────┘
        ↓
retenção, transferência e forgetting
(sem voltar a escolher o checkpoint)
```

A comparação recomendada é:

1. Qwen 3.5 4B base, antes de qualquer treino;
2. checkpoint vencedor de cada fase relevante (`E1`, `E2`, `E3` e `E4`),
   apenas quando essa fase for declarada concluída;
3. teacher open-weight usado na distilação;
4. student final distilled;
5. opcionalmente, o modelo grande de referência sob o mesmo prompt e decoding.

Esta grelha permite medir três quantidades separadas: ganho dermatológico,
retenção de conhecimento médico e aproximação do student ao teacher. Deve ser
reportada accuracy exact-match, taxa de outputs inválidos, resultado por
subdomínio, diferença relativamente ao student base e diferença relativamente
ao teacher, acompanhadas por latência, VRAM e número de parâmetros. Uma subida
em MedQA/MMLU não prova melhoria visual; uma descida não invalida por si só o
especialista, mas quantifica o custo da especialização.

MedQA, PubMedQA e MMLU **não entram na seleção de checkpoint, learning rate,
mistura de dados, prompt ou método de distilação**. Consultá-los repetidamente
para tomar essas decisões transformá-los-ia em conjuntos de desenvolvimento.
Devem ser executados apenas sobre modelos já congelados, com o mesmo protocolo
entre teacher e student. Como são benchmarks públicos e amplamente usados, os
resultados devem ainda ser apresentados com uma nota explícita sobre possível
contaminação de pretraining e não como validação clínica.

## 17. Protocolo de utilização dos splits

```text
sft_train
  -> construção dos targets e otimização

sft_dev
  -> early stopping, seleção de checkpoint e debugging do trainer

Validation ISEPDermaBench
  -> comparação de prompts, teachers e variantes durante desenvolvimento

Internal Benchmark selado
  -> uma avaliação final depois de congelar dados, modelo e decisão

DDI + SkinDisNet
  -> generalização externa, nunca geração de targets de treino

MedQA + PubMedQA PQA-L + MMLU médico
  -> retenção textual e transferência teacher-student depois de congelar cada fase
  -> nunca seleção de checkpoint ou tuning retroativo
```

O Internal Benchmark não deve ser consultado para escolher mistura de tarefas, prompt, temperatura, checkpoint ou arquitetura. Se for usado repetidamente, deixa de ser um teste selado e deve ser substituído ou declarado como conjunto de desenvolvimento.

## 18. Critérios de avanço entre fases

### Gate A: corpus canónico pronto

- 100% das linhas têm source, hashes, license status e `leakage_group_id`.
- zero overlap conhecido entre `sft_*` e benchmarks.
- conflitos de labels resolvidos ou excluídos.
- splits e taxonomy congelados.

### Gate B: protocolo de geração aprovado

- pilot estratificado concluído.
- Etapa A e Etapa B parseiam de forma fiável.
- grounding auditável e sem uso do gold na condição principal.
- custos e truncamentos compatíveis com escala.
- revisão confirma que `G3` é pelo menos preferível a `G0/G1` nas métricas clínicas prioritárias.

Os limiares numéricos devem ser pré-registados depois de medir o baseline do pilot. Inventar agora uma melhoria mínima sem estimativa de variância daria uma aparência de rigor sem fundamento.

### Gate C: release de treino pronta

- quality gates executados em todas as linhas.
- acceptance/rejection report por subgrupo.
- sample de revisão humana concluído.
- JSON/open response consistentes.
- dataset card, versão, hashes e prompts incluídos.

### Gate D: treino reproduzível

- collator multimodal e assistant-only loss verificados.
- overfit smoke test num subset pequeno.
- baseline `D0/V0` reproduzido.
- seeds, hyperparameters, checkpoints e métricas guardados.

### Gate E: complexidade adicional justificada

Feature KD, on-policy distillation, Vision LoRA, multi-scale ou `FDLinear` só avançam quando existe uma hipótese ligada a um erro medido no baseline. Cada alteração deve ter uma ablation com compute comparável.

## 19. Riscos e medidas de controlo

| Risco | Consequência | Controlo |
| --- | --- | --- |
| Gold leakage | rationales convincentes mas não fiéis à imagem | answer-blind nas Etapas A/B e comparação com gold apenas depois |
| Teacher hallucination | sinais, história ou exames inventados | evidence IDs, provenance, schema e revisão |
| Dataset composto por casos fáceis | falsa melhoria e pior generalização | acceptance report estratificado e revisão de rejeitados |
| Duplicados entre treino e benchmark | estimativa otimista | exact/perceptual hashing e split por grupo |
| Licenças incompatíveis | impossibilidade de publicar ou treinar | registry por asset e exclusão fail-closed |
| Overweight de imagens com muitas tarefas | memorização de poucos casos | sampler por `sample_id` e relatório de imagens únicas |
| JSON excessivamente rígido | pior resposta natural | config `open_response` e alternância por tarefa |
| Raw CoT longo | truncamento, exposição de artefactos e claims não verificáveis | rationale curta. Raw output privado para auditoria |
| Teacher demasiado diferente | KD ineficiente | teacher selection pela aprendizagem do student |
| Alteração visual prematura | custo e conclusão causal ambígua | baseline, targets visuais e alternativas simples antes de FDLinear |
| Reutilização de MedQA/MMLU durante tuning | benchmark externo convertido em validation e resultado otimista | executar apenas modelos de fase congelados e proibir seleção retroativa |
| Contaminação de benchmarks públicos | resultado textual sobrestima generalização | declarar risco de pretraining overlap e não interpretar como validação clínica |

## 20. Reprodutibilidade e versionamento

Cada release deve congelar:

- commit/revision de todos os datasets upstream.
- taxonomy, mappings e vocabulários.
- IDs dos benchmarks excluídos e hash do exclusion manifest.
- modelo/revision do teacher e chat template.
- prompt text, `prompt_id` e SHA-256.
- sampling parameters e versão do inference engine.
- versão dos JSON Schemas e do parser.
- regras de acceptance e respetiva implementação.
- contagens antes/depois de cada filtro.
- seed e algoritmo de split.
- revisão humana e protocolo de disagreements.
- hash de cada Parquet e asset publicado.

O dataset card deve explicar que respostas sintéticas são supervisão gerada e filtrada, não pareceres médicos nem prova de reasoning humano. Deve ainda declarar as limitações de cobertura, labels, demographic metadata e licenças.

## 21. Sequência de implementação recomendada

| Ordem | Entrega | Definição de concluído |
| ---: | --- | --- |
| 1 | canonical builder | gera `canonical_samples` e relatório de exclusões sem overlap |
| 2 | schema package | JSON Schemas, enums e exemplos passam testes positivos/negativos |
| 3 | human-source joins | SKINCON/SkinCAP elegíveis ligados com conflitos explícitos |
| 4 | two-stage generator | Etapas A/B executam 100 casos com provenance completa |
| 5 | quality pipeline | filtros produzem estados e acceptance parcial reproduzíveis |
| 6 | task renderer | cria configs `diagnosis`, `morphology`, `caption`, `structured`, `open_response` |
| 7 | HF dataset preview | Dataset Viewer apresenta imagens e colunas. Card e licenças corretos |
| 8 | SFT smoke test | student memoriza subset pequeno e loss multimodal é válida |
| 9 | baseline e ablações faseadas | `E0_base`, `E1_label`, `E2_structured`, `E3_hard_kd` e `E4_soft_kd` comparados em Validation; componentes D/G declarados |
| 10 | ablações visuais e KD avançada | `E5_vision`, V2–V5, token-level KD ou on-policy apenas quando o baseline indicar necessidade |
| 11 | freeze final | componentes vencedores combinados em `E6_final`; modelo e dataset congelados. Internal Benchmark executado uma vez |

## 22. Conclusão metodológica

O `ISEPDistillDataset` deve ser construído como um conjunto de targets clínicos curtos, separados por capacidade e ligados por provenance. A mesma imagem terá várias respostas porque ensina várias tarefas. Não porque se pretende apresentar respostas alternativas indistinguíveis ao student.

A decisão mais defensável é usar multi-step reasoning no teacher como protocolo externo de geração e verificação, mantendo o target do student compacto. A Etapa A answer-blind protege o valor científico da experiência. A Etapa B volta a observar a imagem e liga hipóteses a evidência. O gold filtra depois. JSON e resposta aberta coexistem, mas em tarefas distintas e com consistência obrigatória.

SKINCON é imediatamente útil para supervisionar visão e morfologia. SkinCAP/SkinCoT são úteis apenas depois de resolver licença, overlap e gold-conditioning. SkinFlow reforça a prioridade de descrição visual, mas o seu `FDLinear` deve permanecer uma ablation. Para aproximar a visão do Qwen 4B à do Qwen 27B, o dataset é necessário mas não suficiente. Será preciso testar adaptação visual ou feature distillation no treino do modelo.

Esta arquitetura de dados permite que a tese responda a perguntas causais mais limpas: se o ganho vem de labels, conceitos humanos, supervisão estruturada, resposta aberta, política adaptativa, distilação de representações ou alteração da visão.

## 23. Estudos e documentação citados

### Datasets de origem, conceitos e avaliação externa

- Groh et al. (2021). [Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology With the Fitzpatrick 17k Dataset](https://doi.org/10.1109/CVPRW53098.2021.00201). *CVPR Workshops 2021*, 1820–1828.
- Abhishek, Jain e Hamarneh (2025). [Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets](https://doi.org/10.1038/s41597-025-04382-5). *Scientific Data*, 12, 196. A release corrigida associada encontra-se em [Fitzpatrick17k-C](https://doi.org/10.5281/zenodo.11101337).
- Pacheco et al. (2020). [PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones](https://doi.org/10.1016/j.dib.2020.106221). *Data in Brief*, 32, 106221.
- Ward et al. (2024). [Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements](https://doi.org/10.1001/jamanetworkopen.2024.46615). *JAMA Network Open*, 7(11), e2446615.
- Hospital Italiano de Buenos Aires (2023). [HIBA Skin Lesions](https://doi.org/10.34970/559884). ISIC Archive dataset record.
- Daneshjou et al. (2022). [SkinCon: A skin disease dataset densely annotated by domain experts for fine-grained debugging and analysis](https://papers.nips.cc/paper_files/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html). *Advances in Neural Information Processing Systems*, 35, Datasets and Benchmarks Track. DOI: [10.52202/068431-1320](https://doi.org/10.52202/068431-1320).
- Daneshjou et al. (2022). [Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set](https://doi.org/10.1126/sciadv.abq6147). *Science Advances*, 8(32), eabq6147.
- Shen et al. (2025). [SkinCaRe: A Multimodal Dermatology Dataset Annotated with Medical Caption and Chain-of-Thought Reasoning](https://arxiv.org/abs/2405.18004). arXiv:2405.18004v2.
- [SkinCaRe dataset card and release documentation](https://huggingface.co/datasets/yuhos16/SkinCaRe/blob/main/README.md). Hugging Face, consultado em 8 de agosto de 2026. Esta entrada documenta os ficheiros e termos apresentados na release; não é tratada como validação independente dos targets.

### Dermatologia e representação visual

- Zhou et al. (2024). [Pre-trained multimodal large language model enhances dermatological diagnosis using SkinGPT-4](https://doi.org/10.1038/s41467-024-50043-3). *Nature Communications*, 15, 5649.
- Yan et al. (2025). [MAKE: Multi-Aspect Knowledge-Enhanced Vision-Language Pretraining for Zero-shot Dermatological Assessment](https://arxiv.org/abs/2505.09372). MICCAI 2025.
- Yan et al. (2025). [A multimodal vision foundation model for clinical dermatology](https://doi.org/10.1038/s41591-025-03747-y). *Nature Medicine*, 31, 2691–2702.
- Ru et al. (2026). [DermoGPT: Open Weights and Open Data for Morphology-Grounded Dermatological Reasoning MLLMs](https://arxiv.org/abs/2601.01868). arXiv:2601.01868.
- Shen et al. (2025). [SkinGPT-R1: Adapter-Only Dual Distillation for Efficient Dermatology Reasoning](https://arxiv.org/abs/2511.15242). arXiv:2511.15242.
- Liu et al. (2026). [SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL](https://arxiv.org/abs/2601.09136). arXiv:2601.09136.

### Knowledge distillation

- Hsieh et al. (2023). [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://doi.org/10.18653/v1/2023.findings-acl.507). *Findings of the Association for Computational Linguistics: ACL 2023*, 8003–8017.
- Gu et al. (2024). [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543). ICLR 2024.
- Agarwal et al. (2024). [GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models](https://arxiv.org/abs/2306.13649). ICLR 2024.
- Xu et al. (2024). [LLAVADI: What Matters for Multimodal Large Language Models Distillation](https://arxiv.org/abs/2407.19409). arXiv:2407.19409.
- Cai et al. (2025). [LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.html). ICCV 2025, pp. 239–249.
- Feng et al. (2025). [Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Align-KD_Distilling_Cross-Modal_Alignment_Knowledge_for_Mobile_Vision-Language_Large_Model_CVPR_2025_paper.html). CVPR 2025, 4178–4188.
- [When Better Teachers Don't Make Better Students](https://arxiv.org/abs/2511.17886). arXiv:2511.17886.

### Referências internas do projeto

- [Investigação sobre MLLMs dermatológicos e estratégias de reasoning](../notes/11_dermatology_mllm_reasoning_strategy_research.md).
- [Comparação do dataset sintético e decisão de arquitetura](../notes/14_synthetic_dataset_comparison_and_architecture_decision.md).
- [Reasoning, visão e ablação de grounding sem imagem](../notes/15_reasoning_vision_and_no_image_grounding_ablation.md).
- [SkinFlow and the visual-encoder strategy](../notes/16_skinflow_architecture_and_visual_encoder_strategy.md).
- [SKINCON local dataset card](../../configs/datasets/skincon/README.md).
- [SkinCaRe local dataset card](../../configs/datasets/skincare/README.md).
- [ISEPDermaBench Hugging Face release](../dataset_pipeline/13_isep_dermabench_huggingface_release.md).

## 24. Provenance desta nota

Esta especificação sintetiza a investigação académica e os audits locais registados nas referências internas. A cobertura SKINCON foi reproduzida em 14 de agosto de 2026 e registada em `data/training/ISEPDistillDataset/metadata/skincon_coverage.json`: 3.866 anotações utilizáveis, zero labels upstream em falta, 271 overlaps internos excluídos e 3.595 linhas de morfologia publicadas. O builder versionado materializou ainda as 7.541 linhas de `diagnosis`; os counts, splits, revisões e SHA-256 dos shards ficaram congelados em `metadata/release.json`.

Em 15 de agosto de 2026, após o autor do dataset declarar possuir autorização escrita para criar derivados SkinCAP privados, foi materializada a release aditiva corrigida `isep_distill_dataset_v0.4.1`. A transformação `skincap_observation_prefix_v1` aceitou 3.250 de 3.318 candidatos técnicos: 2.767 em `sft_train` e 483 em `sft_dev`. A auditoria é conjunta sobre `diagnosis`, `morphology` e `caption`, com zero overlap de `leakage_group_id` entre train/dev. Os shards de treino não expõem a caption original, diagnóstico ou sufixo removido; conservam o target filtrado e hashes de proveniência. Como o texto humano original foi produzido com conhecimento do diagnóstico, `target_source=human_caption_gold_conditioned_filtered` e esta condição deve ser comparada como ablação, não apresentada como evidência answer-blind. `structured` e `open_response` continuam por gerar.

Uma auditoria pós-E2 em 16 de agosto de 2026 alterou a interpretação de
“corrigida”: 1.315 dos 3.250 targets de caption (40,46%) não tinham pontuação
terminal ou eram semanticamente incompletos, incluindo 234 terminados em
`It is` e 374 em `which is`. A release v0.4.1 permanece congelada como
proveniência do E2, mas não deve ser reutilizada diretamente no E3. A evidência
completa está registada em [E2 completo](../training_steps/04_e2_full_multitask_campaign_and_e1_comparison.md)
e na [comparação E1–E2](../benchmarks/14_e1_vs_e2_internal_benchmark.md).

## 25. Decisão pós-E2 para Stage A e Stage B no E3

**Stage A e Stage B permanecem no E3.** O resultado do E2 reforça, em vez de
eliminar, a necessidade da decomposição: o student aprendeu a verbalizar
achados, mas não aprendeu a integrá-los num diferencial completo. O teacher
deve, por isso, gerar e validar perceção e inferência como objetos separados
antes da renderização dos targets.

### Stage A: registo perceptual canónico

- recebe apenas a imagem e instruções de observação; não recebe gold;
- declara qualidade, avaliabilidade, vistas e limitações;
- produz observações atómicas com presença, ausência no âmbito observado,
  incerteza ou não avaliabilidade;
- gera apenas descrições completas, terminadas em fronteira de frase;
- falha o quality gate se terminar num fragmento, introduzir diagnóstico ou
  inventar história/contexto não visível.

Stage A alimenta principalmente `morphology` e `caption`, mas o seu JSON
canónico também fornece IDs de evidência para Stage B. Não se reutiliza o
prefixo truncado SkinCAP v0.4.1.

### Stage B: diferencial grounded

- volta a receber a imagem e a saída Stage A congelada;
- não recebe gold na condição principal answer-blind;
- declara qualquer correção a Stage A em vez de a alterar silenciosamente;
- ordena o diferencial e liga cada hipótese a IDs de observações de suporte e
  contradição;
- identifica discriminadores em falta, confiança, risco e uma ação permitida;
- falha o quality gate se apenas descrever a imagem, omitir o diferencial ou
  referir evidência inexistente.

O `gold_diagnosis` continua a entrar **apenas após as duas gerações**, para
aceitação/rejeição parcial. Assim, uma Stage A válida pode ser preservada mesmo
quando Stage B erra, sem transformar a label numa explicação retrospetiva.

### Como chega ao student

As duas stages são obrigatórias no **pipeline do teacher**, não necessariamente
duas chamadas sequenciais do student em produção. O E3 principal pode treinar
uma única resposta por tarefa, derivada do registo aceite:

| Target E3 | Origem canónica | O que ensina |
| --- | --- | --- |
| `DIAGNOSIS` | gold humano normalizado | classe/ranking diagnóstico |
| `MORPHOLOGY` | Stage A aceite | conceitos e limitações visíveis |
| `CAPTION` | rendering completo de Stage A | descrição clínica curta sem fragmentos |
| `GROUNDED_DIFFERENTIAL` | Stage A + Stage B aceites | descrição, Top-K, evidência, incerteza e discriminadores |

Cada rendering deve ter token/instrução de tarefa inequívoco e um validador de
completude específico. Uma variante student em duas chamadas pode ser estudada
depois como ablação, mas não é requisito para demonstrar a utilidade do
protocolo teacher em duas etapas.

A ação `REQUEST_CLINICAL_CONTEXT` pode permanecer como campo canónico de Stage
B. No entanto, ensinar e avaliar uma política interativa para pedir contexto
continua separado como `D4_adaptive_context`, depois de E3 e da seleção do
student estarem congelados. Esta separação evita confundir melhoria de
distilação com uma nova capacidade agentic.

O texto foi preparado com assistência de IA e deve ser revisto pelo autor da dissertação. Decisões clínicas, licenças e critérios de revisão especializada exigem validação humana antes da geração ou publicação do dataset.
