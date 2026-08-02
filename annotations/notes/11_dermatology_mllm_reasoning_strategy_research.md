# Investigação sobre MLLMs dermatológicos e estratégias de reasoning

Data: 2026-08-02

## 1. Objetivo e decisão executiva

Esta nota investiga duas questões:

1. que lógica de treino tem sido usada para desenvolver modelos multimodais de
   diagnóstico dermatológico;
2. que estratégias de reasoning ao nível da prompt aumentam, ou não, a
   accuracy clínica.

A conclusão para o projeto é:

- **não alterar a prompt open-ended congelada 1.1.0** com base apenas nesta
  literatura;
- usar morphology grounding, diferencial e evidência como **tarefas de treino
  separadas e verificáveis**;
- testar uma geração sintética com duas chamadas externas, `perceção ->
  diagnóstico`, em vez de pedir reasoning mais longo dentro de uma única
  prompt;
- tornar a segunda etapa adaptativa: só pedir contexto ou verificação adicional
  quando a imagem é insuficiente ou o diferencial permanece ambíguo;
- não usar chain-of-thought bruto como target principal do student.

Esta decisão combina evidência publicada com o resultado emparelhado do
próprio projeto: a prompt 1.3.0 aumentou os tokens médios de 1.562 para 3.223,
introduziu dois truncamentos e reduziu Top-1, visual findings e evidence
grounding. A evidência externa não substitui esse resultado específico do
Luna e do ISEPDermaBench.

## 2. Lógica encontrada em modelos dermatológicos

### 2.1 Decompor diagnóstico em capacidades clínicas

O padrão mais relevante é deixar de treinar apenas `imagem -> label` e passar
a representar várias capacidades:

```text
imagem
  -> qualidade e possibilidade de avaliação
  -> conceitos morfológicos observáveis
  -> descrição clínica curta
  -> diferencial ordenado
  -> relação entre evidência e hipóteses
  -> diagnóstico ou ação seguinte
```

O preprint DermoGPT apresenta DermoInstruct com 211.243 imagens, 772.675
trajetórias e cinco formatos de tarefa, cobrindo observação morfológica,
reasoning e diagnóstico. O modelo usa SFT e depois um objetivo de RL que tenta
manter consistência entre observações visuais e conclusão diagnóstica. O seu
benchmark separa Morphology, Diagnosis, Reasoning e Fairness. Esta separação é
muito próxima da direção já adotada pelo ISEPDermaBench.

Fonte: <https://arxiv.org/abs/2601.01868>.

O preprint Skin-R1 constrói trajetórias informadas por livros de dermatologia,
pela hierarquia das doenças e pelos seus diferenciais; usa essas trajetórias
em SFT e incorpora a hierarquia no reward de RL. As ablations reportadas pelos
autores atribuem importância à fundação de reasoning criada por SFT.

Fonte: <https://arxiv.org/abs/2511.14900>.

Estes dois trabalhos são recentes e devem ser tratados como evidência
preliminar até existir revisão independente e reprodução dos resultados.

### 2.2 Alinhar partes do texto com conceitos visuais

MAKE divide descrições clínicas em vários subtextos, alinha cada subcaption
com regiões ou aspetos visualmente relevantes e pondera-os pela sua importância
diagnóstica. Foi pré-treinado em 403.563 pares imagem-texto e avaliado em
classificação zero-shot, anotação de conceitos e retrieval. A principal lição
para um SLM não é reproduzir toda a arquitetura, mas evitar uma única caption
longa e ruidosa: morfologia, distribuição, superfície e diagnóstico podem ser
targets curtos e distintos.

Fonte: <https://arxiv.org/abs/2505.09372>.

O preprint VL-MedGuide é especialmente próximo da opção proposta neste
projeto: separa um módulo de perceção de conceitos dermatológicos de um módulo
de reasoning que recebe esses conceitos e a imagem. No Derm7pt, os autores
reportam 83,55% de balanced accuracy no diagnóstico e 76,10% na deteção de
conceitos. É uma demonstração promissora, mas limitada a dermoscopia, a um
dataset específico e a um preprint; não prova o mesmo efeito em fotografias
clínicas heterogéneas.

Fonte: <https://arxiv.org/abs/2508.06624>.

O DermaBench de 2026 também usa uma annotation hierarchy explícita para local,
morfologia, distribuição, superfície, cor, qualidade de imagem, descrições e
diagnóstico. Embora seja um benchmark e não um método de treino, confirma que
estas dimensões podem ser anotadas e avaliadas separadamente.

Fonte: <https://arxiv.org/abs/2601.14084>.

Os concepts não devem, contudo, tornar-se um bottleneck rígido pelo qual toda
a decisão tenha obrigatoriamente de passar. Uma análise recente do Derm7pt
encontrou perfis de concepts iguais associados a labels diferentes, impondo um
limite teórico a modelos baseados exclusivamente nesses conceitos. No projeto,
a Etapa B deve receber simultaneamente a imagem e os findings da Etapa A. Os
findings organizam e tornam o processo auditável, mas não substituem a
representação visual original.

Fonte: <https://arxiv.org/abs/2604.19323>.

### 2.3 Treino em fases e integração multimodal

SkinGPT-4 alinhou um encoder visual com Llama-2-13B-chat usando 52.929 imagens,
conceitos clínicos e notas médicas através de uma estratégia de treino em duas
fases. O estudo avaliou o sistema em 150 casos reais com dermatologistas. Isto
apoia treino faseado, mas não demonstra que uma prompt zero-shot com duas
etapas melhore qualquer modelo.

Fonte: <https://pubmed.ncbi.nlm.nih.gov/38969632/>.

PanDerm foi pré-treinado de forma self-supervised em mais de dois milhões de
imagens de 11 instituições e quatro modalidades. Foi avaliado em 28 tarefas e
frequentemente manteve desempenho competitivo usando apenas 10% dos labels.
Para esta tese, o resultado reforça a utilidade de um bom backbone visual e de
multitask learning; reasoning textual não consegue reparar sistematicamente
uma representação visual fraca.

Fonte: <https://pubmed.ncbi.nlm.nih.gov/40481209/>.

### 2.4 Retrieval seguido de re-ranking

DermPrompt usou GPT-4V como retriever e re-ranker. Com imagem e história breve,
o retriever incluiu a condição correta em 85% dos casos. Naive CoT foi útil no
retrieval, enquanto o diagnóstico final beneficiou de critérios
dermatológicos e comparação entre candidatos. O estudo também testou crítica
multi-agent, mas este componente acrescenta múltiplas chamadas e não é a opção
mais proporcional para o baseline de uma tese sobre SLMs.

Fonte: <https://arxiv.org/abs/2404.17749>.

Retrieval pode ser uma ablation futura:

```text
imagem + contexto validado
  -> candidatos ou exemplos semelhantes
  -> re-ranking pela evidência da imagem
```

O índice deve excluir todos os grupos de Validation, Internal Benchmark e
testes externos para impedir leakage. Por essa razão, retrieval não deve ser
introduzido antes de existir um corpus de referência e auditoria de grupos.

### 2.5 Contexto e interação guiados por incerteza

Uma fotografia isolada não resolve muitas doenças com morfologia sobreposta.
Num estudo com 1.118 médicos e 364 imagens, os autores destacam a diferença
entre uma simulação store-and-forward com uma imagem e a consulta, onde se pode
perguntar por sintomas, história pessoal e familiar e obter melhor iluminação
ou enquadramento. O estudo também mediu Top-1, Top-3 e diferenças por tom de
pele.

Fonte: <https://www.nature.com/articles/s41591-023-02728-3>.

O sistema multimodal AMIE usa estado e incerteza intermédios para controlar
fases de história, diagnóstico e gestão e para decidir quando pedir artefactos
multimodais. A implicação para o projeto é ensinar uma política explícita de
ação, não obrigar sempre a classificar de imediato.

Fonte: <https://pubmed.ncbi.nlm.nih.gov/42135531/>.

## 3. O que a literatura diz sobre reasoning em prompts

### 3.1 Evidência favorável a duas etapas estruturadas

Um estudo de 322 casos de *Diagnosis Please* comparou zero-shot CoT com uma
estratégia que primeiro organizava história e findings de imagem e depois fazia
o diagnóstico. A abordagem em duas etapas atingiu 60,6% de primary accuracy
contra 56,5% e Top-3 de 70,5% contra 66,5% no baseline. Contudo, eram casos de
radiologia com texto clínico rico e Claude 3.5 Sonnet, não fotografias
dermatológicas image-only.

Fonte: <https://pubmed.ncbi.nlm.nih.gov/39625594/>.

Um estudo em *npj Digital Medicine* separou diagnóstico inicial, verificação
predefinida e diagnóstico final. Em MedQA e casos NEJM, com GPT-4o e
DeepSeek-V3, obteve ganhos máximos de 4,0 pontos de accuracy relativamente a
CoT, além de menor incerteza e maior consistência. Os efeitos variaram por
modelo e dataset; uma redução da incerteza transformou por vezes respostas
incertas em respostas erradas, isto é, maior confiança sem maior correção.

Fonte: <https://www.nature.com/articles/s41746-025-02146-4>.

Estas experiências suportam **etapas com objetivos e outputs distintos**, não
uma instrução genérica para produzir uma cadeia mais longa.

### 3.2 CoT genérico não melhora sempre

Num estudo dermatológico de perguntas sobre cancro cutâneo não melanoma, a
accuracy média foi 4,92/5 com a prompt normal e 4,87/5 com “Let's think step by
step”; a diferença não foi significativa. A tarefa era educação textual para
doentes, não classificação visual, mas demonstra que a frase CoT por si só não
é uma intervenção fiável.

Fonte: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10755659/>.

Noutro estudo de reasoning diagnóstico, GPT-3.5 passou de 31% sem CoT para 46%
com CoT tradicional. Porém, prompts analíticas e de differential diagnosis
desceram para 40% e 38%. Os autores observaram melhores resultados quando a
prompt se concentrava numa estratégia, em vez de combinar várias. Em GPT-4,
imitar diferentes estilos clínicos não aumentou a accuracy face ao CoT
tradicional.

Fonte: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10808088/>.

Um preprint recente sobre self-reflection em MedQA, HeadQA e PubMedQA concluiu
que os ganhos foram dependentes do modelo e dataset: modestos num conjunto e
limitados ou negativos noutros; mais ciclos de reflexão não garantiram melhor
resultado.

Fonte: <https://arxiv.org/abs/2604.00261>.

### 3.3 Accuracy final não valida o reasoning

Um preprint de auditoria step-level de distillation médica encontrou um student
com melhores métricas de resposta e calibração, mas maior taxa de erro nos
passos do rationale. Embora recente e ainda preliminar, o resultado mostra
porque o projeto deve medir `clinical_rationale`, `evidence_grounding` e
`unsupported_claims` além de Top-K.

Fonte: <https://arxiv.org/abs/2605.28301>.

Outro preprint de 2026 alerta para answer-conditioned CoT: mostrar o gold label
ao teacher e pedir uma cadeia que chegue a esse label pode produzir
racionalização retrospetiva que passa num filtro de resposta correta. Por isso,
o fluxo principal proposto gera answer-blind e só compara com o label depois.

Fonte: <https://arxiv.org/abs/2607.14552>.

## 4. Pipeline sintético recomendado

### Etapa A — Perceção visual estruturada

Executar uma chamada dedicada com a imagem e sem label. O schema mínimo deve
conter:

```json
{
  "is_evaluable": true,
  "image_limitations": [],
  "anatomic_site_if_visible": null,
  "primary_morphology": [],
  "configuration": [],
  "distribution_within_frame": [],
  "color_relative_to_adjacent_skin": [],
  "surface_and_secondary_changes": [],
  "uncertain_findings": []
}
```

Os nomes finais devem usar vocabulário controlado sempre que SkinCon ou outra
fonte fornecer concepts compatíveis. Texto livre pode ser guardado num campo
separado, mas não deve substituir concepts normalizados.

### Etapa B — Diferencial, evidência e ação

Executar uma segunda chamada com a imagem, a saída congelada da Etapa A e
contexto clínico apenas quando real e validado:

```json
{
  "differential": [
    {
      "rank": 1,
      "disease": "...",
      "supporting_visible_evidence": ["..."],
      "contradicting_visible_evidence": []
    }
  ],
  "uncertainty": "low|moderate|high",
  "action": "CLASSIFY|ASK_CONTEXT|REQUEST_BETTER_IMAGE|ABSTAIN_OUT_OF_DOMAIN",
  "discriminating_question": null,
  "concise_clinical_rationale": "..."
}
```

Uma evidência só pode referir findings existentes na Etapa A. O modelo pode
usar conhecimento médico geral para comparar hipóteses, mas não pode converter
esse conhecimento em factos do doente.

### Quality gates

1. validar JSON, schema, vocabulário e unicidade dos três diagnósticos;
2. confirmar que cada evidência cita um finding da Etapa A;
3. rejeitar factos de história ou exame não fornecidos;
4. aceitar a rationale diagnóstica principal apenas quando Top-1 equivale ao
   ground truth;
5. aproveitar apenas a Etapa A quando os findings são válidos mas o diagnóstico
   está errado;
6. rejeitar loops, truncamentos e respostas com claims centrais não suportados;
7. medir taxas de aceitação por classe, source e grupo demográfico para não
   construir um dataset sintético composto apenas por casos fáceis;
8. nunca usar Validation ou conjuntos selados para gerar targets de treino.

Para `ASK_CONTEXT`, a pergunta deve separar as duas hipóteses mais próximas e
ter potencial para mudar o ranking. A resposta do utilizador só pode ser criada
a partir de metadata real; quando não existir, treina-se apenas a decisão de
perguntar, não uma história clínica inventada.

## 5. Mistura de tarefas proposta para SFT

Em vez de repetir o mesmo target longo para todas as imagens, criar exemplos
de instrução diferentes a partir do mesmo caso:

| Tarefa | Input | Target |
| --- | --- | --- |
| Image quality | Imagem | avaliável e limitações |
| Morphology grounding | Imagem | conceitos observáveis |
| Clinical description | Imagem | descrição curta |
| Closed-set diagnosis | Imagem + 21 classes | Top-K |
| Open-ended diagnosis | Imagem | Top-3 em texto clínico |
| Evidence grounding | Imagem + diferencial | evidência por hipótese |
| Interactive action | Imagem + estado | classificar, perguntar, pedir foto ou abster |

Esta mistura permite testar se o student aprende uma representação reutilizável
em vez de decorar um único formato JSON.

## 6. Ablations necessárias

Usar os mesmos grupos e seed em todas as condições:

| Variante | Supervisão adicional |
| --- | --- |
| A — baseline | imagem e label |
| B — concepts | A + morphology grounding |
| C — grounded rationale | B + diferencial, evidência e rationale curta |
| D — adaptive policy | C + decisão de pedir contexto/foto ou abster |
| E — raw CoT, apenas investigação | chain-of-thought bruto do teacher |

A variante E não deve ser a principal. Para distinguir treino de prompt
engineering, comparar também no teacher, sobre uma amostra estratificada de
Validation:

1. geração direta com a prompt congelada;
2. Etapa A e Etapa B em chamadas separadas;
3. opcionalmente, Etapa B com verificação apenas quando a incerteza for alta.

Medir Top-1, Top-3, MRR, findings, grounding, rationale, differential quality,
unsupported claims, validade, truncamentos, latência e tokens. A segunda opção
só avança para todo o Train se melhorar qualidade clínica sem enviesamento de
seleção ou custo desproporcional.

## 7. O que alterar e o que manter

### Alterar na fase de dados sintéticos

- implementar o protótipo de duas chamadas externas;
- gerar targets answer-blind e filtrar contra o ground truth posteriormente;
- incluir morphology concepts e image limitations como targets próprios;
- guardar reasoning bruto apenas para auditoria;
- ensinar rationale clínica curta e verificável;
- ensinar uma política adaptativa de contexto e qualidade de imagem;
- amostrar hard negatives usando a taxonomia e confusion sets já definidos.

### Manter por agora

- prompt open-ended 1.1.0 e judge 1.2.0 congelados;
- protocolo emparelhado do ISEPDermaBench;
- separação entre accuracy, grounding, rationale e unsupported claims;
- Top-1 e Top-3 como métricas distintas;
- reasoning do teacher como campo diferente do target do student.

### Não introduzir ainda

- uma prompt única mais longa com morphology, Bayesian reasoning, reflexão e
  crítica simultâneas;
- multi-agent reasoning como baseline;
- self-consistency para todos os casos;
- RAG antes de existir um índice leakage-safe;
- RL clínico antes de demonstrar benefício com SFT multitask;
- racionales criados depois de mostrar o gold label ao teacher.

## 8. Próximo decision gate

Depois de terminar a comparação dos teachers na Validation, selecionar um
teacher e executar um estudo emparelhado pequeno, por exemplo 100 casos
estratificados por classe, source e dificuldade:

```text
direto
versus
perceção estruturada -> diagnóstico fundamentado
```

O estudo é um protótipo de geração de dados e não altera o benchmark. Se a
segunda condição melhorar morphology grounding, clinical rationale e Top-K sem
loops ou custo excessivo, pode ser aplicada ao Train. Caso contrário, manter
multitask targets derivados de annotations existentes e não adicionar
reasoning sintético longo.
