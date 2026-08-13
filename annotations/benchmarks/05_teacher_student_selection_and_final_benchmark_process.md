# Processo de seleção do teacher, student e benchmark final

Data: 2026-08-02

## 1. O que já foi feito

Foi executado um screening fixo de 100 casos por task para validar a pipeline,
os prompts, os parsers, as métricas e o comportamento dos modelos com thinking
solicitado como ligado ou desligado. Qwen 3.7 Flash, MiMo V2.5 e MiniMax M3 não
apresentaram diferenças grandes ou consistentes entre os dois regimes. O
OpenRouter também nem sempre respeitou o pedido de desligar thinking.

Por isso, não é necessário continuar a repetir experiências ON/OFF. A partir
deste ponto, cada modelo deve usar uma única configuração congelada. O student
foi entretanto fixado como Qwen 3.5 4B. O objetivo da próxima fase passa a ser
medir o baseline oficial do student e escolher o teacher entre os restantes
modelos que avançam.

### 1.1 Tamanho dos modelos

Os tamanhos apresentados ao lado dos nomes são o número de parâmetros do
checkpoint completo publicado no Hugging Face, arredondado a duas casas. Nos
modelos MoE também é indicado o número de parâmetros ativados por token. Este
último aproxima melhor o custo computacional, enquanto o total continua a
determinar grande parte da memória necessária para alojar os pesos.

| Modelo | Tamanho usado neste documento | Fonte/observação |
| --- | ---: | --- |
| GPT-5.6 Luna | Não divulgado | Modelo proprietário de API, sem pesos ou contagem oficial no Hugging Face |
| Qwen 3.5 4B | 4,66B | [Metadata Safetensors](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen 3.5 9B | 9,65B | [Metadata Safetensors](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen 3.6 27B | 27,78B | [Metadata Safetensors](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Gemma 4 E4B | 8,00B total; 4,5B efetivos | [Model card oficial](https://huggingface.co/google/gemma-4-E4B-it); `E` significa *effective* e não MoE |
| Gemma 4 31B | 32,68B no checkpoint; 30,7B no modelo dense | [Model card oficial](https://huggingface.co/google/gemma-4-31B-it); o checkpoint multimodal também inclui o vision encoder |
| Qwen 3.7 Flash | Não divulgado | À data desta análise não existia um repositório oficial de pesos Qwen 3.7 Flash no Hugging Face |
| Qwen 3.8 Max | 2,4T total; ativos não divulgados | [Anúncio oficial da Alibaba](https://www.alibabagroup.com/en-US/document-2016703577908576256); o valor foi divulgado para o Qwen 3.8 Max Preview e a versão GA sucede-lhe diretamente |
| MiniMax M3 | 427,04B total; ~23B ativados | [Model card oficial](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| MiMo V2.5 | 310,78B total; 15B ativados | [Model card oficial](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) |

Os valores exatos de checkpoint foram obtidos do índice Safetensors. Nas
tabelas seguintes são usados valores curtos e legíveis. `Não divulgado` não
significa que o modelo seja pequeno; significa apenas que o provider não
publicou informação verificável sobre o número de parâmetros.

## 2. Screening fixo de 100 casos: configuração sem thinking

A tabela seguinte resume os mesmos 100 casos por task. Confusion Sets contém
100 pares e, por isso, 200 requests. As métricas de ranking são canónicas e
recuperam apenas transformações determinísticas de formato. Open-ended usa o
único judge multimodal definido pelo protocolo.

| Modelo | Top-K 1/3/6 | Confusion 1/2 | Evidence 1/3/6 | Open judge 1/3 | Rationale 0-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Luna (tamanho não divulgado) | 39 / 64 / 78% | 73 / 92.5% | 35 / 56 / 72% | 23 / 41% | **2.52** |
| Qwen 3.5 4B (4,66B) | 34 / 54 / 67% | 72 / 88% | 36 / 62 / 71% | 18.1 / 35.1% | 1.53 |
| Qwen 3.5 9B (9,65B) | 42 / 59 / 76% | 70.5 / 90.5% | 34 / 55 / 73% | 24.0 / 41.7% | 1.80 |
| Qwen 3.6 27B (27,78B) | **44** / 63 / 79% | 71 / 91% | 42 / 62 / 79% | 25.8 / 43.3% | 1.93 |
| Gemma 4 E4B (8,00B total; 4,5B efetivos) | 19 / 33 / 50% | 59 / 83.5% | 13 / 28 / 43% | 13 / 31% | 1.68 |
| Gemma 4 31B (32,68B checkpoint; 30,7B dense) | 41 / **71 / 86%** | **75 / 92%** | **45 / 65 / 83%** | 29.3 / 47.5% | 2.20 |
| Qwen 3.7 Flash (tamanho não divulgado) | 42 / 64 / 77% | **75 / 92%** | 41 / 57 / 74% | **35.8 / 53.7%** | 2.17 |
| MiniMax M3 (427,04B total; ~23B ativos) | 32 / 61 / 74% | 73 / **93.5%** | 34 / 52 / 69% | 22.3 / 41.5% | 1.87 |
| MiMo V2.5 (310,78B total; 15B ativos) | 37 / 51 / 62% | 68 / 88% | 33 / 57 / 71% | 24 / 48% | 1.91 |

Luna é a exceção da tabela: foi executado uma única vez com o seu
`reasoning_effort=high` congelado. Os restantes modelos locais foram executados
com thinking desligado. Qwen 3.7 devolveu zero reasoning nos 499 requests que
obtiveram resposta. MiniMax e MiMo receberam o mesmo pedido OFF, mas os
endpoints OpenRouter continuaram a devolver reasoning em parte dos casos;
essas linhas representam **OFF solicitado**, não uma garantia de zero thinking.

Os resultados Open-ended usam apenas judgments válidos. A comparação detalhada
deve acompanhar sempre coverage, recusas e casos inválidos, especialmente para
Qwen 3.7 e MiniMax.

## 3. Screening com thinking: Qwen 3.7, MiniMax e MiMo

Os três candidatos API foram repetidos nos mesmos IDs com thinking solicitado
como ligado. O limite total foi 14.336 tokens, com até 10.240 reservados para
reasoning. Nenhuma task concluída foi truncada.

| Modelo | Top-K 1/3/6 | Confusion 1/2 | Evidence 1/3/6 | Open judge 1/3 | Rationale 0-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.7 Flash (tamanho não divulgado), ON | **49 / 73 / 84%** | **76.5 / 92.5%** | **49 / 71 / 82%** | **37.9 / 53.7%** | **2.23** |
| MiniMax M3 (427,04B total; ~23B ativos), ON | 34 / 59 / 77% | 72 / 89.5% | 34 / 56 / 74% | 29.5 / 42.1% | 1.97 |
| MiMo V2.5 (310,78B total; 15B ativos), ON | 28 / 46 / 71% | 71.5 / 91.5% | 34 / 58 / 73% | 25 / 47% | 2.12 |

### 3.1 Diferença ON menos OFF solicitado

| Modelo | Top-K 1/3/6 | Confusion 1/2 | Evidence 1/3/6 | Open 1/3 | Rationale |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.7 Flash (tamanho não divulgado) | +7 / +9 / +7 pp | +1.5 / +0.5 pp | +8 / +14 / +8 pp | +2.1 / 0 pp | +0.06 |
| MiniMax M3 (427,04B total; ~23B ativos) | +2 / -2 / +3 pp | -1 / -4 pp | 0 / +4 / +5 pp | +7.2 / +0.6 pp | +0.10 |
| MiMo V2.5 (310,78B total; 15B ativos) | -9 / -5 / +9 pp | +3.5 / +3.5 pp | +1 / +1 / +2 pp | +1 / -1 pp | +0.21 |

Thinking não apresentou uma melhoria global e consistente. Qwen 3.7 foi o
modelo com o benefício clínico mais claro, sobretudo em Evidence e Top-K. No
MiniMax e MiMo, os ganhos foram pequenos ou contraditórios entre tasks e
ranks. A conclusão prática é que a capacidade base do modelo parece explicar
mais do resultado do que o simples interruptor de thinking.

Thinking também alterou a obediência ao formato. Por exemplo, no MiniMax o raw
JSON de Confusion caiu de 67% para 29% e o de Evidence de 72% para 22%, embora
quase todos os objetos continuassem recuperáveis pelo parser determinístico.
No Qwen 3.7, thinking melhorou a accuracy clínica, mas originou muitas JSON
fences. Portanto, a decisão deve considerar capacidade canónica e qualidade
end-to-end separadamente.

Esta não é uma ablação causal perfeita. MiniMax e MiMo produziram reasoning
mesmo em parte dos requests OFF, e o OpenRouter pode escolher implementações
de provider diferentes. Não se deve afirmar na dissertação que thinking não
funciona; apenas que, neste protocolo e nestes modelos, não justificou repetir
mais um ciclo completo ON/OFF.

## 4. Feedback por modelo

### 4.1 Candidatos a teacher que avançam

| Modelo | Pontos fortes | Limitações e risco | Decisão atual |
| --- | --- | --- | --- |
| Qwen 3.6 27B (27,78B) | Melhor Top-K Top-1 do screening sem thinking e Evidence competitivo | Modelo local denso, com maior custo de hosting; thinking apresentou latência e loops extensos | **Avança como candidato local a teacher**, com thinking desligado |
| Qwen 3.7 Flash (tamanho não divulgado) | Melhor combinação API de diagnosis, morphology e Open-ended; obteve o maior ganho com thinking | Unsupported claims elevados, semantic compliance/output exigem parsing e houve erros isolados de provider | Avança como candidato principal a teacher |
| Qwen 3.8 Max (2,4T total; ativos não divulgados) | Flagship multimodal mais recente da Alibaba, com contexto de 1M e structured output no endpoint oficial | Não participou no screening inicial; reasoning é obrigatório e o custo é superior ao Qwen 3.7 Flash | **Novo candidato API a teacher**, sujeito à mesma Validation completa |
| MiniMax M3 (427,04B total; ~23B ativos) | Confusion competitivo e melhoria Open-ended Top-1 com thinking | Ganhos inconsistentes, pior JSON com thinking e rationale/grounding moderados | Avança para confirmar estabilidade na Validation completa |
| MiMo V2.5 (310,78B total; 15B ativos) | Evidence/morphology razoáveis e melhor rationale com thinking | Diagnosis ranking irregular, reasoning muito extenso, OFF não respeitado e execução lenta | Avança como candidato complementar orientado a rationale |

### 4.2 Modelos grandes mantidos como reservas

| Modelo | Feedback |
| --- | --- |
| GPT-5.6 Luna/Azure (tamanho não divulgado) | Continua documentado como referência clínica forte em visual findings, evidence grounding e rationale. Não integra a próxima Validation completa definida nesta decisão. |
| Gemma 4 31B (32,68B checkpoint; 30,7B dense) | Foi o modelo local mais equilibrado: liderou Top-K Top-3/6 e Evidence. Contudo, colocou sistematicamente JSON em fences e tem maior custo de hosting. Deve ser preservado como reserva/local baseline, mas não precisa de integrar a primeira passagem API completa se o objetivo for controlar custo. |

### 4.3 Candidatos a student

| Student | Feedback | Papel recomendado |
| --- | --- | --- |
| Qwen 3.5 4B (4,66B) | Tamanho alinhado com a hipótese SLM, baseline razoável e Evidence Top-1 de 36%; rationale, semantic compliance e unsupported claims ainda são fracos | **Student oficial selecionado**; avança para a Validation completa e será o modelo fine-tuned |
| Qwen 3.5 9B (9,65B) | Melhor Top-K e morphology do que o 4B em várias métricas, mas exige mais memória e enfraquece a afirmação de modelo realmente pequeno | Não avança nesta fase; permanece apenas como resultado de screening |
| Gemma 4 E4B (8,00B total; 4,5B efetivos) | Arquitetura/família alternativa e dimensão pequena, mas foi claramente o modelo mais fraco em diagnosis no screening | Não avança nesta fase; permanece apenas como controlo histórico cross-family |

Qwen 3.5 4B deixa de ser apenas um candidato: é o **student oficial** da tese.
Esta escolha fixa antecipadamente o modelo cuja diferença antes/depois do
fine-tuning será a comparação central. Qwen 3.5 9B e Gemma E4B não participam
na próxima fase.

## 5. Decisão resultante do screening

- não será repetido outro A/B geral de thinking;
- Qwen 3.7 avança com thinking ON, condição em que apresentou o ganho clínico
  mais consistente;
- Qwen 3.8 Max entra posteriormente como novo candidato e usa thinking
  obrigatório com `reasoning_effort=high`; não é possível incluí-lo numa
  comparação ON/OFF equivalente;
- MiniMax avança com thinking OFF solicitado, devido aos resultados mistos e à
  degradação do JSON quando ON;
- MiMo avança inicialmente com thinking OFF solicitado para reduzir custo e
  latência; o provider pode continuar a produzir reasoning residual;
- a configuração histórica de Luna mantém `reasoning_effort=high`, mas o
  modelo não avança para esta Validation completa;
- Qwen 3.5 4B é congelado como **student oficial**;
- Qwen 3.6 27B, Qwen 3.7 Flash, Qwen 3.8 Max, MiniMax M3 e MiMo V2.5 avançam como
  candidatos a teacher;
- neste contexto, «todos os modelos do OpenRouter» significa os quatro modelos
  cujo `backend.default_profile` é `openrouter`: Qwen 3.7 Flash, Qwen 3.8 Max,
  MiniMax M3 e MiMo V2.5;
- o teacher final só é escolhido depois da Validation completa;
- Luna/Azure, Gemma 31B, Qwen 9B e Gemma E4B não avançam nesta passagem, mas
  os resultados de screening permanecem documentados.

## 6. Próxima etapa: Validation completa

A próxima execução é a passagem completa pela Validation, ainda destinada a
desenvolvimento e seleção de modelos. Não é a benchmark interna final da tese.

| Task | Casos/requests por modelo |
| --- | ---: |
| Visual Top-K | 1.000 |
| Visual Confusion Sets | 834 requests, correspondentes a 417 pares |
| Evidence-grounded diagnosis | 137 |
| Open-ended diagnosis | 100 |
| **Total por modelo** | **2.071 requests** |

Os seis modelos que serão executados são:

| Nome no estudo | Papel nesta fase | Configuração |
| --- | --- | --- |
| Qwen 3.5 4B (4,66B) | Student oficial; baseline antes do treino | `configs/models/qwen_small_4b.yaml` |
| Qwen 3.6 27B (27,78B) | Candidato local a teacher | `configs/models/qwen_3_6_27b.yaml` |
| Qwen 3.7 Flash (tamanho não divulgado) através de OpenRouter | Candidato API a teacher | `configs/models/qwen_3_7_flash_openrouter.yaml` |
| Qwen 3.8 Max (2,4T total; ativos não divulgados) através de OpenRouter | Candidato API a teacher; thinking obrigatório | `configs/models/qwen_3_8_max_openrouter.yaml` |
| MiniMax M3 (427,04B total; ~23B ativos) através de OpenRouter | Candidato API a teacher | `configs/models/minimax_m3_openrouter.yaml` |
| MiMo V2.5 (310,78B total; 15B ativos) através de OpenRouter | Candidato API a teacher | `configs/models/mimo_v2_5_openrouter.yaml` |

Esta matriz compara o student oficial com um candidato local maior e quatro
candidatos API do OpenRouter. Representa **seis modelos e 12.426 requests**
para a Validation completa. O student não concorre ao papel de teacher: a sua
execução serve para fixar o baseline pré-treino nos mesmos casos.

## 7. Student oficial

O student oficial é `qwen_small_4b` (Qwen 3.5 4B, 4,66B parâmetros). A escolha
fica congelada antes da Validation completa e já não depende de superar Qwen
9B ou Gemma E4B nessa passagem. A Validation do Qwen 4B mede o ponto de partida
pré-treino, permite analisar os seus erros e fornece a comparação emparelhada
com os candidatos a teacher nos mesmos casos.

Depois da criação do dataset sintético, esta mesma arquitetura será submetida
a fine-tuning. A hipótese principal será avaliada comparando Qwen 3.5 4B base
com Qwen 3.5 4B fine-tuned, e ambos com o teacher selecionado.

## 8. Como escolher o teacher

O teacher não deve ser escolhido apenas por Top-1. A decisão deve considerar:

1. Top-1, Top-3 e Top-6 nas tarefas de diagnóstico;
2. desempenho nos confusion sets de baixa e alta confundibilidade;
3. morphology e description F1;
4. evidence grounding e semantic compliance;
5. clinical rationale e visual findings no Open-ended;
6. unsupported-claim rate;
7. JSON estrito, output recuperável, truncamentos, erros e recusas;
8. custo e tempo necessários para anotar o dataset de treino.

Deve ser selecionado um teacher que seja clinicamente forte e produza dados
sintéticos utilizáveis. Um modelo com Top-1 ligeiramente superior, mas com
muitas afirmações sem suporte ou outputs instáveis, pode ser um teacher pior.

## 9. Congelamento da decisão

Depois da Validation completa:

- escolhe-se e regista-se o teacher; o student Qwen 3.5 4B já está congelado;
- congelam-se prompts, parsers, model revisions e métricas;
- deixam de ser feitas alterações com base na benchmark interna;
- a Validation pode continuar a apoiar checkpoints e hiperparâmetros, mas não
  deve ser movida para o Train nem usada como dados de fine-tuning.

Este congelamento evita escolher o sistema final depois de observar o conjunto
de teste selado.

## 10. Criação dos dados sintéticos e treino

O teacher selecionado analisa as imagens do Train e produz os exemplos
sintéticos necessários. O registo de treino deve privilegiar a resposta final
e uma clinical rationale visível, verificável e ligada à imagem. Reasoning
privado ou cadeias internas extensas não devem ser tratados automaticamente
como ground truth de treino.

O dataset aumentado é dividido por leakage group em:

- SFT Train, que participa nos gradientes;
- SFT Dev, que acompanha loss, overfitting e seleção de checkpoint.

O student é então treinado, o melhor checkpoint é escolhido sem consultar o
Internal Benchmark e é produzida uma versão final congelada.

## 11. Benchmark interna final

Só depois do treino é executada a benchmark interna completa e selada. A
comparação principal da tese será:

1. student base antes do fine-tuning;
2. student fine-tuned;
3. teacher selecionado.

Também é aceitável executar Luna, Qwen 3.7 Flash, Qwen 3.8 Max, MiniMax M3 e
MiMo V2.5 uma única vez na benchmark interna, depois de todas as decisões
estarem congeladas.
Esses resultados funcionam como baselines adicionais e permitem contextualizar
o student. Não podem ser usados retroativamente para mudar o teacher, o
student, os prompts ou o treino.

O resultado central continuará a ser a diferença emparelhada entre o mesmo
student antes e depois do fine-tuning, usando exatamente os mesmos casos.

### 11.1 Benchmarks textuais complementares após fine-tuning e distilação

Depois de congelar o checkpoint vencedor de uma fase, pode ser executada uma
avaliação complementar em MedQA, no subconjunto expert-labeled do PubMedQA
(`PQA-L`) e nos cinco domínios médicos de MMLU (`anatomy`,
`clinical_knowledge`, `college_medicine`, `medical_genetics` e
`professional_medicine`). O objetivo é medir retenção de conhecimento médico,
catastrophic forgetting e transferência de conhecimento entre um teacher
open-weight e o student; estes resultados não medem visão dermatológica.

O PubMedQA é uma tarefa determinística `yes/no/maybe` baseada em abstracts
biomédicos ([Jin et al., 2019](https://arxiv.org/abs/1909.06146)). A release,
o split, o prompt e o parser de todas estas tarefas devem ser congelados antes
da primeira execução confirmatória.

Devem ser comparados o Qwen 3.5 4B base, os vencedores de fase relevantes, o
teacher open-weight e o student distilled final com a mesma release, prompt,
decoding e parser. MedQA, PubMedQA e MMLU não podem selecionar checkpoints ou
alterar hiperparâmetros. O seu uso para tuning invalidaria a interpretação
como avaliação externa. Os resultados permanecem separados do ISEPDermaBench
e do DermoBench e incluem uma ressalva de possível contaminação por pretraining,
dado serem benchmarks públicos amplamente usados.

Assim, a ordem é: `SFT Dev -> checkpoint selection -> checkpoint congelado ->
MedQA/PubMedQA/MMLU`. Os benchmarks públicos quantificam retenção e
transferência; não reabrem a seleção do checkpoint.

## 12. Flow final

```text
Screening fixo de 100 casos
        ↓
Congelar uma configuração por modelo
        ↓
Validation completa: Qwen 4B base + cinco teachers candidatos
        ↓
Escolher e congelar o teacher
        ↓
Teacher anota o Train e cria dados sintéticos
        ↓
Fine-tuning com SFT Train + seleção com SFT Dev/Validation
        ↓
Congelar o checkpoint final
        ↓
Internal Benchmark completo, uma única vez
        ↓
Comparar student base vs fine-tuned vs teacher e baselines
```

Esta sequência mantém fixa a arquitetura do student, usa cinco modelos fortes
na seleção do teacher e preserva a benchmark interna como avaliação honesta da
hipótese da tese.
