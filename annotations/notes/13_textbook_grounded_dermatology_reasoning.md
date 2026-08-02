# Raciocínio dermatológico fundamentado em livros e literatura clínica

Data: 2026-08-02

## 1. Questão investigada

Depois de a prompt `open_ended_diagnosis` 1.1.0 ter superado prompts mais
prescritivas, foi feita uma pesquisa adicional em livros de dermatologia,
PubMed e arXiv. O objetivo não foi acrescentar mais terminologia médica à
prompt, mas identificar um processo clínico que pudesse ser implementado e
testado sem confundir texto mais longo com melhor raciocínio.

Foram analisadas três fontes de evidência:

1. referências clínicas sobre como descrever uma lesão e gerar um diferencial;
2. estudos cognitivos e educacionais sobre reconhecimento dermatológico;
3. sistemas multimodais recentes que separam perceção, conhecimento e
   diagnóstico.

## 2. O algoritmo clínico comum aos livros

Fitzpatrick descreve a morfologia macroscópica como o núcleo do diagnóstico
dermatológico. A descrição parte do tipo de lesão e inclui cor, marginação,
forma, arranjo e distribuição. A consistência também é importante no exame
presencial, mas não deve ser inferida a partir de uma fotografia.

O *Guidebook to Dermatologic Diagnosis* organiza o processo como uma “wheel of
diagnosis”:

```text
lesão primária e alterações secundárias
  -> agrupamento e configuração
  -> distribuição
  -> cabelo, unhas e mucosas, quando observáveis
  -> contexto e história do doente
  -> diferencial
```

O livro também reconhece que nem sempre é necessário percorrer toda a roda:
uma característica altamente discriminativa pode gerar imediatamente um
diferencial, usando os restantes elementos para o confirmar ou estreitar.

O capítulo *Clinical Methods* da NCBI converge para três componentes mínimos
antes de formular o diagnóstico:

1. morfologia da lesão primária;
2. arranjo ou configuração;
3. distribuição anatómica.

DermNet acrescenta número, tamanho, cor, simetria, forma, superfície e
alterações secundárias. Parte destas propriedades depende da qualidade e do
campo da fotografia; profundidade, consistência, temperatura, dor e
blanching são propriedades do exame presencial e não podem ser transformadas
em factos visuais.

### Implicação para o projeto

Uma descrição image-only deve distinguir três classes de informação:

| Classe | Exemplos | Regra |
| --- | --- | --- |
| Observável | placa, escala, bordo, cor relativa, configuração | pode fundamentar o diagnóstico |
| Não avaliável pela fotografia | consistência, calor, dor, profundidade | não inferir |
| Possivelmente ausente do enquadramento | distribuição corporal, outras lesões, unhas ou mucosa | declarar limitação, não ausência clínica |

## 3. Um dermatologista não usa apenas uma checklist

Os livros fornecem um método analítico, mas a investigação sobre expertise
mostra também reconhecimento não analítico de padrões.

Ko et al. descrevem o diagnóstico como uma sequência de perceção, categorização
e comparação dos dados visuais e clínicos com a base mental de diagnósticos.
Cada transição pode introduzir erro, o que justifica tornar a perceção
auditável sem assumir que uma explicação longa é correta.

No estudo prospetivo de Gachon et al., 135 dermatologistas avaliaram 4.036
lesões excisadas. O reconhecimento imediato de melanoma foi explicado
sobretudo pelo padrão global, pela comparação com os outros nevos do mesmo
doente e pela evolução recente. Critérios morfológicos isolados como ABCD não
explicaram, por si só, o comportamento dos especialistas.

Treino percetual em morfologia, configuração e distribuição melhorou o
reconhecimento destes conceitos em imagens novas. Isto favorece ensiná-los ao
student como capacidades explícitas, mas não prova que obrigar um teacher a
recitar uma checklist durante inferência melhore o diagnóstico.

### Relação com o A/B já executado

A candidata 1.3.0 tentou combinar impressão global e verificação analítica
numa única prompt. Nos mesmos 50 casos, reduziu Top-1 de 32% para 28%, piorou
visual findings e evidence grounding, introduziu dois truncamentos e mais do
que duplicou os tokens médios. A literatura clínica não invalida este
resultado; ajuda a explicá-lo. Uma checklist interna longa pode aumentar
deliberação sem melhorar a perceção e pode interferir com reconhecimento
global útil.

## 4. O que dizem os sistemas multimodais recentes

Os trabalhos mais relevantes não dependem apenas da frase “think step by
step”. Em vez disso, alteram os dados, a arquitetura ou o protocolo:

- **DermoGPT** cria trajetórias que cobrem morfologia, reasoning e diagnóstico
  e usa um objetivo que procura consistência entre observações visuais e a
  conclusão. O seu benchmark separa Morphology, Diagnosis, Reasoning e
  Fairness.
- **SkinGPT-R1** reporta ganhos ao combinar supervisão de reasoning
  dermatológico com distillation visual. É evidência preliminar de preprint e
  não demonstra que CoT zero-shot seja suficiente.
- **DermAgent** separa ferramentas de descrição morfológica, anotação de
  conceitos, diagnóstico e retrieval de imagens e guidelines, acrescentando
  uma auditoria determinística de conflitos. É um sistema agentic e muito mais
  pesado do que o SLM pretendido, mas a separação das funções é informativa.
- **DermPrompt** separa retrieval e re-ranking. A comparação final entre
  candidatos é suportada por critérios dermatológicos, em vez de depender
  apenas de uma resposta direta.
- Uma avaliação hospitalar recente de MLLMs encontrou uma queda acentuada
  entre benchmarks públicos e casos reais. Adicionar contexto clínico aumentou
  Top-3, mas os resultados ficaram sensíveis a contexto incompleto ou errado.

Estes estudos apoiam decomposição e grounding; não justificam promover uma
prompt mais longa sem um A/B local.

## 5. Limite fundamental do benchmark image-only

O diagnóstico dermatológico clínico combina imagem com sintomas, evolução,
distribuição corporal, história, palpação, dermoscopia, testes e, por vezes,
histopatologia. O sistema de Liu et al. para diagnóstico diferencial foi
treinado e avaliado com fotografias e dados clínicos, e o trabalho real-world
de 2026 observou ganhos de Top-3 quando foi adicionado contexto.

Consequentemente, quando duas doenças são visualmente plausíveis, a melhor
ação clínica pode ser pedir uma informação discriminativa e não produzir uma
explicação image-only mais confiante. Isto deve fazer parte da política
interativa do futuro student, mas não deve ser inventado dentro do benchmark
image-only.

## 6. Nova experiência proposta: dual-process externo

A investigação sugere testar uma mudança de **protocolo**, não substituir já a
prompt 1.1.0. A nova condição usa duas chamadas independentes e outputs
observáveis.

### Etapa A — representação visual do problema

O modelo recebe a imagem, não recebe o label e não produz diagnósticos. O
output é curto e estruturado:

```json
{
  "is_evaluable": true,
  "primary_morphology": ["plaque"],
  "surface_secondary_changes": ["adherent scale"],
  "configuration": ["well-demarcated"],
  "distribution_within_frame": ["localized"],
  "color_relative_to_adjacent_skin": ["erythematous"],
  "dominant_visual_pattern": "sharply demarcated scaly plaque",
  "uncertain_or_not_assessable": []
}
```

Regras da Etapa A:

- `dominant_visual_pattern` conserva a impressão holística num único campo;
- os restantes campos permitem auditar a descrição analítica;
- campos irrelevantes ou não observáveis ficam vazios;
- não são permitidos sintomas, história, palpação nem diagnóstico;
- a saída deve ser limitada a poucos conceitos para evitar preenchimento
  mecânico e hallucinations.

### Etapa B — diferencial fundamentado

O modelo recebe novamente a imagem e a representação congelada da Etapa A.
Produz exatamente o mesmo contrato final da prompt 1.1.0: findings concisos e
três diagnósticos explicitamente ordenados em prosa clínica.

A imagem continua disponível na Etapa B. A representação intermédia é uma
ajuda auditável, não um concept bottleneck rígido. Cada rationale deve ligar-se
a um finding da Etapa A ou a uma característica diretamente reavaliada na
imagem. Se a Etapa A estiver errada, a Etapa B pode corrigi-la explicitamente.

```text
imagem
  -> A: representação visual sem diagnóstico
  -> imagem + representação A
  -> B: Top-3 + rationale clínica curta
```

Esta é uma aproximação operacional a dois modos complementares:

- perceção holística e descrição do padrão;
- comparação analítica entre hipóteses.

Não pretende reproduzir literalmente a cognição privada de um dermatologista.

## 7. Desenho do teste

O primeiro teste deve usar uma amostra estratificada da Validation e os mesmos
task IDs em todas as condições:

| Condição | Descrição | Papel |
| --- | --- | --- |
| A | prompt direta congelada 1.1.0 | controlo |
| B | Etapa A sem diagnóstico + Etapa B | experiência principal |
| C | B + retrieval de fichas morfológicas das 21 doenças | ablation posterior |

Sequência recomendada:

1. smoke test em 20 casos para validar schema, latência e loops;
2. A/B emparelhado em pelo menos 100 casos, estratificados por classe, source,
   dificuldade e tom de pele quando disponível;
3. apenas se B mostrar um sinal favorável, executar toda a Validation;
4. não usar o Internal Benchmark para desenvolver este protocolo.

Métricas da resposta final:

- Top-1, Top-3 e MRR;
- visual findings, evidence grounding, clinical rationale e differential
  quality;
- unsupported-claim rate, invalid outputs e truncamentos;
- tokens, latência e custo por caso.

Métricas adicionais da Etapa A:

- validade do schema;
- precisão dos conceitos quando existem referências SkinCon compatíveis;
- taxa de findings não suportados;
- consistência entre findings da Etapa A e rationales da Etapa B;
- frequência com que a Etapa B corrige a Etapa A.

## 8. Retrieval baseado em livros: experiência separada

Pode ser criado um pequeno corpus versionado de fichas para as 21 classes,
derivado de referências autorizadas. Cada ficha conteria apenas conhecimento
geral:

```text
primary morphology
typical surface
common configuration
typical distribution
strong visual discriminators
features that usually require history or examination
```

Depois da Etapa A, seriam recuperadas fichas candidatas e a Etapa B faria o
re-ranking. Esta variante não deve substituir o baseline zero-shot: mede um
sistema knowledge-assisted diferente. O corpus não pode conter exemplos,
labels ou descrições provenientes de Validation e testes selados.

Deve também ser confirmado que a licença de cada fonte permite a forma de
derivação e distribuição pretendida. Não se devem copiar capítulos ou imagens
protegidas; as fichas devem ser sínteses próprias, curtas e citadas.

## 9. Decisão

Com base nesta pesquisa:

- **manter congeladas** a model prompt 1.1.0 e a judge prompt 1.2.0;
- **não criar outra prompt única, extensa e mais prescritiva**;
- testar a condição de duas chamadas externas como experiência separada na
  Validation;
- se o objetivo for apenas escolher o melhor teacher zero-shot, continuar a
  usar a prompt congelada;
- se o objetivo for gerar dados sintéticos para o student, a condição externa
  é especialmente relevante porque produz targets de perceção e diagnóstico
  que podem ser filtrados separadamente;
- guardar reasoning privado do provider apenas para auditoria; o target do
  student deve ser a representação visual e a rationale clínica curta e
  verificável.

## 10. Fontes principais

### Livros e referências clínicas

- Garg A, Levin NA, Bernhard JD. *Structure of Skin Lesions and Fundamentals
  of Clinical Diagnosis*. Fitzpatrick's Dermatology in General Medicine, 8e.
  <https://accessmedicine.mhmedical.com/content.aspx?bookId=392&sectionId=41138697>
- Wolff K, Johnson RA, Saavedra AP. *Fitzpatrick's Color Atlas and Synopsis of
  Clinical Dermatology, 8e: How to Use this Book*.
  <https://accessmedicine.mhmedical.com/content.aspx?bookid=2043&sectionid=154893619>
- Alikhan A, Hocker TLH. *Guidebook to Dermatologic Diagnosis: The Wheel of
  Diagnosis*.
  <https://accessmedicine.mhmedical.com/content.aspx?bookid=2960&sectionid=248574636>
- McKay M. *Clinical Methods: An Overview of the Skin and Appendages*.
  <https://www.ncbi.nlm.nih.gov/books/NBK206/>
- DermNet. *Terminology in dermatology*.
  <https://dermnetnz.org/topics/terminology>

### Cognição e educação dermatológica

- Ko CJ, et al. *Visual perception, cognition, and error in dermatologic
  diagnosis: Key cognitive principles*. JAAD, 2019.
  <https://pubmed.ncbi.nlm.nih.gov/30797839/>
- Gachon J, et al. *First prospective study of the recognition process of
  melanoma in dermatological practice*.
  <https://pubmed.ncbi.nlm.nih.gov/15837860/>
- Rimoin L, et al. *Training pattern recognition of skin lesion morphology,
  configuration, and distribution*. JAAD, 2015.
  <https://doi.org/10.1016/j.jaad.2014.11.016>
- Gropper CA. *An approach to clinical dermatologic diagnosis based on
  morphologic reaction patterns*.
  <https://pubmed.ncbi.nlm.nih.gov/12739317/>

### Sistemas multimodais e prompting

- Liu Y, et al. *A deep learning system for differential diagnosis of skin
  diseases*. Nature Medicine, 2020.
  <https://pubmed.ncbi.nlm.nih.gov/32424212/>
- Ru J, et al. *DermoGPT: Open Weights and Open Data for
  Morphology-Grounded Dermatological Reasoning MLLMs*.
  <https://arxiv.org/abs/2601.01868>
- Shen Y, et al. *SkinGPT-R1: Adapter-Only Dual Distillation for Efficient
  Dermatology Reasoning*. <https://arxiv.org/abs/2511.15242>
- Liu Y, et al. *DermAgent: A Self-Reflective Agentic System for
  Dermatological Image Analysis with Multi-Tool Reasoning and Traceable
  Decision-Making*. <https://arxiv.org/abs/2605.14403>
- Jiang R, et al. *Are Multimodal LLMs Ready for Clinical Dermatology? A
  Real-World Evaluation in Dermatology*.
  <https://arxiv.org/abs/2605.04098>
- Sadanandan B, Behzadan V. *When Chain-of-Thought Backfires: Evaluating
  Prompt Sensitivity in Medical Language Models*.
  <https://arxiv.org/abs/2603.25960>

Os trabalhos arXiv recentes são preprints e os seus resultados devem ser
tratados como preliminares até revisão independente e reprodução.
