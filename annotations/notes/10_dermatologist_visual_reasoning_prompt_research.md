# Investigação para uma prompt de raciocínio visual dermatológico

Data: 2026-08-02

## Objetivo

Criar uma prompt candidata que aproxime o comportamento visual observável de
um dermatologista, sem afirmar que o modelo possui experiência clínica e sem
pedir ou expor chain-of-thought privado. A prompt congelada 1.1.0 permanece
inalterada até existir evidência emparelhada que justifique uma nova release.

## Evidência encontrada

1. O diagnóstico dermatológico depende de ver e organizar corretamente o
   achado, nomear a morfologia e comparar os dados visuais com padrões mentais.
   A literatura também alerta para gestalt incorreto, cegueira desatencional e
   transições imperfeitas entre perceção e diagnóstico.
2. Um estudo com dermatologistas certificados, dermatologistas em formação e
   não médicos encontrou padrões de movimento ocular e narrativas dependentes
   da experiência durante a inspeção de fotografias dermatológicas. Isto
   suporta uma pesquisa visual organizada, mas não uma checklist rígida.
3. Treino percetual de morfologia primária, configuração e distribuição
   anatómica melhorou significativamente reconhecimento e discriminação em
   imagens novas. Estas são, portanto, dimensões visuais centrais.
4. Num estudo prospetivo de reconhecimento de melanoma, dermatologistas
   experientes dependeram fortemente do padrão global e da comparação, e não
   apenas da aplicação isolada de critérios ABCD. A prompt deve combinar
   impressão global com verificação analítica.
5. A accuracy visual varia com o tom de pele. Estudos com médicos e sistemas de
   apoio mostram menor accuracy em imagens de pele escura, e a avaliação visual
   de eritema é menos fiável em pigmentação mais elevada. A prompt não deve
   exigir vermelho vivo; deve descrever cor e contraste relativamente à pele
   adjacente.
6. Diferenciais Top-3 em texto livre são clinicamente plausíveis e já foram
   usados em estudos com centenas de dermatologistas. A imagem isolada mantém
   limitações reais: até dermatologistas apresentam accuracy modesta em casos
   open-ended sem história clínica.

## Tradução para a prompt

A candidata 1.3.0 usa dois passes internos complementares:

- reconhecimento global do padrão dominante e localização da anormalidade
  mais informativa;
- verificação através da lesão primária mais representativa, morfologia,
  configuração, distribuição, bordos, simetria, superfície, alterações
  secundárias e contraste de cor.

Antes do output, o modelo compara silenciosamente a hipótese principal com o
mimic visual mais forte e procura evidência discriminativa realmente visível.
O output continua a ser prosa clínica concisa com Top-3 explícito. A prompt não
pede uma narrativa passo a passo, não mostra chain-of-thought e proíbe dados de
doente não observáveis.

## Salvaguarda metodológica

Esta prompt é uma operacionalização inspirada em estudos de expertise; não é
uma reprodução da cognição de um dermatologista. Instruções não substituem
treino clínico, dados multimodais, exame presencial ou validação médica.

A candidata está em
`src/benchmark/resources/open_ended_diagnosis/model_prompt_v1_3_0_candidate.yaml`.
Não faz parte da release congelada. Antes de promoção, deve ser comparada com
1.1.0 nos mesmos task IDs, modelo, decoding e judge.

## A/B emparelhado de 50 casos

A candidata foi posteriormente executada nos mesmos 50 task IDs usados para
congelar a prompt 1.1.0. Modelo, seed, decoding, reasoning effort e judge foram
mantidos. Não foi usado fallback judge.

| Métrica | Congelada 1.1.0 | Dermatologist-vision 1.3.0 | Candidata − congelada |
| --- | ---: | ---: | ---: |
| Respostas válidas | 47/50 | 45/50 | -2 |
| Safety refusals | 3 | 3 | 0 |
| Truncamentos | 0 | 2 | +2 |
| Top-1 accuracy | 32% | 28% | -4 pp |
| Top-3 accuracy | 44% | 46% | +2 pp |
| Mean reciprocal rank | 0.370 | 0.353 | -0.017 |
| Diagnosis correctness, 0–4 | 1.58 | 1.58 | 0.00 |
| Visual findings, 0–4 | 3.02 | 2.78 | -0.24 |
| Evidence grounding, 0–4 | 3.42 | 3.20 | -0.22 |
| Clinical rationale, 0–4 | 2.68 | 2.56 | -0.12 |
| Differential quality, 0–4 | 2.66 | 2.70 | +0.04 |
| Unsupported-claim rate | 16% | 18% | +2 pp |

Nos pares, cinco casos foram Top-1 corretos apenas com 1.1.0, três apenas com
a candidata e onze com ambas. Para Top-3, dois casos favoreceram apenas 1.1.0
e três apenas a candidata.

A candidata também mais do que duplicou o custo médio por resposta válida:
3.223 tokens totais contra 1.562. O reasoning médio subiu de 862 para 2.178
tokens. Nos dois casos truncados, o reasoning consumiu o limite completo de
8.192 tokens, sugerindo que a instrução de inspeção em dois passes induziu
deliberação excessiva ou um reasoning loop no Luna.

## Decisão após o A/B

Rejeitar a candidata 1.3.0 para o protocolo atual. A melhoria de 2 pontos em
Top-3 e 0,04 em differential quality não compensa menor Top-1, findings,
grounding e rationale, mais claims não suportados, dois truncamentos e mais do
dobro dos tokens. A release 1.5.0 permanece congelada com prompt 1.1.0.

O resultado não demonstra que princípios de expertise dermatológica sejam
inúteis. Mostra que pedir explicitamente uma inspeção interna em dois passes a
um modelo já configurado com reasoning high pode aumentar computação sem
melhorar a perceção. Estes princípios poderão ser mais úteis como estrutura de
dados de treino, annotations de morfologia ou critério do judge do que como
instruções extensas em inferência zero-shot.

## Fontes

- Ko CJ, et al. *Visual perception, cognition, and error in dermatologic
  diagnosis: Key cognitive principles*. JAAD, 2019.
  https://pubmed.ncbi.nlm.nih.gov/30797839/
- Tourassi GD, et al. *Modeling eye movement patterns to characterize
  perceptual skill in image-based diagnostic reasoning processes*.
  https://pubmed.ncbi.nlm.nih.gov/36046501/
- Rimoin L, et al. *Training pattern recognition of skin lesion morphology,
  configuration, and distribution*. JAAD, 2015.
  https://doi.org/10.1016/j.jaad.2014.11.016
- Gachon J, et al. *First prospective study of the recognition process of
  melanoma in dermatological practice*.
  https://pubmed.ncbi.nlm.nih.gov/15837860/
- Krefting F, et al. *Comparison of visual diagnostic accuracy of
  dermatologists practicing in Germany in patients with light skin and skin
  of color*. Scientific Reports, 2024.
  https://pubmed.ncbi.nlm.nih.gov/38627499/
- Jain A, et al. *Deep learning-aided decision support for diagnosis of skin
  disease across skin tones*. Nature Medicine, 2024.
  https://www.nature.com/articles/s41591-023-02728-3
