# Estratégia para um modelo multimodal dermatológico interativo

## 1. Contexto e prioridade atual

O objetivo central da tese é estudar se um modelo multimodal pequeno,
especializado através de fine-tuning, consegue atingir desempenho próximo de
um modelo multimodal grande num domínio dermatológico controlado, utilizando
menos memória, menor latência e menor custo de inferência.

Antes de desenvolver a interação com o utilizador ou o routing entre modelos,
a prioridade deve ser:

1. terminar e validar os benchmarks;
2. executar os modelos candidatos nas mesmas partições congeladas;
3. escolher o modelo teacher;
4. escolher o modelo pequeno que será alvo de fine-tuning;
5. estabelecer os resultados baseline antes de criar dados sintéticos.

O teacher deve ser escolhido usando apenas a partição de validação. O teste
interno e os conjuntos externos devem permanecer selados até todas as decisões
de desenvolvimento estarem congeladas.

## 2. Questão de investigação principal

Uma formulação possível é:

> Um pequeno modelo multimodal especializado consegue atingir desempenho não
> inferior ao de um modelo multimodal grande numa taxonomia dermatológica
> fechada?

Não deve ser utilizada apenas a expressão vaga "accuracy semelhante". Deve ser
definida previamente uma margem de não inferioridade, por exemplo cinco pontos
percentuais, acompanhada por intervalos de confiança emparelhados.

As comparações mínimas são:

1. modelo multimodal grande;
2. modelo multimodal pequeno sem fine-tuning;
3. o mesmo modelo pequeno depois do fine-tuning.

Além do desempenho clínico, devem ser registados:

- número de parâmetros;
- memória/VRAM;
- latência por caso;
- custo por caso;
- tamanho do output;
- taxa de outputs estruturalmente válidos.

## 3. Benchmarks definidos

### 3.1 Visual Top-K

O benchmark Visual Top-K mede a capacidade de ordenar seis doenças da
taxonomia fechada:

- Top-1;
- Top-3;
- Top-6;
- Mean Reciprocal Rank;
- macro-F1;
- validade JSON e cumprimento do schema.

Este benchmark deve permanecer simples, pois representa a medição mais direta
da capacidade de diagnóstico visual.

### 3.2 Evidence-Grounded Diagnosis

O segundo benchmark é mais exigente e avalia, na mesma execução:

1. reconhecimento estruturado de morfologia;
2. descrição clínica da imagem;
3. diagnóstico diferencial;
4. ligação explícita entre diagnósticos e achados;
5. confiança.

Um output esperado tem a forma:

```json
{
  "findings": [
    {
      "finding_id": "F1",
      "concept_id": "plaque",
      "confidence": 0.91
    },
    {
      "finding_id": "F2",
      "concept_id": "scale",
      "confidence": 0.86
    }
  ],
  "clinical_description": "Placa eritematosa bem delimitada com descamação superficial.",
  "differential": [
    {
      "rank": 1,
      "disease_id": "D003",
      "confidence": 0.72,
      "supporting_finding_ids": ["F1", "F2"]
    }
  ],
  "case_confidence": "moderate"
}
```

Os três componentes devem receber métricas separadas. Não se recomenda um
único score agregado que esconda se o erro ocorreu na perceção, na descrição
ou no raciocínio.

As coortes atualmente declaradas são:

- 636 casos DDI–SKINCON para morfologia;
- 635 casos com caption SkinCAP para descrição;
- 294 casos cobertos pela taxonomia ativa para diagnóstico e grounding.

### 3.3 Avaliação determinística

O modelo não gera um schema: o schema é definido pelo projeto e o modelo gera
JSON que deve obedecer-lhe.

O avaliador futuro deverá executar:

1. parsing JSON;
2. validação JSON Schema;
3. validação semântica;
4. junção com as referências;
5. cálculo determinístico das métricas.

As previsões estruturadas podem ser comparadas diretamente:

```text
Conceitos previstos: plaque, scale
Conceitos SKINCON: plaque, scale, erythema
```

Daqui resultam precision, recall, F1 e taxa de achados não suportados.

A descrição livre requer um extrator terminológico fixo, com:

- nomes dos conceitos SKINCON;
- sinónimos aprovados;
- variações morfológicas;
- expressões compostas;
- deteção de negação;
- limites de palavra;
- versão congelada do normalizador.

Uma simples condição como `if "scale" in text` não é suficiente. Por exemplo,
`scaly` deve mapear para `scale`, enquanto `no visible scale` não deve ser
considerado uma observação positiva.

BERTScore contra SkinCAP pode ser reportado como métrica secundária. Os
captions SkinCAP incluem ocasionalmente diagnósticos, exames e recomendações,
pelo que semelhança textual não deve ser considerada prova de correção
clínica.

### 3.4 Evidência de que fotografia e contexto clínico são complementares

Existe evidência empírica de que combinar a fotografia com dados clínicos pode
melhorar a classificação dermatológica. A formulação deve, contudo, ser
cuidadosa: os estudos sustentam a hipótese de complementaridade entre
modalidades, mas não provam que qualquer metadata ou qualquer método de fusão
melhore sempre o resultado. A qualidade dos dados, a arquitetura de fusão, a
doença e a distribuição de avaliação continuam a ser determinantes.

#### Estudos diretamente relevantes

| Estudo | Dados e tarefa | Comparação relevante | Resultado principal |
| --- | --- | --- | --- |
| [Ou et al., 2022 — MMF-Net](https://doi.org/10.3389/fsurg.2022.1029991) | PAD-UFES-20, seis classes, fotografias de smartphone e até 21 variáveis clínicas | Imagem sem metadata vs. imagem com metadata e fusão multimodal | Accuracy de 0,616 para 0,768; balanced accuracy de 0,651 para 0,775; AUC de 0,901 para 0,947. As três diferenças foram significativas com `p < 0,001`. |
| [Cai et al., 2022/2023 — Multimodal Transformer](https://doi.org/10.1007/s00371-022-02492-4) | Classificação multiclasse de doenças cutâneas num dataset clínico privado | Imagem sem metadata vs. imagem e metadata com mutual attention | Accuracy de 0,750 para 0,816; F1 de 0,716 para 0,820; AUC de 0,944 para 0,974. O estudo também mostrou que alguns métodos de fusão mal ajustados não melhoraram e chegaram a reduzir o desempenho. |
| [Ward et al., 2024 — SCIN](https://doi.org/10.1001/jamanetworkopen.2024.46615) | 10.408 imagens de 5.033 contribuições, com sintomas e dados demográficos autorreportados e diferenciais de dermatologistas | Quantidade de variáveis autorreportadas disponível durante a anotação | A confiança do dermatologista aumentou com o número de variáveis disponíveis (`Spearman R = 0,15`, `p < 0,001`). A proporção de casos em que foi possível produzir um diferencial variou de 59% sem variáveis autorreportadas para 93% com dez variáveis. Esta comparação também variou no número de dermatologistas por caso, portanto demonstra associação e não isola um efeito causal puro da metadata. |
| [Jiang et al., 2026 — avaliação real de MLLMs](https://arxiv.org/abs/2605.04098) | Três datasets públicos e 5.811 consultas hospitalares, com 46.405 imagens; geração de diferencial e triagem | Imagem apenas vs. imagem com contexto clínico | Na coorte real, o melhor resultado Top-3 dos modelos abertos passou de um máximo de 13,35% com imagem para 28,75% com contexto. No GPT-4.1 passou de 24,65% para 38,93%. É um preprint recente e deve ser citado como evidência preliminar, não como validação clínica definitiva. |

O estudo SCIN é especialmente relevante para a política `ASK_CONTEXT`: os
dados autorreportados incluíam aspetos que o utilizador consegue fornecer
remotamente, como localização anatómica, duração, sintomas e textura. O
resultado suporta a ideia de pedir apenas contexto que possa realmente
resolver uma ambiguidade entre diagnósticos.

Há também evidência para `REQUEST_BETTER_IMAGE`. Num estudo prospetivo
randomizado com 360 adultos, [Saade et al.,
2025](https://doi.org/10.1155/ijta/5789165), a concordância diagnóstica dos
dermatologistas foi de 79% com fotografias tiradas pelo doente sem orientação,
84% depois de instruções breves e 87% com fotografias padronizadas tiradas por
um residente. A confiança também aumentou de 6,43 para 6,91 numa escala de
1–10. Este resultado é humano e não garante o mesmo ganho num modelo, mas
justifica testar se a deteção de baixa qualidade e o pedido de nova fotografia
melhoram o sistema.

#### Consequência experimental para esta tese

A melhoria não deve ser assumida; deve ser medida com uma ablação multimodal
nos mesmos casos:

1. `image_only`: o modelo recebe apenas a fotografia;
2. `context_only`: o modelo recebe apenas os dados clínicos, para detetar
   atalhos e leakage;
3. `image_plus_context`: recebe fotografia e contexto clínico;
4. opcionalmente, `image_plus_requested_context`: começa apenas com a imagem e
   recebe a resposta a uma pergunta escolhida pelo modelo.

As quatro condições devem usar o mesmo split, taxonomia, prompt de output e
métricas. Devem ser comparados Top-1, Top-3, macro-F1, balanced accuracy,
calibração, taxa de abstention e intervalos de confiança emparelhados. O ganho
multimodal pode ser definido como:

```text
Δ contexto = score(image_plus_context) - score(image_only)
```

Para avaliar o valor da interação:

```text
Δ pergunta = score(depois_da_resposta) - score(antes_da_pergunta)
```

Também interessa medir o custo desse ganho: número médio de perguntas,
latência adicional e percentagem de perguntas que alteram corretamente o
Top-1 ou o Top-3. Desta forma, a tese não se limita a repetir que contexto
ajuda; demonstra em que classes ajuda, quanto ajuda e quando o modelo deveria
solicitá-lo.

## 4. Partições finais

As partições atuais são adequadas:

| Conjunto | Imagens | Grupos | Classes |
| --- | ---: | ---: | ---: |
| Train | 6.355 | 4.962 | 21 |
| Validation | 1.763 | 1.063 | 21 |
| Internal test | 1.704 | 1.063 | 21 |
| External DDI | 300 | 299 | 8 |
| External SkinDisNet | 1.365 | 333 | 4 |

Não se recomenda ficar apenas com Train e Validation:

- Train serve para otimizar os parâmetros;
- Validation serve para selecionar checkpoints, prompts e limiares;
- Internal test serve para a avaliação final na distribuição interna;
- DDI e SkinDisNet medem generalização externa.

O teste interno não deve ser consultado repetidamente durante o
desenvolvimento. Caso seja utilizado para tomar decisões, transforma-se na
prática numa segunda validação.

Os resultados externos devem permanecer separados dos resultados internos.
DDI tem classes com suporte muito baixo e SkinDisNet contém múltiplas imagens
por paciente. Devem ser apresentados counts, resultados por classe e
intervalos de confiança agrupados por grupo/paciente.

## 5. Sistema interativo futuro

O sistema pretendido recebe uma fotografia e escolhe entre quatro ações:

```text
CLASSIFY
ASK_CONTEXT
REQUEST_BETTER_IMAGE
ABSTAIN_OUT_OF_DOMAIN
```

Fluxo conceptual:

```text
Fotografia
    |
    v
Qualidade e domínio
    |--- qualidade insuficiente ---> pedir nova fotografia
    |--- fora do domínio ---------> abster/reencaminhar
    |
    v
Modelo multimodal pequeno
    |
    v
Morfologia + diferencial + confiança
    |--- confiança suficiente ----> apresentar resultado
    |--- candidatos semelhantes --> pedir contexto
    |--- evidência insuficiente ---> abster/reencaminhar
```

Esta política deve ser explícita e calibrada na validação. Não é suficiente
pedir ao modelo que decida livremente através de texto.

Um exemplo de política:

```text
se qualidade < limiar:
    REQUEST_BETTER_IMAGE
senão se OOD > limiar:
    ABSTAIN_OUT_OF_DOMAIN
senão se confiança top-1 < limiar de abstenção:
    ABSTAIN_OUT_OF_DOMAIN
senão se confiança top-1 - confiança top-2 < limiar de ambiguidade:
    ASK_CONTEXT
senão:
    CLASSIFY
```

Os limiares devem ser determinados na validação e posteriormente congelados.
As probabilidades declaradas por um modelo generativo não devem ser assumidas
como calibradas. Será necessário avaliar temperature scaling, isotonic
regression, conformal prediction ou outra técnica adequada ao modelo
selecionado.

## 6. Pedido de contexto

Pedir contexto é especialmente relevante em dermatologia. Doenças visualmente
semelhantes podem ser distinguidas por:

- idade;
- duração;
- progressão;
- comichão;
- dor;
- localização;
- recorrência;
- exposição;
- medicamentos;
- sintomas sistémicos.

Para manter o sistema avaliável:

- limitar a interação a uma ou duas perguntas;
- utilizar um vocabulário controlado de perguntas;
- selecionar perguntas que distingam os candidatos principais;
- medir a alteração do ranking depois da resposta;
- impedir perguntas irrelevantes ou repetidas.

SCIN poderá permitir simular esta interação, pois contém imagens, sintomas,
duração e outros dados reportados pelo utilizador.

Devem ser comparadas três condições:

```text
Imagem apenas
Imagem + todo o contexto disponível
Imagem + contexto pedido adaptativamente
```

## 7. Pedido de nova fotografia

O cenário `REQUEST_BETTER_IMAGE` necessita de exemplos de:

- desfocagem;
- iluminação insuficiente;
- sobre-exposição;
- lesão cortada;
- resolução insuficiente;
- distância inadequada;
- oclusão;
- imagem sem conteúdo dermatológico utilizável.

Podem ser utilizados casos já marcados como não avaliáveis e degradações
controladas de imagens válidas. O objetivo não é ensinar o modelo a pedir
sempre outra imagem quando está inseguro, mas a reconhecer insuficiência
visual concreta.

## 8. Fora do domínio e abstenção

Devem ser distinguidas três situações:

1. doença pertencente à taxonomia;
2. doença dermatológica fora das 21 classes;
3. imagem não dermatológica.

Uma doença dermatológica fora da taxonomia não deve ser forçada para uma das
classes disponíveis.

Exemplo:

```json
{
  "action": "abstain",
  "reason": "possible_dermatological_condition_outside_supported_taxonomy"
}
```

Possíveis métricas:

- AUROC para OOD;
- FPR@95TPR;
- selective accuracy;
- curva risk–coverage;
- taxa de abstenção;
- taxa de erro confiante em OOD;
- cobertura do sistema.

## 9. O que fazer depois do diagnóstico

### 9.1 Recomendação principal: um SLM textual com RAG

Depois do diagnóstico, o modelo multimodal pode enviar um objeto estruturado a
um único SLM textual:

```json
{
  "findings": ["plaque", "scale", "erythema"],
  "top_diagnoses": [
    {"disease": "psoriasis", "confidence": 0.72},
    {"disease": "eczema", "confidence": 0.18}
  ],
  "user_context": {
    "itching": true,
    "duration": "several_months"
  }
}
```

Um sistema RAG recuperaria informação validada para os diagnósticos principais
e o SLM produziria:

- explicação do diferencial;
- relação entre achados e diagnóstico;
- incerteza e limitações;
- indicação de quando procurar avaliação presencial;
- fontes utilizadas.

Devem ser recuperados documentos para o Top-3, não apenas para o Top-1. Um
erro inicial não deve encaminhar todo o processo para conhecimento sobre a
doença errada.

Esta arquitetura mantém responsabilidades claras:

```text
Modelo multimodal pequeno: perceção e diagnóstico
SLM textual + RAG: explicação fundamentada
```

### 9.2 Routing para modelos especialistas

Não se recomenda começar com um SLM independente para cada uma das 21
doenças. Isso fragmentaria os dados, aumentaria custos operacionais e tornaria
difícil atribuir as melhorias a uma componente concreta.

Uma experiência posterior mais viável seria routing por famílias:

- neoplásicas;
- inflamatórias;
- infecciosas/infestações;
- pigmentares;
- foliculares e outras.

Os especialistas poderiam ser:

- adapters LoRA sobre o mesmo backbone;
- pequenos modelos por família;
- um SLM único com índices RAG por família.

Para avaliar routing:

| Condição | Objetivo |
| --- | --- |
| Generalista sem routing | Baseline |
| Router previsto + especialista | Sistema real |
| Router oracle + especialista correto | Limite superior |
| RAG condicionado pelo Top-3 | Alternativa mais simples |

A diferença entre router oracle e router previsto mede o custo dos erros de
routing. Esta parte deve ser opcional ou trabalho futuro, exceto se a tese for
explicitamente reformulada como uma tese sobre mixture-of-experts.

## 10. Necessidade de dados sintéticos

É provável que sejam necessários dados sintéticos para ensinar a política
interativa, porque os datasets atuais supervisionam sobretudo diagnóstico,
morfologia, captions ou perguntas/respostas; não fornecem diretamente labels
para as quatro ações do sistema.

No entanto, deve distinguir-se:

### Dados sintéticos recomendados

- instruções e respostas estruturadas;
- diálogos de pedido de contexto;
- seleção da pergunta seguinte;
- pedidos de nova fotografia;
- exemplos de abstenção;
- outputs com morfologia, diferencial, confiança e evidência;
- explicações fundamentadas em referências;
- perturbações controladas de qualidade.

### Dados sintéticos que não são prioritários

Não se recomenda começar por gerar imagens dermatológicas artificiais. Imagens
sintéticas podem introduzir artefactos, morfologia impossível ou atalhos
visuais. Primeiro devem ser exploradas imagens reais com supervisão sintética
de interação e decisão.

## 11. Como construir o dataset sintético

Cada exemplo sintético deve derivar de um caso real do conjunto de treino.
Nunca se devem utilizar Validation, Internal Test, DDI externo ou SkinDisNet
externo para gerar exemplos de fine-tuning.

Schema conceptual:

```text
scenario_id
source_sample_id
image_uri
scenario_type
visible_context
hidden_context
target_action
target_question
simulated_user_answer
target_findings
target_description
target_differential
target_confidence
target_evidence_links
provenance
teacher_model
teacher_prompt_version
validation_status
```

### Cenário CLASSIFY

- imagem real, válida e in-domain;
- label real;
- conceitos SKINCON quando disponíveis;
- teacher produz diferencial e descrição estruturada;
- filtros verificam schema, doença e conceitos.

### Cenário ASK_CONTEXT

- escolher casos em que dois diagnósticos sejam plausíveis;
- ocultar inicialmente parte do metadata SCIN;
- teacher seleciona uma pergunta de um catálogo controlado;
- a resposta simulada vem do metadata real, não é inventada;
- avaliar se o novo contexto altera corretamente o ranking.

### Cenário REQUEST_BETTER_IMAGE

- aplicar blur, crop, compressão, baixa resolução, oclusão ou exposição;
- manter a imagem original como referência;
- target action é pedir nova fotografia;
- guardar transformação e parâmetros para reprodução.

### Cenário ABSTAIN_OUT_OF_DOMAIN

- utilizar doenças dermatológicas excluídas da taxonomia;
- adicionar um conjunto licenciado de imagens não dermatológicas;
- separar OOD dermatológico de OOD não dermatológico;
- impedir que o teacher force uma das 21 doenças.

### Explicação pós-diagnóstico

- fornecer ao teacher apenas diagnóstico, achados verificados e fontes;
- exigir referências ou IDs de documentos;
- rejeitar afirmações não suportadas;
- evitar recomendações terapêuticas específicas sem validação clínica.

## 12. Controlo de qualidade sintético

Os outputs do teacher não devem ser aceites automaticamente.

Pipeline recomendado:

```text
Caso real de treino
    |
    v
Geração pelo teacher
    |
    v
Validação de schema
    |
    v
Validação determinística de labels e conceitos
    |
    v
Deteção de contradições e conteúdo proibido
    |
    v
Deduplicação
    |
    v
Amostra revista manualmente
    |
    v
Dataset de fine-tuning versionado
```

Guardar sempre:

- modelo e revisão do teacher;
- prompt e versão;
- temperatura e seed;
- referências utilizadas;
- origem do caso;
- filtros aplicados;
- razão de aceitação/rejeição.

Deve existir uma auditoria humana estratificada, sobretudo para:

- casos malignos;
- pele escura;
- doenças raras;
- abstenções;
- perguntas de contexto;
- recomendações de avaliação presencial.

## 13. Experiências futuras para a política interativa

Comparar:

```text
A. Classificar sempre
B. Classificar ou abster
C. Classificar, pedir contexto, pedir imagem ou abster
```

Métricas:

- Top-1/3/6 e MRR;
- accuracy seletiva;
- curva risk–coverage;
- taxa de perguntas;
- ganho após contexto;
- número médio de turnos;
- taxa de pedidos desnecessários de nova fotografia;
- qualidade da pergunta;
- taxa de OOD aceite incorretamente;
- erro confiante;
- custo e latência por caso resolvido.

## 14. Sequência recomendada do projeto

### Fase atual

1. congelar benchmarks, schemas, prompts e partições;
2. correr todos os modelos candidatos;
3. selecionar teacher na validação;
4. selecionar o small multimodal model;
5. registar baseline zero-shot;
6. manter testes internos e externos fechados.

### Fine-tuning inicial

1. criar dataset supervisionado apenas com dados de Train;
2. gerar outputs sintéticos do teacher quando necessário;
3. filtrar e auditar;
4. fazer fine-tuning do modelo pequeno;
5. selecionar checkpoint na Validation;
6. comparar base e fine-tuned na Validation.

### Avaliação final

1. congelar modelo, prompt, thresholds e política;
2. executar uma vez no Internal Test;
3. executar no External DDI;
4. executar no External SkinDisNet;
5. reportar desempenho, eficiência, subgrupos e incerteza.

### Extensões

1. política interativa;
2. dataset sintético de cenários;
3. SLM textual com RAG;
4. routing por família/adapters;
5. estudo oracle versus router previsto.

## 15. Recomendação final

A narrativa mais coerente para a tese é:

> Um pequeno modelo multimodal pode aproximar o desempenho de modelos maiores
> quando é especializado num domínio fechado, oferecendo menor custo e maior
> controlo estrutural. Uma política explícita de incerteza pode ainda permitir
> que o sistema saiba quando diagnosticar, quando pedir informação, quando
> pedir uma imagem melhor e quando se deve abster.

O routing por doença não é necessário para provar esta hipótese. Um SLM
textual com RAG é uma segunda etapa mais simples e controlável. Routing por
famílias pode ser uma experiência adicional depois de estarem estabelecidos o
teacher, o modelo pequeno e os resultados baseline.
