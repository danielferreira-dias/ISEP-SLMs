# Limitações de output, parsing e política de métricas

## 1. Objetivo desta nota

Esta nota consolida as decisões e observações obtidas durante os smoke tests
dos modelos multimodais. O objetivo é separar quatro questões que não devem
ser confundidas:

1. capacidade clínica do modelo;
2. capacidade natural de seguir o formato pedido;
3. capacidade do backend de restringir a geração;
4. robustez necessária para integrar o modelo numa aplicação.

Um modelo pode identificar corretamente a doença e, ainda assim, produzir
JSON com Markdown, confiança inconsistente ou evidência mal referenciada.
Inversamente, JSON perfeitamente válido não garante um diagnóstico correto.

Os resultados numéricos desta nota resultam de smoke tests de dez casos. Não
devem ser interpretados como estimativas finais de desempenho nem utilizados
para conclusões estatísticas.

## 2. Representações guardadas pela pipeline

Cada resposta deve preservar representações diferentes e auditáveis:

| Campo | Conteúdo | Utilização |
| --- | --- | --- |
| `raw_text` | Conteúdo original devolvido pelo modelo | Auditoria |
| `reasoning` | Reasoning ou summary exposto pelo provider/modelo | Auditoria; nunca scoring |
| `final_text` | Resposta depois de separar reasoning model-specific | Parsing e validação |
| `parsed_output` | JSON recuperado sem alterar o significado | Métricas por campo |
| `canonical_output` | Projeção determinística para uma forma equivalente | Métricas clínicas canónicas |
| `validation_errors` | Violações de formato, schema e semântica | Reliability metrics |

O output original nunca deve ser substituído pelo output normalizado. Esta
separação permite reproduzir a avaliação e alterar a apresentação sem
reescrever aquilo que o modelo realmente produziu.

## 3. Modos de structured output

### 3.1 `prompt_only`

O modelo recebe na prompt a descrição do contrato e produz livremente a
resposta. Esta condição mede simultaneamente:

- reconhecimento clínico;
- seguimento natural de instruções;
- capacidade natural de serialização;
- estabilidade do modelo.

Esta deve ser a condição principal para a comparação científica entre
modelos, porque todos recebem o mesmo tipo de restrição: apenas a prompt.

### 3.2 `json_schema`

O backend envia o JSON Schema através de `response_format`. O vLLM ou o
provider restringe os tokens permitidos durante a geração.

Esta condição responde a uma pergunta diferente:

> Qual é o desempenho do modelo quando integrado num sistema que restringe a
> forma do output?

Deve ser tratada como uma experiência de deployment e nunca misturada na
mesma tabela principal com resultados `prompt_only` sem identificar o modo.

JSON Schema garante sobretudo sintaxe e estrutura. Não garante:

- diagnóstico correto;
- confiança coerente;
- evidência realmente visível;
- conceitos da descrição declarados em `findings`;
- ausência de informação clínica não visual;
- grounding correto.

### 3.3 Decisão específica para MedGemma

MedGemma fica limitado a `prompt_only`.

No smoke test de Evidence-Grounded Diagnosis com `json_schema`, oito de dez
respostas consumiram o limite de 8192 tokens e foram truncadas. A gramática
permitiu geração prolongada de uma estrutura incompleta. O constrained
decoding resolveu parte da sintaxe, mas introduziu uma falha operacional
grave.

O launcher rejeita modos diferentes de `prompt_only` antes de iniciar a GPU.
Os artefactos do smoke test com JSON Schema permanecem apenas como evidência
da limitação e não definem uma configuração suportada.

## 4. Parsers e normalizações implementadas

### 4.1 JSON estrito

O primeiro passo tenta interpretar `final_text` diretamente como um único
objeto JSON. Este resultado alimenta `json_validity_rate`.

Não são aceites:

- texto antes ou depois do JSON;
- Markdown;
- múltiplos objetos;
- comentários;
- números JSON não standard;
- chaves duplicadas.

### 4.2 Recuperação de uma única JSON fence

É aceite, para análise recuperável, exatamente uma fence completa:

````text
```json
{"field": "value"}
```
````

Esta regra:

- não altera os valores;
- não extrai labels de prosa;
- mantém o caso como `format_invalid`;
- permite validar os campos internos;
- alimenta `recoverable_json_validity_rate`.

Foi especialmente relevante para Kimi e MedGemma. Sete das dez respostas do
Kimi no smoke test de evidence estavam envolvidas numa JSON fence.

### 4.3 Canonicalização de ranked lists

MiniCPM produziu frequentemente:

```json
{
  "predictions": ["D003", "D014", "D009"]
}
```

O schema pedia:

```json
{
  "predictions": [
    {"rank": 1, "disease_id": "D003"},
    {"rank": 2, "disease_id": "D014"},
    {"rank": 3, "disease_id": "D009"}
  ]
}
```

Como a ordem da lista determina inequivocamente o rank, a pipeline pode criar
uma projeção canónica `ranked_disease_id_list_to_objects`. A resposta estrita
continua inválida e a regra utilizada fica guardada em
`canonicalization_rules`.

São reportadas duas famílias:

- métricas estritas;
- métricas `canonical_*`.

### 4.4 Separação do reasoning do MedGemma

MedGemma pode colocar reasoning dentro do content:

```text
<unused94>thought
...
<unused95>
{"predictions": [...]}
```

O parser `medgemma_special_tokens`:

1. extrai apenas blocos completos entre `<unused94>` e `<unused95>`;
2. remove a label literal `thought`;
3. guarda o bloco em `response.reasoning`;
4. envia apenas o conteúdo exterior para `response.final_text`;
5. nunca infere uma resposta final a partir do reasoning.

Se não existir conteúdo depois de `<unused95>`, `final_text` fica vazio e o
caso permanece inválido.

### 4.5 Reasoning fornecido por APIs

A pipeline reconhece:

- `reasoning`;
- `reasoning_content`;
- reasoning de streams;
- summaries oficiais da Responses API;
- contagem de reasoning tokens.

O Luna expõe summaries oficiais quando disponíveis. No smoke test de dez
casos, cinco continham summary e cinco apenas contagem de reasoning tokens.

O Kimi, na configuração atual com thinking desativado, não expôs reasoning.

Reasoning nunca é interpretado como resposta, nunca é utilizado para reparar
um diagnóstico e nunca contribui diretamente para as métricas.

## 5. Reparações deliberadamente proibidas

Para preservar a validade da avaliação, a pipeline não deve:

- extrair doenças de prosa livre com regex;
- utilizar embeddings ou similarity para escolher a label mais próxima;
- utilizar outro LLM para corrigir a resposta durante scoring;
- inferir o JSON final a partir do chain-of-thought;
- completar arrays incompletos;
- corrigir probabilidades ou confidence bands;
- inventar evidence links ausentes;
- substituir IDs desconhecidos por IDs parecidos.

Estas operações introduziriam inteligência do avaliador e poderiam aumentar
artificialmente o desempenho do modelo.

## 6. Camadas de validação e status

A precedência dos status é:

1. `truncated_output`;
2. `format_invalid`;
3. `schema_invalid`;
4. `semantic_noncompliant`;
5. `ok`.

Outros status terminais incluem:

- `safety_refusal`;
- `backend_error`;
- `image_error`.

O status identifica a primeira camada que falhou. Um caso `format_invalid`
pode também conter erros semânticos, que continuam guardados em
`validation_errors`.

### 6.1 Formato

Pergunta:

> O modelo devolveu exatamente um objeto JSON, sem Markdown ou prosa?

Métricas:

- `json_validity_rate`;
- `recoverable_json_validity_rate`;
- `truncated_output`;
- `format_invalid`.

### 6.2 Schema

Pergunta:

> Os campos, tipos, cardinalidades, IDs e referências obedecem ao contrato?

Inclui:

- campos obrigatórios;
- número exato de diagnósticos;
- ranks consecutivos;
- IDs dentro da taxonomia;
- ausência de duplicados;
- probabilities entre zero e um;
- `supporting_finding_ids` que resolvem para achados declarados.

Métricas:

- `schema_compliance_rate`;
- `invalid_disease_id_rate`;
- `invalid_concept_id_rate`;
- `duplicate_prediction_rate`;
- `duplicate_finding_rate`;
- `broken_evidence_reference_rate`.

### 6.3 Semântica do contrato

Pergunta:

> Os campos são coerentes entre si e representam apenas evidência permitida?

Inclui:

- confiança não crescente no diferencial;
- `case_confidence` compatível com a probabilidade Top-1;
- conceitos da descrição também declarados em `findings`;
- diagnósticos com evidence links;
- descrição sem nomes de doenças, testes ou informação não visual.

Métrica principal:

- `semantic_compliance_rate`.

## 7. Métricas clínicas

### 7.1 Diagnóstico

- Top-1 accuracy;
- Top-3 accuracy;
- Top-6 accuracy;
- Mean Reciprocal Rank;
- macro-F1 Top-1.

### 7.2 Morfologia

- finding precision;
- finding recall;
- finding F1;
- micro-F1 de conceitos;
- macro-F1 de conceitos com suporte suficiente;
- unsupported finding rate.

### 7.3 Descrição clínica

- description concept precision;
- description concept recall;
- description concept F1;
- consistência entre descrição e `findings`;
- unsupported description concept rate;
- forbidden description content rate.

### 7.4 Grounding e calibração

- valid evidence link rate;
- visible evidence precision;
- grounded Top-1 success;
- correct diagnosis unsupported evidence rate;
- Brier score;
- Expected Calibration Error.

Não se recomenda um único score agregado. Um score único pode esconder que um
modelo classifica bem, mas não consegue justificar o diagnóstico, ou produz
JSON correto com fraca capacidade clínica.

## 8. Relação entre outputs inválidos e accuracy

Um status inválido não reduz automaticamente a accuracy diagnóstica.

As métricas clínicas utilizam campos recuperáveis e individualmente válidos.
Por exemplo, uma JSON fence ou uma categoria de confiança incoerente não
transforma automaticamente um diagnóstico Top-1 correto num erro.

Em paralelo, a falha reduz:

- JSON validity;
- schema compliance, quando aplicável;
- semantic compliance;
- end-to-end success.

Esta política mede separadamente:

1. inteligência clínica;
2. instruction following;
3. reliability de integração.

No smoke test de Kimi:

- existiam dez casos no total;
- apenas cinco pertenciam à coorte com ground truth diagnóstico;
- Top-1 foi `3/5 = 60%`;
- Top-3 foi `4/5 = 80%`;
- JSON estrito foi `3/10 = 30%`;
- JSON recuperável foi `10/10 = 100%`;
- semantic compliance foi `0/10 = 0%`;
- end-to-end `ok` foi `0/10`.

Os cinco casos sem ground truth diagnóstico pertencem às coortes de
morfologia e descrição. Não devem ser tratados como labels em falta nem
incluídos no denominador da accuracy diagnóstica.

## 9. Limitações observadas por modelo

### 9.1 MiniCPM-V 4.6

- Produziu ranked lists como arrays de IDs, em vez dos objetos pedidos.
- A inteligência diagnóstica pode ser medida através da canonicalização
  determinística.
- O formato estrito continua penalizado.
- JSON Schema corrigiu o Top-K estruturalmente.
- No evidence com JSON Schema, todos os outputs foram estruturalmente válidos,
  mas nenhum cumpriu completamente o contrato semântico.

### 9.2 MedGemma 1.5 4B

- Pode misturar reasoning no content com tokens especiais.
- O parser model-specific separa reasoning e resposta final.
- Em prompt-only, pode utilizar JSON fences e produzir reasoning extenso.
- JSON Schema no evidence causou loops/estruturas longas e oito truncamentos
  em dez casos.
- Fica suportado apenas em `prompt_only`.

### 9.3 Gemma 4 E4B

- Produziu Top-K estruturalmente válido em prompt-only e JSON Schema.
- JSON Schema não alterou o smoke Top-K.
- No evidence, os principais erros foram de coerência semântica e não de
  sintaxe.
- Isto mostra que constrained decoding não resolve o contrato clínico.

### 9.4 Qwen 3.5 4B

- Foi consistente no Top-K nos dois modos do smoke test.
- No evidence prompt-only ocorreram respostas com formato inválido.
- JSON Schema garantiu estrutura, mas a semantic compliance permaneceu baixa.
- Structured output pode alterar o conteúdo clínico gerado, pelo que os dois
  modos são condições experimentais diferentes.

### 9.5 Luna

- No smoke de evidence, dez de dez respostas ficaram `ok`.
- JSON, schema e semântica atingiram 100% nesse pequeno conjunto.
- Top-1 foi inferior ao do Kimi no mesmo smoke.
- Grounded Top-1 foi zero, apesar da boa conformidade estrutural.
- Reasoning summary não esteve disponível em todos os casos; alguns apenas
  expuseram reasoning-token usage.

### 9.6 Kimi K2.6

- Sete de dez respostas estavam dentro de uma JSON fence.
- O parser recuperou todos os dez objetos sem alterar valores.
- Oito casos tinham `case_confidence` inconsistente com a confiança Top-1.
- Três descrições mencionaram conceitos não declarados em `findings`.
- Duas descrições incluíram informação de biopsia, proibida por não ser
  evidência puramente visual.
- A accuracy diagnóstica recuperável foi boa no smoke, mas nenhum caso ficou
  totalmente `ok`.
- Thinking está desativado e o provider não devolveu reasoning.

## 10. Observações dos smoke tests

### 10.1 Visual Top-K

| Modelo | Modo | Top-1 | Top-3 | Top-6 | Observação |
| --- | --- | ---: | ---: | ---: | --- |
| Gemma E4B | prompt | 30% | 50% | 50% | 10/10 schema válido |
| Gemma E4B | JSON Schema | 30% | 50% | 50% | Sem alteração no smoke |
| MiniCPM | prompt, canónico | 20% | 40% | 60% | 0% schema estrito |
| MiniCPM | JSON Schema | 20% | 40% | 70% | 100% schema |
| Qwen 4B | prompt | 50% | 60% | 80% | 10/10 `ok` |
| Qwen 4B | JSON Schema | 50% | 60% | 80% | 10/10 `ok` |
| MedGemma | prompt, canónico | 20% | 80% | 90% | 0% formato estrito |

O resultado de MedGemma com JSON Schema não é incluído como modo suportado.

### 10.2 Evidence-Grounded Diagnosis

| Modelo | Modo | Top-1 | Top-3 | Grounded Top-1 | Semantic |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma E4B | prompt | 40% | 60% | 20% | 30% |
| Gemma E4B | JSON Schema | 40% | 60% | 20% | 40% |
| MiniCPM | prompt | 0% | 0% | 0% | 0% |
| MiniCPM | JSON Schema | 0% | 20% | 0% | 0% |
| Qwen 4B | prompt | 40% | 40% | 0% | 30% |
| Qwen 4B | JSON Schema | 60% | 60% | 0% | 20% |
| MedGemma | prompt | 0% | 40% | 0% | 50% |
| Luna | prompt | 20% | 40% | 0% | 100% |
| Kimi | prompt | 60% | 80% | 0% | 0% |

Estes valores não são diretamente comparáveis como resultados finais:

- `n=10`;
- a coorte diagnóstica do smoke de evidence tinha apenas cinco casos;
- alguns modelos usam sampling;
- JSON Schema altera o espaço de geração;
- não existem intervalos de confiança informativos com esta amostra.

## 11. Limitações operacionais adicionais

### 11.1 Limite de output

Não existe inferência verdadeiramente sem limite. O limite atual de 8192
tokens funciona como proteção contra:

- loops;
- reasoning sem resposta final;
- whitespace ou JSON incompleto;
- custos API inesperados;
- ocupação prolongada de GPUs.

O limite não deve ser interpretado apenas como restrição: `truncated_output`
é também uma métrica de estabilidade.

### 11.2 Async API lifecycle

O cliente `AsyncOpenAI` deve ser fechado no mesmo event loop em que executou
os requests. Um smoke do Kimi revelou um erro `Event loop is closed` durante
cleanup. Os dez casos já tinham sido gravados, mas o report não foi gerado
automaticamente.

A pipeline foi corrigida para executar requests e `aclose()` no mesmo event
loop. O erro não alterou predictions ou métricas desse run.

### 11.3 Azure API version

Luna e Kimi utilizam o endpoint Azure `/openai/v1`. Não é necessária uma
variável `AZURE_API_VERSION`; endpoint, deployment e API key são suficientes.

## 12. Decision gates antes das benchmarks completas

As decisões seguintes devem ser congeladas antes das execuções completas.

### 12.1 Modo principal por modelo

Recomendação:

- utilizar `prompt_only` como condição principal para todos os modelos;
- utilizar `json_schema` apenas como ablation de deployment em modelos
  comprovadamente compatíveis;
- manter MedGemma exclusivamente em `prompt_only`;
- não misturar modos na seleção do teacher.

### 12.2 Política de scoring principal

É necessário declarar previamente:

- métrica clínica primária;
- métricas secundárias;
- reliability gates;
- tratamento de outputs recuperáveis.

Recomendação:

- Top-1 e Grounded Top-1 como endpoints clínicos principais;
- Top-3, morphology F1 e description F1 como endpoints secundários;
- JSON/schema/semantic compliance reportados separadamente;
- end-to-end `ok` como métrica operacional;
- sem score agregado único.

### 12.3 Partição utilizada para escolher o teacher

Teacher, prompts e decisões de parsing devem ser escolhidos apenas na
Validation.

Existe atualmente uma lacuna de configuração que deve ser resolvida antes da
seleção real do teacher:

- `validation.parquet` existe para Visual Top-K, mas ainda não está exposto
  como `evaluation_set` na configuração da benchmark;
- Evidence-Grounded Diagnosis expõe apenas `external_ddi_evidence`;
- não existe atualmente uma coorte interna de Validation com a mesma
  combinação de labels DDI, SKINCON e captions SkinCAP.

Assim, o evidence externo pode ser utilizado como teste final ou smoke test
de engenharia, mas não deve ser utilizado repetidamente para escolher
prompts, parsers ou o teacher. É necessário decidir se:

1. Evidence-Grounded Diagnosis será exclusivamente uma avaliação externa
   final; ou
2. será criada uma coorte de desenvolvimento separada com anotações
   equivalentes.

O Internal Test completo, o benchmark interno fixo e os conjuntos externos
não devem ser utilizados para ajustar:

- prompts;
- thresholds;
- parsers;
- sampling;
- escolha do teacher;
- regras semânticas.

Os dez casos externos já inspecionados devem ser documentados como smoke test
de engenharia, não como evidência final nem como dados de desenvolvimento.

### 12.4 Sampling e repetições

Alguns modelos usam sampling. Um único run mistura capacidade do modelo com
variância de geração.

É necessário decidir entre:

1. um único run primário com seed/config congelados;
2. múltiplas repetições para todos;
3. um run para todos e três repetições apenas para finalistas.

Recomendação de custo controlado:

- um run completo por candidato na Validation;
- três repetições para os dois ou três finalistas;
- uma única execução selada no Internal Test e conjuntos externos.

### 12.5 Reasoning capture

É necessário manter a mesma política dentro de cada comparação.

Recomendação:

- `available` para auditoria durante seleção do teacher;
- reasoning sempre separado do answer;
- não usar reasoning bruto como target de treino;
- reportar disponibilidade como `full`, `summary`, `tokens_only` ou `none`;
- não penalizar um modelo apenas porque o provider não expõe reasoning.

### 12.6 Limite de tokens

Recomendação:

- manter 8192 como safety cap para os benchmarks atuais;
- reportar truncamentos;
- não aumentar o limite para esconder loops;
- reduzir apenas se uma análise de distribuição mostrar que respostas válidas
  utilizam muito menos tokens e que o novo limite não corta casos legítimos.

### 12.7 Congelamento de parsers e schemas

Antes dos runs finais devem ser versionados e congelados:

- prompt;
- schema;
- taxonomias;
- parser JSON;
- parser MedGemma;
- canonicalization rules;
- regras de semantic validation;
- preprocessing da imagem;
- seleção e seed;
- configurações dos modelos.

Não devem ser adicionadas reparações depois de observar os resultados do
Internal Test ou dos conjuntos externos.

### 12.8 Estatística e critério de seleção

Antes da comparação final devem ser definidos:

- intervalos de confiança emparelhados;
- método de bootstrap;
- margem de não inferioridade;
- regra de desempate;
- tratamento de safety refusals e backend errors;
- comparação de latência, tokens e custo.

## 13. Recomendação imediata

Ainda não é aconselhável iniciar todas as benchmarks finais.

Primeiro devem ser tomadas e registadas quatro decisões:

1. confirmar `prompt_only` como condição principal comum;
2. escolher as métricas primárias e reliability gates;
3. expor a Validation do Visual Top-K e decidir se Evidence será apenas teste
   externo ou terá uma coorte de desenvolvimento separada;
4. decidir a política de uma ou múltiplas repetições para modelos com
   sampling.

Depois destas decisões, devem ser executados:

1. um dry run final de configuração;
2. a benchmark completa de Validation;
3. a seleção congelada de teacher e student;
4. apenas no fim, Internal Test e avaliações externas.
