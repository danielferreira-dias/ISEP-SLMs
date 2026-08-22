# E3 Stage A teacher dataset — geração Batch, auditoria e freeze

Data do freeze final: 2026-08-22.

## Resultado

O Stage A answer-blind foi gerado com `gemini-3.7-flash` no Vertex, reasoning
`medium`, seed 42 e a prompt congelada `e3_stage_a_v1`. A fonte é
`danielfdias98/ISEPDistillDataset`, revisão
`b215f0474e4931b5951da768e79a0d579d26919d`, configuração `diagnosis`, split
`sft_train`.

O Batch v3 terminou com 5.999 pedidos processados pelo provider. A validação
local inicial aceitou 5.991 e colocou oito em quarentena: uma resposta vazia,
uma truncada por `MAX_TOKENS` e seis rejeitadas pelo validador de observações
duplicadas. Uma única passagem síncrona recuperou seis casos. A inspeção dos
dois restantes mostrou que não eram duplicados semânticos: cada imagem continha
duas lesões distintas com a mesma morfologia primária `papule`, mas scopes e
regiões de evidência diferentes. O validador passou a rejeitar apenas
repetições com conceito, valor, estado, scope e evidência iguais. Esta foi uma
correção de materialização; não alterou prompt, schema enviado ao Teacher,
imagem, seed ou target textual.

Cobertura final:

- 6.312 rows aceites e 6.312 `sample_id` únicos;
- 6.291 imagens avaliáveis e 21 corretamente marcadas como não avaliáveis;
- 6.312 captions únicas e 6.312 objetos morfológicos serializados únicos;
- zero campos `gold_diagnosis` e zero menções exatas do gold label no target;
- audit log preservado com 6.318 tentativas: 6.312 `ok` e seis `error`;
- oito outputs Batch inválidos preservados em quarentena.

## Custos

O materializador separa agora preços Standard dos preços Batch. Para Batch
global foram fixados USD 0,375/M tokens de input e USD 1,875/M tokens de
output/reasoning. O canário, anteriormente registado por engano à tarifa
Standard, foi recalculado a partir do output Batch original; apenas a metadata
de custo mudou e foi preservado um backup integral.

Custo list-price conhecido da campanha Stage A:

| Componente | USD |
| --- | ---: |
| 312 gerações síncronas anteriores + tentativa síncrona inválida faturável | 2,964814 |
| canário Batch | 0,004626 |
| Batch v3 completo | 27,972857 |
| passagem síncrona dos oito casos | 0,075207 |
| **total conhecido** | **31,017504** |

O total é uma estimativa por tokens e não substitui a fatura Cloud Billing. Não
inclui storage residual, impostos, conversão cambial ou créditos promocionais.

## Freeze final

Diretório final:

`project/data/morphology/frozen/e3_stage_a_v1_20260822/`

Hashes principais:

| Artefacto | SHA-256 |
| --- | --- |
| accepted-only `stage_a.jsonl` | `1eefa665d791c5138ffc00d57c5d9161ab899985949d8d4c2f7e54d12db89bd2` |
| prompt Stage A | `c28f6ff4f9a47ba23bc02f2a6d14541ee5afeeaf134bca5cf48936f150121a4f` |
| schema Stage A, bytes do ficheiro | `d2c328f33e819bb87bf886055a7005e65bcc13f594a9a3af23b73a56da8d8029` |
| configuração Vertex | `8d6dd5feae3af2b955e9e370e012750f97c09024beaf77505c0731cbd0218963` |
| audit log canónico | `bea352593b9f9d29feec4417bbcd18b0fc21ffb24ee189e6ec22bc8e412716b1` |

O freeze preliminar `e3_stage_a_v1_20260821` fica preservado mas foi
supersedido: continha os mesmos targets, porém ainda guardava o custo do canário
à tarifa Standard. A release de 2026-08-22 é a referência final.

## Limitações de aderência à prompt

A estrutura e a separação do gold passaram, mas a aderência linguística do
Teacher não foi perfeita. Foram preservadas e devem ser declaradas na tese:

- `melanocytic nevi`: quatro rows;
- `pigmented nevi`: uma row;
- `solar lentigines`: cinco rows;
- `photodamaged`, `indurated` e `acute`: uma row cada;
- `palpable`: 77 rows, maioritariamente em declarações de que a propriedade não
  era avaliável, apesar de a prompt pedir que o termo não fosse usado.

As dez referências a nevi/lentigines são termos diagnósticos incidentais e
violam o contrato answer-blind, mas nenhuma coincide com o gold label privado.
Não foram reescritas nem filtradas depois de observar o corpus, para evitar uma
correção pós-hoc silenciosa. Devem ser tratadas como uma limitação de
instruction-following do Teacher, não como leakage do gold.
