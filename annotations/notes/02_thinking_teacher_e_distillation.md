# Thinking do teacher e distillation para o modelo pequeno

## 1. Decisão principal

O teacher pode utilizar thinking internamente quando isso melhorar de forma
mensurável a qualidade clínica. Contudo, o modelo pequeno não deve ser
treinado automaticamente com o reasoning bruto produzido pelo teacher.

A estratégia principal é:

> Permitir que o teacher raciocine internamente, mas ensinar o student através
> de uma justificação clínica curta, estruturada, observável e verificável.

Esta separação é importante porque uma resposta final correta não garante que
todos os passos de um chain-of-thought longo sejam clinicamente corretos ou
estejam fundamentados na imagem.

## 2. Separar seleção do teacher de geração de dados

Existem dois objetivos experimentais diferentes:

1. selecionar o melhor teacher;
2. decidir como esse teacher deve gerar supervisão sintética.

### 2.1 Benchmark principal sem thinking

O benchmark principal deve ser executado com thinking desativado porque esta
condição representa melhor o comportamento eficiente pretendido para o
sistema final:

- menor latência;
- menor custo de inferência;
- menor risco de loops;
- menos outputs truncados;
- maior previsibilidade do JSON;
- comparação direta do resultado clinicamente observável.

Todos os candidatos devem ser comparados nas mesmas partições e o teacher deve
ser selecionado apenas com a Validation. O Internal Test e os conjuntos
externos permanecem selados.

### 2.2 Ablation com thinking

Depois do benchmark inicial, os dois ou três melhores candidatos devem ser
novamente avaliados com thinking ativado. Não é necessário pagar o custo desta
experiência para todos os modelos se alguns forem claramente inferiores.

Comparar pelo menos:

| Métrica | Sem thinking | Com thinking |
| --- | --- | --- |
| Accuracy e Top-K | Sim | Sim |
| Morphology grounding | Sim | Sim |
| Correção da descrição clínica | Sim | Sim |
| Correção da evidência | Sim | Sim |
| Validade do schema | Sim | Sim |
| Tokens e latência | Sim | Sim |
| Truncamentos e loops | Sim | Sim |

Thinking só deve ser utilizado pelo teacher para gerar dados se produzir uma
melhoria clínica relevante que compense o custo e a instabilidade adicionais.

O smoke test anterior do Qwen 3.6 27B com thinking mostrou a importância desta
ablation: cinco dos dez casos consumiram o limite total de 8192 output tokens
antes de produzirem uma resposta final. Este resultado não prova que thinking
seja sempre inferior, mas impede que seja adotado sem uma comparação
controlada.

## 3. Perfis de inferência a comparar

Para modelos que suportam explicitamente thinking e non-thinking, devem ser
distinguidas três condições:

1. `no_thinking_controlled`: thinking desativado, mantendo os parâmetros de
   sampling do perfil thinking;
2. `no_thinking_official`: thinking desativado e parâmetros oficialmente
   recomendados para instruct/non-thinking;
3. `thinking_official`: thinking e parâmetros oficiais ativados.

A primeira condição isola o efeito do interruptor `enable_thinking`. As duas
condições oficiais estimam o melhor desempenho esperado em cada modo.

As configurações atuais que mantêm os parâmetros de general tasks com
`enable_thinking: false` correspondem à condição `no_thinking_controlled`.
Não devem ser descritas como parâmetros non-thinking ótimos sem uma ablation.

A documentação do Qwen 3.5 indica que os modelos pequenos têm thinking
desativado por omissão e recomenda parâmetros diferentes para os modos
thinking e instruct:

- [Qwen 3.5 — Unsloth documentation](https://unsloth.ai/docs/models/qwen3.5)

## 4. O que guardar durante a geração sintética

O output bruto e o target de treino devem ser campos separados:

```json
{
  "teacher_reasoning_raw": "...",
  "training_target": {
    "visual_findings": [
      "well-demarcated erythematous plaque",
      "silvery scale"
    ],
    "clinical_description": "A sharply demarcated erythematous plaque with overlying scale.",
    "differential": [
      {
        "disease": "psoriasis",
        "rank": 1,
        "evidence": [
          "well-demarcated plaque",
          "silvery scale"
        ]
      },
      {
        "disease": "atopic dermatitis",
        "rank": 2,
        "evidence": [
          "erythema",
          "scale"
        ]
      }
    ],
    "final_diagnosis": "psoriasis"
  },
  "ground_truth": "psoriasis"
}
```

O campo `teacher_reasoning_raw` pode ser preservado para:

- auditoria e rastreabilidade;
- investigação qualitativa;
- análise de loops, contradições e alucinações;
- comparação entre thinking e non-thinking;
- documentação da tese.

Este campo deve ficar excluído do dataset de treino por omissão. Uma eventual
experiência com chain-of-thought bruto deve utilizar uma versão separada e
claramente identificada do dataset.

Também devem ser guardados:

- modelo, revisão e fornecedor do teacher;
- modo de thinking;
- parâmetros de geração;
- prompt e versão do schema;
- seed, quando suportada;
- resposta final original;
- estado de validação;
- razões de aceitação ou rejeição.

## 5. Target recomendado para o student

O student deve aprender uma sequência clínica verificável:

```text
Imagem
  -> achados visuais
  -> descrição clínica
  -> diagnóstico diferencial
  -> evidência associada
  -> diagnóstico final ou ação
```

Esta representação ensina raciocínio clínico explícito sem obrigar o student a
imitar um monólogo interno longo. Cada componente pode ser avaliado:

- o achado está visível na imagem;
- a terminologia morfológica está correta;
- a descrição não inventa história clínica;
- a evidência suporta o diagnóstico;
- o diagnóstico corresponde ao ground truth;
- a ação escolhida é adequada ao cenário.

Para a política interativa, o target pode ainda incluir uma ação entre:

- `CLASSIFY`;
- `ASK_CONTEXT`;
- `REQUEST_BETTER_IMAGE`;
- `ABSTAIN_OUT_OF_DOMAIN`.

## 6. Controlo de qualidade

Os rationales do teacher não devem ser tratados como ground truth. A aceitação
de cada exemplo sintético deve incluir:

1. validação do JSON e do schema;
2. correspondência do diagnóstico com o label conhecido;
3. validação dos conceitos morfológicos disponíveis;
4. deteção de afirmações que não podem ser inferidas da imagem;
5. deteção de contradições entre descrição, evidência e diagnóstico;
6. rejeição de respostas truncadas ou repetitivas;
7. deduplicação;
8. revisão humana de uma amostra estratificada.

Pode também ser usada consistência entre múltiplas gerações ou entre mais do
que um teacher, mas o consenso entre modelos não substitui validação clínica.

O dataset sintético deve derivar exclusivamente de casos do Train. Validation,
Internal Test, DDI externo e SkinDisNet externo não podem alimentar o
fine-tuning.

## 7. Ablations propostas para o student

Devem ser treinadas e comparadas pelo menos estas variantes:

| Variante | Supervisão |
| --- | --- |
| Baseline | Imagem e diagnóstico |
| Structured rationale | Imagem, descrição, achados, diferencial, evidência e diagnóstico |
| Raw reasoning | Imagem, chain-of-thought do teacher e diagnóstico |

A hipótese principal é que `Structured rationale` oferece o melhor compromisso:

- ensina informação clínica observável;
- permite explicações avaliáveis;
- reduz o tamanho dos targets;
- diminui a imitação de loops e afirmações não fundamentadas;
- facilita validação determinística e auditoria humana.

`Raw reasoning` deve ser considerado uma ablation de investigação e não a
configuração principal.

## 8. Evidência científica

O trabalho *Distilling Step-by-Step* mostrou que rationales produzidos por um
LLM podem funcionar como supervisão adicional e melhorar a eficiência de
dados no treino de modelos mais pequenos:

- [Hsieh et al., Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)

No entanto, melhor accuracy final não implica necessariamente melhor
factualidade do raciocínio. Um preprint recente sobre medical chain-of-thought
distillation encontrou casos em que as respostas melhoraram enquanto os erros
nos passos intermédios aumentaram:

- [Jiang et al., Better Accuracies, Worse Reasoning](https://arxiv.org/abs/2605.28301)

Este segundo resultado deve ser interpretado como evidência preliminar, por se
tratar de um preprint recente, mas suporta a necessidade de avaliar os
rationales separadamente das respostas finais.

## 9. Sequência recomendada

1. correr todos os modelos sem thinking;
2. selecionar os dois ou três melhores na Validation;
3. repetir esses candidatos com thinking;
4. medir qualidade, schema, custo, tokens, latência e loops;
5. escolher o modo do teacher com base nessa comparação;
6. gerar supervisão sintética apenas sobre Train;
7. guardar reasoning bruto apenas como artefacto de auditoria;
8. produzir targets clínicos estruturados e filtrados;
9. treinar as variantes baseline e structured rationale;
10. considerar raw reasoning apenas como ablation;
11. selecionar checkpoints na Validation;
12. executar uma única avaliação final nos conjuntos selados.

## 10. Recomendação

Thinking deve ser uma capacidade opcional do teacher, não um requisito do
student. A tese fica metodologicamente mais forte se demonstrar que um small
multimodal model aprende descrições, evidência e decisões clínicas verificáveis
em vez de apenas reproduzir longas cadeias de pensamento do teacher.

## 11. Validação prática dos endpoints em 30 de julho de 2026

O parâmetro oficial do Kimi K2.6 na API Moonshot,
`thinking: {type: disabled}`, foi rejeitado pelo deployment
Direct-from-Azure com `unrecognized_request_argument`. No mesmo deployment,
`reasoning_effort: none` foi aceite e produziu uma resposta completa com 470
output tokens, sem reasoning devolvido. A pipeline guarda esta diferença por
perfil: Moonshot/vLLM e Azure não devem ser tratados como contratos idênticos.

Nos smoke tests finais com seed 42:

| Modelo e benchmark | Resultado estrutural |
| --- | --- |
| Luna, evidence, JSON Schema | 10/10 JSON e schema; 8/10 semântica |
| Kimi, evidence, sem thinking | 3/10 JSON estrito; 9/10 recuperável |
| Luna, confusion, JSON Schema | 16/20 válidos; 4 recusas de segurança |
| Kimi, confusion, sem thinking | 4/20 JSON estrito; 20/20 recuperável |

O Kimi deixou de entrar em loops de thinking, mas continuou a envolver
frequentemente o JSON numa fence Markdown. Por isso, a pipeline distingue
`json_validity_rate` de `recoverable_json_validity_rate`: a recuperação é útil
na produção, mas não apaga o incumprimento do contrato prompt-only.

As quatro recusas do Luna correspondem a duas imagens, cada uma usada nas
condições low/high do benchmark emparelhado. O Azure devolveu código
`content_policy_violation` e request IDs, mas não devolveu categoria nem
severidade. Estes casos são agora `safety_refusal`, não `backend_error`, e
devem ser reportados separadamente. Não se deve remover globalmente os filtros
de segurança; deve primeiro testar-se uma configuração Azure menos restritiva
aprovada para contexto clínico e manter uma análise de sensibilidade com o
filtro original.
