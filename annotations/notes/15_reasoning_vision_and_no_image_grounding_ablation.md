# Reasoning, visão e ablação de grounding sem imagem

## Decisão

O estudo de *reasoning* não deve comparar apenas a accuracy com *thinking off*
e *thinking on*. Um modelo pode acertar recorrendo ao prior textual, enquanto
deixa de observar a imagem. Foi por isso adicionada ao ISEPDermaBench uma
ablação de Validation com 50 controlos sem evidência visual dermatológica.

A conclusão metodológica é:

> Um bom teacher deve combinar diagnóstico correto em imagens reais com
> perceção fiel, abstention perante imagens não avaliáveis e baixa alucinação.
> Mais tokens de reasoning, isoladamente, não demonstram melhor raciocínio
> visual.

Esta tarefa é de desenvolvimento e reforça o estudo do reasoning. Não é uma
nova benchmark final, não mede accuracy clínica e não deve ser executada no
Internal Benchmark.

## O que a literatura mostra

### Estudos que encontram benefícios

- **Multimodal-CoT** separa a geração da rationale da inferência da resposta
  num processo de duas etapas. O estudo reporta ganhos em ScienceQA e A-OKVQA
  e, para o modelo avaliado, menor alucinação e melhor convergência. Isto apoia
  a ideia de separar primeiro a observação visual e depois a conclusão
  diagnóstica ([Zhang et al., TMLR 2024](https://arxiv.org/abs/2302.00923)).
- **LLaVA-CoT** estrutura a geração em resumo, interpretação visual, raciocínio
  e conclusão; usa 100 mil anotações de reasoning e reporta uma melhoria média
  de 9,4% sobre o modelo-base em benchmarks multimodais de raciocínio. O ganho
  refere-se a tarefas reasoning-intensive, não prova que CoT melhore perceção
  dermatológica fina ([Xu et al., ICCV 2025](https://arxiv.org/abs/2411.10440)).
- **Grounded CoT** liga passos de raciocínio a coordenadas visuais e avalia
  separadamente answer accuracy, grounding accuracy e consistência entre
  resposta e grounding. Os autores mostram que maior dimensão do modelo não
  implica menor alucinação visual. A implicação para esta tese é não usar
  accuracy como único indicador de grounding
  ([Wu et al., 2025, preprint](https://arxiv.org/abs/2503.12799)).
- **V2T-CoT** e **ClinCoT** tentam tornar o reasoning médico dependente de
  regiões patológicas localizadas, em vez de produzir apenas texto plausível.
  Ambos reportam ganhos em Med-VQA ou report generation, mas são preprints e
  não estudam dermatologia clínica
  ([Wang et al., 2025](https://arxiv.org/abs/2506.19610);
  [Liu et al., 2026](https://arxiv.org/abs/2603.01124)).
- **CheXthought** recolhe 103.592 traces e mais de 6,6 milhões de anotações de
  atenção visual de 501 radiologistas. Os autores reportam que reasoning humano
  é mais factual e espacialmente grounded que CoT produzido por VLMs e que
  pistas de atenção visual reduzem alucinações. Embora seja radiologia e ainda
  um preprint, é evidência forte a favor de ensinar *onde observar* juntamente
  com *o que concluir*
  ([Sharma et al., 2026](https://arxiv.org/abs/2604.26288)).
- **ConceptVLM** usa conceitos clínicos e atenção mascarada para concentrar o
  treino na evidência relevante e reporta bons resultados médicos com uma
  fração pequena dos dados. É um artigo revisto por pares, mas testa outras
  modalidades e uma intervenção de treino, não uma simples prompt de CoT
  ([Li et al., npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02676-5)).
- **SkinGPT-R1** é particularmente próximo desta tese: propõe um corpus DermCoT
  com narrativas dermatológicas, distilação visual e avaliação em seis
  dimensões: accuracy, safety, medical groundedness, clinical coverage,
  reasoning coherence e description precision. Contudo, é um preprint; parte
  das narrativas e da filtragem depende de teachers/evaluators automáticos e os
  resultados não devem ser tratados como validação independente de que o CoT
  é fiel
  ([Shen et al., 2025](https://arxiv.org/abs/2511.15242)).

### Estudos que encontram riscos

- **Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of
  Multimodal LLMs** avalia 16 modelos em 13 benchmarks espaciais. O trabalho
  conclui que CoT degrada sistematicamente este tipo de raciocínio e usa
  No-Image++ para revelar atalhos e detalhes visuais inventados quando a imagem
  está ausente. É a principal inspiração direta desta ablação
  ([Kancheti et al., ACL 2026](https://aclanthology.org/2026.acl-short.71/)).
- **Look Light, Think Heavy** compara 12 tarefas, 14 modelos sem reasoning e
  oito modelos de reasoning. CoT prejudica tarefas perceptuais como grounding
  e counting, mas ajuda matemática, ciência e raciocínio multi-imagem. Durante
  cadeias longas, a reflexão verbal persiste enquanto a reflexão visual
  diminui
  ([Jin et al., ACL 2026](https://aclanthology.org/2026.acl-long.387/)).
- **More Thinking, Less Seeing?** associa reasoning mais longo a menor atenção
  visual, maior dependência de priors linguísticos e mais alucinação. O artigo
  propõe RH-AUC para estudar accuracy percetual em função do comprimento da
  cadeia e recomenda avaliar conjuntamente reasoning e fidelidade visual
  ([Liu et al., 2025, preprint](https://arxiv.org/abs/2505.21523)).
- **MIRAGE** separa alucinação originada por erros de perceção da alucinação
  originada pelo raciocínio e combina accuracy, factuality e um hallucination
  score. Esta decomposição apoia a separação entre métricas diagnósticas e
  métricas de evidência no ISEPDermaBench
  ([Dong et al., 2025, preprint](https://arxiv.org/abs/2505.24238)).
- **Seeing Through the Chain** encontra maior dependência de priors linguísticos
  quando são introduzidos mecanismos de reasoning e propõe comprimir CoT e usar
  preference optimization. Isto sugere que traces sintéticos longos não devem
  ser aceites automaticamente como melhores dados de distilação
  ([Fang et al., 2026, preprint](https://arxiv.org/abs/2602.03380)).
- **Hallucination-as-Cue** mostra que reinforcement post-training pode melhorar
  resultados mesmo quando informação visual crítica é removida ou substituída.
  Logo, uma subida de accuracy após treino não prova que o modelo aprendeu a
  usar a imagem
  ([Zhang et al., CVPR 2026](https://arxiv.org/abs/2604.03179)).
- **Understanding and Mitigating Hallucinations in Multimodal CoT Models**
  localiza muita fabricação nos passos de associação ou “divergent thinking”
  e mostra que intervir nesses passos reduz hallucination. Para a tese, reforça
  que o comprimento da cadeia não é o objetivo; interessa a ligação explícita
  entre cada conclusão e pixels observáveis
  ([Ma et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Understanding_and_Mitigating_Hallucinations_in_Multimodal_Chain-of-Thought_Models_CVPR_2026_paper.html)).
- **MiCEval** avalia separadamente a descrição da imagem e cada passo de
  reasoning, em vez de avaliar apenas a resposta final. Isto sustenta o desenho
  já usado na tese: descrição/achados, evidência e diagnóstico têm métricas
  diferentes
  ([Kang et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.504/)).

## Desenho dos 50 casos

Os controlos estão em:

```text
data/benchmarks/ISEPDermaBench/
├── tasks/visual_grounding_no_image/validation-00000-of-00001.parquet
├── references/visual_grounding_no_image/validation-00000-of-00001.parquet
├── artifacts/configs/visual_grounding_no_image.yaml
├── artifacts/prompts/visual_grounding_no_image.yaml
├── artifacts/schemas/visual_grounding_no_image.schema.json
└── metadata/visual_grounding_no_image_v1/
```

Propriedades congeladas:

- 50 casos e 50 leakage groups distintos;
- as mesmas identidades de caso provêm do subset fixo de 100 Visual Top-K;
- cobertura das 21 classes, com pelo menos dois grupos por classe;
- cada imagem original é substituída por JPEG RGB cinzento uniforme;
- largura e altura originais são preservadas para não introduzir um artefacto
  trivial de dimensão;
- a taxonomy completa continua na prompt para testar se o modelo escolhe uma
  doença apenas por prior textual;
- o diagnóstico original existe apenas nas referências e não é um target
  válido para a imagem cinzenta.

A resposta correta é:

```json
{
  "image_status": "not_evaluable",
  "visual_findings": [],
  "predictions": [],
  "confidence": "low"
}
```

## Experiência emparelhada

Para modelos cujo thinking possa ser controlado, o desenho ideal é 2 × 2:

| Imagem | Thinking | Pergunta respondida |
| --- | --- | --- |
| real | off | desempenho visual base |
| real | on | reasoning melhora o diagnóstico/grounding? |
| cinzenta | off | o modelo sabe abster-se sem reasoning? |
| cinzenta | on | reasoning aumenta ou reduz alucinação? |

Devem ser mantidos iguais o modelo, provider, prompt, IDs, ordem, seed,
sampling e limite final; a única diferença dentro de cada comparação é o
thinking. Quando o provider não permite desativá-lo, o resultado não deve ser
apresentado como um A/B causal.

Hipóteses:

1. Thinking pode melhorar accuracy em imagens reais sem melhorar perceção.
2. Thinking longo pode aumentar achados e diagnósticos inventados no controlo.
3. Um teacher grounded precisa simultaneamente de bom desempenho nas imagens
   reais e elevada abstention nas imagens cinzentas.
4. Comprimento do reasoning pode ser correlacionado descritivamente com
   alucinação, mas não prova causalidade quando os providers reportam tokens de
   forma diferente.

## Métricas

Métrica primária:

- `correct_abstention_rate`.

Métricas de falha visual:

- `hallucinated_visual_finding_rate`;
- `hallucinated_diagnosis_rate`;
- `unsupported_clinical_assertion_rate`;
- `overconfidence_rate`;
- `full_visual_grounding_compliance_rate`.

Métricas de contrato:

- `json_validity_rate`;
- `recoverable_json_validity_rate`;
- `schema_compliance_rate`;
- `semantic_compliance_rate`.

Auditorias secundárias:

- match Top-1 com o diagnóstico original oculto, global e condicionado a o
  modelo ter inventado um diagnóstico;
- disponibilidade, caracteres e tokens de reasoning reportados pelo provider.

O match com o diagnóstico original é apenas um sinal de shortcut. Não é
accuracy, porque não há lesão observável na imagem de controlo. A ausência de
reasoning retornado por um provider também não é uma falha do modelo e não
deve diminuir a pontuação.

Com apenas 50 casos, devem ser publicados contagens, taxas e intervalos de
confiança exatos/Wilson. Diferenças thinking off/on são emparelhadas e devem
usar McNemar ou bootstrap emparelhado. Este estudo é exploratório e não tem
poder para estabelecer segurança clínica.

Limitações específicas do controlo:

- uma imagem cinzenta uniforme é mais fácil de reconhecer como inválida do que
  uma fotografia real desfocada, subexposta, ocluída ou sem lesão relevante;
- preservar as dimensões controla o transporte, mas deixa um possível prior de
  resolução/source; o match com o diagnóstico oculto serve também para auditar
  esse risco residual;
- os 50 casos foram escolhidos para cobertura e emparelhamento, não para
  estimar prevalência;
- o teste deteta um tipo de falta de grounding, mas não prova que uma rationale
  produzida para uma imagem real seja causal ou clinicamente fiel.

## Comandos

Validar os 50 casos sem inferência:

```bash
python -m src.data_pipeline.visual_grounding_no_image --validate-only
```

Executar thinking off/on:

```bash
uv run python -m src.benchmark.cli run \
  --model <model_id> \
  --benchmark visual_grounding_no_image \
  --evaluation-set validation \
  --thinking-mode disabled \
  --output-root outputs/visual_grounding_no_image/thinking_off

uv run python -m src.benchmark.cli run \
  --model <model_id> \
  --benchmark visual_grounding_no_image \
  --evaluation-set validation \
  --thinking-mode enabled \
  --max-output-tokens 14336 \
  --output-root outputs/visual_grounding_no_image/thinking_on
```

## Implicação para o dataset sintético

O teacher não deve gerar apenas uma cadeia livre e longa. A opção mais
defensável para uma fase posterior é manter duas camadas distinguíveis:

1. **evidência visual estruturada**: o que é realmente observável, incluindo
   qualidade/evaluabilidade;
2. **inferência clínica**: diagnóstico e diferenciais ligados apenas aos
   achados da primeira camada.

Casos sem evidência suficiente devem ensinar abstention ou pedido de melhor
imagem/contexto, nunca uma explicação plausível inventada. Antes de usar traces
como targets de SFT, devem ser filtrados por correção diagnóstica, consistência
com a imagem, ausência de afirmações não suportadas e calibração. Reasoning
privado de APIs pode ser guardado para auditoria quando disponível, mas não
deve ser tratado automaticamente como verdade clínica nem como supervisão
obrigatória do student.
