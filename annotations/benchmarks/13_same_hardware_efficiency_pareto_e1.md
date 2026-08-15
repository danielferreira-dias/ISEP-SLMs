# Eficiência no mesmo hardware: Qwen 3.8 27B versus Qwen 3.5 4B e E1

## Objetivo e desenho experimental

Esta experiência testa a hipótese central da tese sob uma perspetiva conjunta
de qualidade e custo computacional: um modelo pequeno especializado pode não
apenas superar um modelo maior em diagnóstico dermatológico, mas fazê-lo com
menos latência, memória e energia.

Foram avaliadas quatro condições, sequencialmente, na mesma **RTX PRO 6000
Blackwell Server**:

- Qwen 3.8 27B, generalista;
- Qwen 3.5 4B, generalista e base do student;
- Qwen 3.5 4B E1 Frozen Vision, especializado por label-only LoRA;
- Qwen 3.5 4B E1 Vision LoRA, especializado por label-only LoRA também nas
  camadas visuais.

O Qwen 3.6 27B foi deliberadamente excluído deste cohort por decisão de escopo.
Cada modelo recebeu os mesmos 400 casos congelados, 100 por tarefa, com
concorrência 1, BF16, temperatura zero, thinking desativado e o mesmo backend
vLLM. O manifest comum do cohort tem SHA-256
`37c4fe9913b3fbf4cfb33ab10834ae72833f1f3e61ab2dc224409c4986df4f7d`.

O modelo já estava carregado e aquecido antes da medição; os resultados não
incluem download, inicialização do servidor nem cold start. A potência da GPU
foi amostrada por NVML a cada 0,5 segundos e integrada pelo método trapezoidal.
A energia corrigida subtrai a potência idle medida para cada modelo. `RAM`
significa RSS máximo do processo servidor e não a memória total ocupada pelo
host.

## Resultado principal

| Modelo | Top-K Top-1 | Confusion Top-1 | Evidence Top-1 | Tok/s, média | VRAM máx. |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 27B | 47% | 78% | 40% | 26,5–26,6 | 86,20 GiB |
| Qwen 3.5 4B | 43% | 69% | 37% | 137,8–141,5 | 14,82 GiB |
| E1 Frozen Vision 4B | 60% | 76% | 46% | 140,3–142,8 | 14,27 GiB |
| **E1 Vision LoRA 4B** | **64%** | **79%** | **51%** | **143,1–143,6** | **14,68 GiB** |

No cohort fixo, o E1 Vision LoRA melhorou o modelo base em **+21 pp** no
Top-K, **+10 pp** em Confusion e **+14 pp** em Evidence. Também excedeu o
Qwen 3.8 27B em **+17 pp**, **+1 pp** e **+11 pp**, respetivamente.

O resultado mais relevante para a tese é que o E1 Vision LoRA domina o Qwen
3.8 27B nas três tarefas determinísticas: maior qualidade com menor latência,
menor energia e cerca de um sexto da VRAM. Isto é evidência de Pareto para este
cohort dermatológico, não uma declaração de superioridade geral do 4B.

## Latência, throughput e energia

| Tarefa | Latência p50: 27B | Latência p50: E1 Vision | Speed-up | Wh/query corrigido: 27B | Wh/query corrigido: E1 Vision | Redução energética |
|---|---:|---:|---:|---:|---:|---:|
| Visual Top-K | 6,770 s | 1,281 s | 5,28× | 0,5420 | 0,0856 | 6,33× |
| Confusion Sets | 3,642 s | 0,691 s | 5,27× | 0,2967 | 0,0462 | 6,42× |
| Evidence | 23,515 s | 3,736 s | 6,29× | 1,9193 | 0,2619 | 7,33× |
| Open-ended, sem score | 12,723 s | 1,290 s | 9,86× | 1,0235 | 0,0822 | 12,45× |

O throughput de geração do E1 Vision LoRA foi aproximadamente **5,4 vezes**
superior ao 27B. A VRAM máxima desceu de 86,20 para 14,68 GiB, uma redução de
aproximadamente **5,9 vezes**. A qualidade open-ended permanece em branco
neste cohort até existir julgamento externo; a sua latência e energia continuam
válidas.

## Gráficos e dados para a dissertação

Cada gráfico existe em PNG e SVG e possui um CSV com exatamente os pontos que
o originaram. Os artefactos estão em
`outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/`.

### Qualidade versus latência

![Qualidade versus latência](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/quality_vs_latency.png)

- SVG: `quality_vs_latency.svg`
- Dados: `quality_vs_latency_source.csv`

### Qualidade versus energia por resposta correta

![Qualidade versus energia por resposta correta](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/quality_vs_energy_per_correct.png)

- SVG: `quality_vs_energy_per_correct.svg`
- Dados: `quality_vs_energy_per_correct_source.csv`

### Throughput

![Throughput](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/throughput.png)

- SVG: `throughput.svg`
- Dados: `throughput_source.csv`

### Percentis de latência

![Percentis de latência](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/latency_percentiles.png)

- SVG: `latency_percentiles.svg`
- Dados: `latency_percentiles_source.csv`

### Time to first token

![Percentis de TTFT](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/ttft_percentiles.png)

- SVG: `ttft_percentiles.svg`
- Dados: `ttft_percentiles_source.csv`

### Tempo médio por token gerado

![Percentis de TPOT](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/tpot_percentiles.png)

- SVG: `tpot_percentiles.svg`
- Dados: `tpot_percentiles_source.csv`

### Memória GPU e RAM do servidor

![Memória](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/memory.png)

- SVG: `memory.svg`
- Dados: `memory_source.csv`

### Energia e GPU-seconds por request

![Recursos por request](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/resource_per_request.png)

- SVG: `resource_per_request.svg`
- Dados: `resource_per_request_source.csv`

### Utilização e potência média da GPU

![Utilização e potência média](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/gpu_utilization_power.png)

- SVG: `gpu_utilization_power.svg`
- Dados: `gpu_utilization_power_source.csv`

### Potência e temperatura máximas da GPU

![Telemetria máxima da GPU](../../outputs/efficiency_cohort_v1/comparison_q38_q35_e1/figures/gpu_peak_telemetry.png)

- SVG: `gpu_peak_telemetry.svg`
- Dados: `gpu_peak_telemetry_source.csv`

Estão ainda disponíveis os gráficos `quality_vs_vram` e
`quality_vs_parameters`, bem como a tabela completa em CSV, Parquet e LaTeX:

- `outputs/efficiency_cohort_v1/comparison_q38_q35_e1/tables/same_hardware_comparison.csv`;
- `outputs/efficiency_cohort_v1/comparison_q38_q35_e1/metrics/same_hardware_comparison.parquet`;
- `outputs/efficiency_cohort_v1/comparison_q38_q35_e1/tables/same_hardware_comparison.tex`;
- `outputs/efficiency_cohort_v1/comparison_q38_q35_e1/tables/pareto_frontiers.csv`.

## Limitações e interpretação

- São 100 casos por tarefa e uma única execução por modelo. As diferenças de
  qualidade deste cohort devem ser confirmadas pelos benchmarks completos e
  por mais seeds de treino.
- Qwen 3.8 e Qwen 3.5 diferem em arquitetura, geração e pós-treino; esta não é
  uma experiência causal apenas sobre o número de parâmetros.
- A energia é board power medida por NVML, não consumo elétrico total da
  máquina ou do datacenter.
- O custo monetário não foi estimado porque não foi congelado um preço por
  hora do hardware. Inventar um preço tornaria a comparação temporalmente
  instável.
- APIs externas não podem entrar nesta comparação de energia no mesmo
  hardware; poderão ser comparadas separadamente por latência observada,
  tokens e custo faturado.
- A accuracy não substitui grounding clínico. O E1 label-only melhorou a
  classificação, mas as avaliações estruturadas anteriores mostram perdas de
  schema/concept compliance, motivando E2 structured e E3 hard-KD.

## Decisão experimental

O E1 Vision LoRA é a condição E1 recomendada para continuar como student na
fase estruturada. E1 Frozen permanece um controlo indispensável: apresenta
recursos quase idênticos e permite atribuir o ganho adicional à adaptação
visual. A próxima comparação deve voltar a colocar o student especializado
contra modelos maiores e reportar simultaneamente qualidade, latência,
throughput, VRAM e energia por resposta correta.
