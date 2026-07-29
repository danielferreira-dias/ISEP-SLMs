# Release do dataset Visual Top-K

Esta pasta contém a release congelada e protegida contra leakage do dataset
utilizado pelo benchmark dermatológico Visual Top-K. Os Parquets consumidos
pelos modelos estão separados dos relatórios de inspeção e dos metadados de
integridade.

## Estrutura

```text
visual_top_k_v1/
├── datasets/
│   ├── internal/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── internal_test.parquet
│   │   ├── internal_benchmark_1000.parquet
│   │   └── internal_test_reserve.parquet
│   └── external/
│       ├── external_ddi.parquet
│       └── external_skindisnet.parquet
├── reports/
│   ├── split_summary_v1.csv
│   ├── subgroup_summary_v1.csv
│   └── benchmark_1000_balance_v1.csv
└── release/
    ├── integrity_report_v1.yaml
    └── benchmark_release_v1.yaml
```

## Que dataset utilizar

| Ficheiro | Imagens | Grupos | Utilização |
| --- | ---: | ---: | --- |
| `datasets/internal/train.parquet` | 6.417 | 4.962 | Apenas fine-tuning |
| `datasets/internal/validation.parquet` | 1.683 | 1.063 | Escolha do teacher, prompt, checkpoint e limiares |
| `datasets/internal/internal_test.parquet` | 1.722 | 1.063 | Teste interno completo e selado |
| `datasets/internal/internal_benchmark_1000.parquet` | 1.000 | 1.000 | Comparação emparelhada principal antes/depois do treino |
| `datasets/internal/internal_test_reserve.parquet` | 63 | 63 | Imagem representativa de cada grupo fora do benchmark de 1.000 casos |
| `datasets/external/external_ddi.parquet` | 300 | 299 | Avaliação externa nas 8 classes suportadas |
| `datasets/external/external_skindisnet.parquet` | 1.365 | 333 | Avaliação externa nas 4 classes suportadas |

`internal_benchmark_1000.parquet` é um subconjunto estrito de
`internal_test.parquet` e não pode ser contabilizado como dados adicionais.
`internal_test_reserve.parquet` contém uma imagem representativa por cada
grupo restante. Em conjunto, benchmark e reserva reconstroem os 1.063 grupos
do teste interno, mas não as 1.722 imagens.

## Relatórios e metadados da release

- `reports/split_summary_v1.csv`: contagens de imagens e grupos por conjunto
  de avaliação, dataset de origem e doença.
- `reports/subgroup_summary_v1.csv`: suporte dos subgrupos demográficos.
- `reports/benchmark_1000_balance_v1.csv`: comparação dos 1.000 casos
  selecionados com a distribuição do teste interno, usando um caso por grupo.
- `release/integrity_report_v1.yaml`: verificações de leakage e integridade.
- `release/benchmark_release_v1.yaml`: caminhos e checksums SHA-256 de todas
  as dependências e artefactos da release congelada.

Os resultados externos devem permanecer separados dos internos porque DDI e
SkinDisNet abrangem menos classes e populações de origem diferentes.

## Reconstruir e validar

Executar o pipeline completo:

```bash
.venv/bin/python -m src.data_pipeline.pipeline
```

Validar a release congelada sem a reconstruir:

```bash
.venv/bin/python -m src.data_pipeline.splitting --validate-only
```

Os artefactos Parquet, CSV e YAML gerados não devem ser editados manualmente.
Para os alterar, deve-se modificar a configuração de origem e reconstruir a
release.
