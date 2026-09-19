# Raw data (not tracked in git)

This project uses variant **V3** of the MNIST subset supplied for the course.
The raw CSV files are large (~110 MB + ~18 MB) and are **not** committed.

Place the following two files here:

| File | Rows | Description |
|---|---|---|
| `d3.csv` | 60 000 | Training variant V3: `label`, `1x1` … `28x28` (785 columns), pixel values 0–255 (`int`). |
| `mnist_test.csv` | 10 000 | Standard MNIST test split, same format; single shared test set for all experiments. |

The fastest way to populate this directory:

```bash
make data   # copies both files from ../materials/data/
```

The path is configurable via `data.raw_dir` in the YAML config (default `data/raw/`).
See `docs/data_source.md` for the full format description and provenance.
