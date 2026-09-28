# LGSA — From Small to Large: Cross-Scale Graph Domain Adaptation via Local-Global Structural Alignment

Official implementation of **LGSA** and reference implementation of the
**Cross-Scale Graph Domain Adaptation (CSGDA)** benchmark setting.

CSGDA adapts from a **small sampled source graph** (1% / 5% / 10% of the original
source nodes, with only the induced edges) to a **much larger, fully observed but
unlabelled target graph**. Sampling the source does not only remove labels — it
truncates local neighbourhoods and distorts global structural statistics, which is
why conventional GDA methods collapse towards random accuracy at `ρ = 1%`.

LGSA addresses both effects with two branches and an adaptive fusion module:

| Component | Role |
|---|---|
| **Local branch** (Sec. 3.3) | augments `G^S` with the top-`M` most similar target nodes, propagates source label signals over the augmented adjacency, and encodes it with a shared GCN |
| **Global branch** (Sec. 3.4) | encodes `[X ‖ Φ]` with an MLP — no message passing, so representation learning is decoupled from the incomplete source adjacency; mean-embedding alignment absorbs the sampling-induced first-moment shift in `Φ` |
| **Adaptive GCN–MLP alignment** (Sec. 3.5) | per-node entropy reliability drives a sigmoid gate `γ_v`; confident target pseudo-labels supervise the fused classifier and an InfoNCE term keeps the two branches consistent on the target side |

---

## Repository layout

```
LGSA/
├── README.md
├── requirements.txt
├── docs/
│   └── CROSS_SCALE_PROTOCOL.md      ★ normative spec of the CSGDA setting
├── lgsa/
│   ├── cross_scale.py               ★ the sampling operator S_ρ (self-contained)
│   ├── datasets.py                  benchmark loading + transfer-task registry
│   ├── augment.py                   cross-domain augmentation + label-signal propagation
│   ├── descriptors.py               structural descriptor φ(v) = [deg, c, n₂, n₃]
│   ├── models.py                    local GCN / global MLP / adaptive fusion
│   ├── losses.py                    Eq. 5, 7, 10, 12
│   ├── trainer.py                   Algorithm 1
│   ├── config.py                    hyper-parameters + per-task (μ, η) of Table 12
│   └── utils.py                     seeding, metrics, CSV I/O
└── scripts/
    ├── build_cross_scale_benchmark.py   ★ materialise the sampled source graphs
    ├── run_lgsa.py                      single transfer task
    ├── reproduce_main_results.py        Tables 2 and 3
    ├── run_ablation.py                  Figures 3, 6, 7 (LGSA1/2/3)
    └── analyse_discrepancy.py           Fig. 1(b) and Fig. 5 (Appendix A)
```

**If you only care about the setting**, you need exactly two files:
`lgsa/cross_scale.py` and `docs/CROSS_SCALE_PROTOCOL.md`.

---

## Installation

```bash
conda create -n lgsa python=3.10 -y
conda activate lgsa

# install torch first, matching your CUDA version (see pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric

pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x and PyTorch Geometric 2.5 on a single
NVIDIA RTX A5000 (24 GB). The code also runs on CPU for the smaller benchmarks.

---

## Data

Place each benchmark graph under a common root, following the standard
`torch_geometric.InMemoryDataset` convention:

```
<DATA_ROOT>/
├── ACMv9/processed/data.pt        ├── USA/processed/data.pt
├── Citationv1/processed/data.pt   ├── BRAZIL/processed/data.pt
├── DBLPv7/processed/data.pt       ├── EUROPE/processed/data.pt
├── Blog1/processed/data.pt        ├── ACM3/processed/data.pt
└── Blog2/processed/data.pt        └── ACM4/processed/data.pt
```

Each `data.pt` holds a PyG `Data` with `x`, `y` and `edge_index`. Point the code at
the root with `--data-root <DATA_ROOT>` or `export LGSA_DATA_ROOT=<DATA_ROOT>`.

| Family | Graphs | #Node | #Edge | #Label | Source |
|---|---|---|---|---|---|
| Airport | USA / Brazil / Europe | 1,190 / 131 / 399 | 13,599 / 1,038 / 5,995 | 4 | struc2vec |
| Citation | ACMv9 / Citationv1 / DBLPv7 | 9,360 / 8,935 / 5,484 | 15,556 / 15,098 / 8,117 | 5 | ArnetMiner (UDAGCN) |
| Social | Blog1 / Blog2 | 2,300 / 2,896 | 33,471 / 53,836 | 6 | BlogCatalog |
| ACM | ACM3 (P-S-P) / ACM4 (P-A-P) | 3,025 / 4,019 | 2,221,699 / 57,853 | 3 | One2Multi / MAGNN |

Source and target of a transfer task must share the input feature space; the
loader raises an explicit error otherwise.

---

## Quick start

```bash
export LGSA_DATA_ROOT=/path/to/data

# 1. inspect the cross-scale sampler on one graph
python -m lgsa.cross_scale --source ACMv9 --ratio 0.05 --seed 42

# 2. train LGSA on a single task
python scripts/run_lgsa.py --task A->D --ratio 0.05 --seeds 42 123 456

# 3. reproduce the main tables
python scripts/reproduce_main_results.py --family Citation
```

Task names use the abbreviations of the paper: `A`=ACMv9, `C`=Citationv1,
`D`=DBLPv7, `U`=USA, `B`=Brazil, `E`=Europe, `B1`/`B2`=Blog1/Blog2,
`A3`/`A4`=ACM3/ACM4.

---

## Reproducing the cross-scale benchmark

**Export the sampled source graphs once and share them.** This is the recommended
workflow, because it makes it impossible for two methods to accidentally train on
different source graphs:

```bash
python scripts/build_cross_scale_benchmark.py \
    --data-root $LGSA_DATA_ROOT \
    --out-dir  cross_scale_benchmark
```

Produces

```
cross_scale_benchmark/
├── manifest.csv               one row per (task, ρ, seed): budgets, K, #N, #E, class coverage
├── statistics_table1.csv      Table 1 / Table 4 of the paper
└── A_to_D/rho0.05/seed42.pt   node_idx, edge_index, y, sampler metadata
```

Load an exported source graph:

```python
import torch
from lgsa.cross_scale import induced_subgraph
from lgsa.datasets import load_graph

payload = torch.load("cross_scale_benchmark/A_to_D/rho0.05/seed42.pt")
source  = induced_subgraph(load_graph(payload["source"]), payload["node_idx"].tolist())
```

The sampler itself is specified in **[docs/CROSS_SCALE_PROTOCOL.md](docs/CROSS_SCALE_PROTOCOL.md)**;
read that document before comparing numbers.

---

## Reproducing the paper

| Result | Command |
|---|---|
| Tables 2 & 3 (main accuracy) | `python scripts/reproduce_main_results.py` |
| Table 1 & 4 (source statistics) | `python scripts/build_cross_scale_benchmark.py --stats-only` |
| Fig. 3, 6, 7 (ablation LGSA1/2/3) | `python scripts/run_ablation.py --family Airport` |
| Fig. 1(b), Fig. 5 (discrepancy) | `python scripts/analyse_discrepancy.py --family Citation` |
| Table 5/7 (BFS:DFS mixing `α`) | `python scripts/run_lgsa.py --task D->A --alpha 0.0 0.25 0.5 0.75 1.0` (one run per value) |
| Fig. 4 (μ, η sensitivity) | `python scripts/run_lgsa.py --task A->C --mu 0.05 --eta 0.1` over the 5×5 grid |

---

## Hyper-parameters

Defaults follow Appendix D:

| | value |
|---|---|
| hidden width (both branches, 2 layers) | 128 |
| projection dimension | 16 |
| dropout | 0.5 |
| optimiser | Adam, lr `5e-4`, weight decay `5e-4` |
| epochs | 200 |
| `M` (cross-domain neighbours) | 5 |
| `T` / `α` (label-signal propagation) | 3 / 0.5 |
| `τ_p` (pseudo-label threshold) | 0.99 |
| pseudo-label warm-up | 100 epochs |
| `τ_c` (contrastive temperature) | 0.5 |

The paper fixes only the architecture and the optimiser; `M`, `T`, `α`, `τ_p`,
`τ_c` and `E` are not reported, so the values above are choices of this
implementation.

Only two loss weights need tuning (Eq. 13):

```
L = L_fuse + μ·L_glo + (1−μ)·L_loc + η·L_cons
```

Both are searched over `{0.01, 0.05, 0.1, 0.5, 1}`; the response surfaces are
nearly flat inside `[0.05, 0.5]` (Sec. 4.3). Per-task values (Table 12) are
hard-coded in `lgsa/config.py` and applied automatically by the runner scripts;
`--mu` / `--eta` override them.

Three implementation choices that the paper leaves open are made explicit here:

* **InfoNCE negative pool.** `L_cons` is computed over a random subset of at most
  `consistency_max_nodes = 4096` target nodes per step, which bounds the
  `O(|V^T|²)` similarity matrix. Set it to `0` to use every target node.
* **Descriptor scaling.** `Φ` is log-compressed on its three count-valued columns
  and z-scored over the **union** of the two domains, so that the relative
  source/target shift stays visible to the alignment term of Eq. 7.
* **Pseudo-label schedule.** Algorithm 1 initialises `V^T_conf ← ∅` but does not
  say when it may become non-empty. `V^T_conf` is kept empty for the first
  `pseudo_warmup_epochs = 100` epochs and `τ_p` defaults to `0.99`, so that the
  target term of Eq. 10 only starts acting once the fused classifier has settled.

---

## Notes on evaluation

* Target labels are **never** used during training — not for early stopping, not
  for checkpoint selection. Reported numbers come from the final-epoch model.
* All methods in a comparison must use the same `(task, ρ, seed)` source graph.
* Results are averaged over sampling seeds and reported as `mean ± std`.

`train_lgsa` additionally returns `local_accuracy`, `global_accuracy` and
`mean_gate`, which reproduce the branch-wise decomposition of Appendix C.5.

---

## Citation

```bibtex
@inproceedings{lgsa2026,
  title     = {From Small to Large: Cross-Scale Graph Domain Adaptation via
               Local-Global Structural Alignment},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Released under the MIT License — see `LICENSE`.
