# The Cross-Scale (CSGDA) Benchmark Protocol

> **This document is the normative specification of the CSGDA setting.**
> If you want to compare against the numbers in *"From Small to Large: Cross-Scale
> Graph Domain Adaptation via Local-Global Structural Alignment"*, follow this
> document and use `lgsa/cross_scale.py` — nothing else in the repository is
> required.

---

## 1. Task definition

Let `G^S_full = {V^S_full, E^S_full, A^S_full, X^S_full, Y^S_full}` be a fully
labelled source graph and `G^T = {V^T, E^T, A^T, X^T}` an unlabelled target graph.

Conventional GDA trains on all of `G^S_full`. **Cross-Scale GDA (CSGDA)** replaces
it with a node-induced subgraph produced by a sampling operator `S_ρ`:

```
G^S = S_ρ(G^S_full) = {V^S, E^S, A^S, X^S, Y^S},     N^S = ⌊ρ · |V^S_full|⌋
```

with `ρ ∈ {1%, 5%, 10%}`. The regime of interest is `N^S ≪ N^T`.

| Available at training time | Not available |
|---|---|
| `G^S` (nodes, induced edges, features, **labels**) | any node of `G^S_full \ V^S` |
| `G^T` (nodes, edges, features) | any edge of `G^S_full` with an endpoint outside `V^S` |
| | **any target label** (`Y^T` is used for evaluation only) |

Evaluation: node classification accuracy (and macro-F1) on **all** of `V^T`.

Why this is harder than "fewer labels":

* **Local truncation.** Only edges induced on `V^S` survive, so the neighbourhood
  context that message passing relies on is destroyed, not merely thinned.
* **Global distortion.** Node-level structural statistics (degree, clustering,
  k-hop sizes) shift systematically downward on `G^S`, so any source-side
  distribution estimate is biased.

`scripts/analyse_discrepancy.py` quantifies both effects (Appendix A).

---

## 2. The sampling operator `S_ρ`

The budget is split 50:50 between local cohesion and global coverage:

```
N^S_BFS = ⌊α · N^S⌋,   N^S_DFS = N^S − N^S_BFS,   α = 0.5 (default)
```

### Stage 1 — BFS clusters (local cohesion)

1. Choose `K` seed nodes uniformly at random from `V^S_full`.
2. Each seed grows one BFS ball with budget `q_k = ⌈N^S_BFS / K⌉`. When a node is
   popped, its **unvisited neighbours are randomly shuffled** before being added.
3. If a cluster terminates early because its connected region is exhausted,
   restart BFS from another unsampled node until `N^S_BFS` is reached.

Number of clusters (Eq. 27):

```
K = ⌈ √(N^S_BFS) ⌉ = ⌈ √(ρ|V^S_full| / 2) ⌉
```

so both `K` and the per-cluster size scale as `Θ(√b)`. This is hyper-parameter
free and flat within a factor of two around the default (Appendix C.3).

### Stage 2 — DFS / random probes (global coverage)

Repeatedly pick a start node uniformly from `V^S_full \ V^S_BFS` and walk
depth-first, following one random **unvisited** neighbour at each step. When the
walk gets stuck, restart from another unsampled node. Stop at `N^S_DFS` nodes.

### Stage 3 — node union and induced subgraph

```
V^S = V^S_BFS ∪ V^S_DFS
E^S = {(i, j) ∈ E^S_full | i ∈ V^S and j ∈ V^S}
A^S = A^S_full[V^S, V^S],  X^S = X^S_full[V^S, :],  Y^S = Y^S_full[V^S, :]
```

No edge outside `V^S` is retained.

### Edge cases

| Case | Rule |
|---|---|
| `⌊ρN⌋ < 2` (e.g. Brazil at `ρ=1%`) | floor the budget to **2 nodes** so at least one candidate edge can exist |
| graph too fragmented to fill the budget | top up with uniformly random unsampled nodes |
| isolated nodes | kept; they are a genuine symptom of the setting and must not be dropped |

### What `S_ρ` deliberately does **not** do

* It does **not** rebalance classes. Class coverage `C_y` is an *outcome* of the
  sampler, and under-covered classes are part of the difficulty.
* It does **not** touch the target graph.
* It does **not** re-weight or re-normalise features.

---

## 3. Fair-comparison rules

1. **Same sampled source for every method.** Sample once per
   `(task, ρ, seed)` and feed the identical `G^S` to LGSA and every baseline.
   The easiest way to guarantee this is to export the benchmark to disk:

   ```bash
   python scripts/build_cross_scale_benchmark.py --data-root <DATA> --out-dir cross_scale_benchmark
   ```

2. **No privileged access.** No method may see `G^S_full \ V^S`, target labels,
   or the identity of the removed edges.

3. **No target-label model selection.** Target labels are unavailable in this
   setting, so checkpoints must not be selected by target accuracy. This
   implementation reports the **final-epoch** model.

4. **Averaging.** The paper samples the source graph 10 times and trains 5 times
   per sample, reporting the mean accuracy. A reduced protocol (e.g. 5 sampling
   seeds) is acceptable if stated explicitly.

5. **Report `(mean ± std)`** over seeds, and report which `ρ` values were used.

---

## 4. Reference implementation

```python
from lgsa.cross_scale import CrossScaleConfig, sample_cross_scale_source

sampled_source, info = sample_cross_scale_source(
    full_source_graph,                       # torch_geometric Data
    CrossScaleConfig(ratio=0.05, seed=42),   # rho, seed, alpha, K
)
```

Framework-agnostic core (no torch required) — copy this single function if you
work outside PyTorch Geometric:

```python
from lgsa.cross_scale import select_cross_scale_nodes

node_ids, info = select_cross_scale_nodes(
    edge_list=[(0, 1), (1, 2), ...],   # undirected (src, dst) pairs
    num_nodes=9360,
    ratio=0.05, seed=42, alpha=0.5,
)
```

Determinism: the output depends only on
`(num_nodes, edge_list, ratio, seed, alpha, K)`. Identical arguments produce
identical node sets on any machine.

---

## 5. Reference statistics

Mean source size over the sampling seeds (Table 1 and Table 4 of the paper).
Reproduce with `python scripts/build_cross_scale_benchmark.py --stats-only`.

| Source→Target | Full #N | Full #E | Target #N | Target #E | ρ=1% #N | ρ=5% #N | ρ=10% #N |
|---|---|---|---|---|---|---|---|
| D→A | 5,484 | 16,234 | 9,360 | 31,112 | 55 | 274 | 548 |
| A→D | 9,360 | 31,112 | 5,484 | 16,234 | 94 | 468 | 936 |
| C→D | 8,935 | 30,196 | 5,484 | 16,234 | 89 | 447 | 894 |
| B1→B2 | 2,300 | 66,942 | 2,896 | 107,672 | 23 | 115 | 230 |
| B2→B1 | 2,896 | 107,672 | 2,300 | 66,942 | 29 | 145 | 290 |
| U→B | 1,190 | 13,599 | 131 | 1,038 | 12 | 60 | 119 |
| B→U | 131 | 1,038 | 1,190 | 13,599 | 2 | 7 | 13 |
| E→U | 399 | 5,995 | 1,190 | 13,599 | 4 | 20 | 40 |

Even at `ρ=10%` the sampled source is one to two orders of magnitude smaller than
the target in both nodes and edges.

---

## 6. Sampler ablations

`α` (the BFS share) and `K` are exposed so that the Appendix C studies can be
reproduced:

```bash
# alpha sweep: 0 = pure DFS, 100 = pure BFS; the 50:50 plateau is in [25, 75]
for a in 0.0 0.25 0.5 0.75 1.0; do
  python scripts/run_lgsa.py --task D->A --ratio 0.05 --alpha $a --seeds 42 123 456 789 1024
done

# K sensitivity around the default of Eq. (27)
python -m lgsa.cross_scale --source ACMv9 --ratio 0.05 --num-bfs-clusters 1
```

Expected behaviour (Appendix C.5): the **local** GCN branch improves monotonically
with `α`, the **global** MLP branch degrades monotonically, and the two curves
cross near `α = 0.5` — which is precisely where the entropy fusion gate is most
informative and the fused accuracy peaks.
