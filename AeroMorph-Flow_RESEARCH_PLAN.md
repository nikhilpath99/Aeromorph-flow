# AeroMorph-Flow — Research & Implementation Plan

A retrieval-augmented model for predicting how morphing-airfoil flow **changes** under
gradual geometry deformation, with a rigorous out-of-distribution (OOD) evaluation protocol.

> Scope note: this is a planning document, written to be critical rather than promotional.
> Where a design choice is risky, it says so. Where prior work already does most of what
> you want, it names it.

---

## 0. Reality check up front

Before the plan, the three things that most affect whether this becomes a real paper:

1. **Your components are mostly not novel.** Geometry→Cl/Cd/Cp surrogates are mature.
   Conditioning predictions on retrieved/in-context examples is *In-Context Operator
   Networks* (ICON, arXiv 2304.07993 / PNAS 2023). Residual/delta prediction via retrieval
   of a similar physical state is *DeltaPhi* (arXiv 2406.09795) — read this first; it is
   your nearest competitor. Memory-augmented retrieval for PDE operators also exists
   (e.g. arXiv 2508.01211). Uncertainty/OOD for neural-operator surrogates is its own
   active subfield.

2. **The defensible novelty is narrow and specific**, built on three things:
   - Morphing trajectories as the data-generating object → learning a *local/tangent
     operator* `d(flow)/d(geometry)` along smooth homotopies, not "retrieve any similar
     state."
   - **Path-consistency** as a loss and an evaluation: integrating predicted deltas along a
     morph path should recover the endpoint and be approximately path-independent
     (a discrete conservative-field check). This is the strongest differentiator and I have
     not seen it used this way.
   - An honest, controlled study of **when retrieval helps** vs. a strong conditioned
     baseline, with retrieval distance doubling as a calibrated OOD signal.

3. **Two risks that can quietly kill it:**
   - XFOIL is not ground truth near stall / separation; build the cheap MVP on it but do not
     oversell separation prediction, and validate near-stall against RANS later.
   - Retrieval may not beat a well-conditioned non-retrieval model. That comparison is the
     experiment that decides whether you have a contribution — build that baseline *first*.

---

## 1. Critical novelty assessment

### What already exists

| Capability you proposed | Existing work that covers it |
|---|---|
| `geometry + AoA + Re → Cl, Cd, Cp` | Mature: CNN/GNN/cGAN/neural-field airfoil surrogates; some already test unseen AoA "for morphing" |
| Condition prediction on retrieved/in-context examples, no fine-tune | ICON (2304.07993), ICON-LM (2308.05061) |
| Retrieve similar state → predict the **residual/delta** | **DeltaPhi (2406.09795)** — closest prior art |
| Memory bank + similarity retrieval for PDE operators | Multi-operator few-shot w/ dynamic memory buffer (2508.01211) |
| Uncertainty / OOD for operator surrogates | Calibration-aware UQ (2602.11090), structure-aware epistemic UQ (2603.11052) |
| Full 2D RANS field surrogates | FNO/DeepONet, neural fields (Infinity), Transolver/GNOT/GINO |

### What could still be novel (the part to defend)

- **Morphing-conditioned local operator.** Target = directional derivative of flow w.r.t. a
  *specified* geometry morph direction Δg, learned from smooth paths. More structured than
  DeltaPhi's "delta vs. an arbitrary retrieved neighbor."
- **Path-consistency / conservative-field property.** Treat the learned `Δflow(g, Δg)` as a
  vector field over geometry space; test whether line integrals along different morph paths
  between the same endpoints agree, and whether summed micro-deltas reconstruct macro
  endpoints. Use the violation as both a regularizer and a metric.
- **A reproducible morphing-OOD benchmark** with a clean separation of OOD axes (family,
  camber/thickness, AoA, Re, path length/direction, near-stall) — most surrogate papers only
  report random splits.
- **Retrieval-distance as a calibrated OOD/uncertainty signal**, evaluated rather than
  assumed.

### Honest verdict

As "RAG for physics on airfoils," this is a strong GitHub project / workshop paper but **not**
a strong standalone conference/journal contribution. With the path-consistency angle and a
disciplined "when does retrieval help" study, it becomes defensible. Pick one of those two as
the headline claim and make the rest supporting.

---

## 2. Research problem statement

> **Given** an airfoil geometry `g`, an operating condition `c = (AoA, Re)`, and a small
> geometry morph `Δg`, **predict** the resulting change in aerodynamics
> `Δy = (ΔCp(x), ΔCl, ΔCd, ΔCm)`, and quantify predictive uncertainty, such that:
> (a) prediction error on Δy is lower than predicting absolute `y` and subtracting, and
> (b) error degrades gracefully and *measurably* under controlled distribution shift in
> geometry family, camber/thickness, AoA, Re, and morph magnitude.

Testable sub-hypotheses:

- **H1 (delta framing).** Predicting Δy directly beats `y(g+Δg) − y(g)` from an absolute
  surrogate, for small/medium morphs, measured by Δy error.
- **H2 (retrieval).** Adding retrieval over a morphing-experience memory reduces OOD error
  relative to an identical conditioned model without retrieval; the gain grows with OOD
  distance and is ~0 in-distribution.
- **H3 (path-consistency).** A model trained with a path-consistency penalty has lower
  endpoint-reconstruction error over multi-step morphs than one trained on single steps only.
- **H4 (uncertainty).** Retrieval distance and/or predicted variance are calibrated OOD
  signals (correlate with actual error; conformal intervals achieve nominal coverage).

Each hypothesis maps to a specific experiment and metric in §6 and §8. None of them contains
"the model learns physics."

---

## 3. Minimum viable research version (MVP)

Keep it boringly small so one person can finish it.

- **Geometry:** NACA 4-digit only. Sample camber `m ∈ [0, 6]%`, position `p ∈ [2, 6]` tenths,
  thickness `t ∈ [8, 18]%`.
- **Solver:** XFOIL only, fixed settings (panel count, `n_crit`, iteration cap), log
  convergence. No RANS yet.
- **Conditions:** a small AoA grid (e.g. −2° to 8° in 2° steps), 1–2 Reynolds numbers.
- **Morphs:** linear interpolation in `(m, p, t)` between random pairs, N steps per path.
- **Targets:** ΔCp on a fixed x/c grid (upper+lower, resampled to common abscissa),
  ΔCl, ΔCd, ΔCm.
- **Model:** the *non-retrieval delta model* first → `(enc(g), enc(c), enc(Δg)) → Δy` with a
  heteroscedastic head. **Do not build the memory bank yet.**
- **Deliverable:** show H1 (delta beats absolute-difference) and a clean ID/OOD error table.

Only after that works do you add retrieval (phase 4). If retrieval doesn't beat the MVP on
OOD, that negative result is itself a finding — report it.

---

## 4. Model architecture

```
                 ┌─────────────┐
   g_before ───▶ │  Geometry   │──┐
   Δg       ───▶ │  Encoder    │  │
                 └─────────────┘  │
                 ┌─────────────┐  ├─▶ query q ─▶ [Retrieval] ─▶ top-k memory values
   AoA, Re  ───▶ │  Condition  │──┘                                   │
                 │  Encoder    │                                      ▼
                 └─────────────┘                          ┌──────────────────────┐
                                                          │  Fusion / cross-attn  │
                                                          └──────────────────────┘
                                                                     │
                                                                     ▼
                                                        ┌──────────────────────────┐
                                                        │ Prediction head:         │
                                                        │  ΔCp(x), ΔCl, ΔCd, ΔCm   │
                                                        │  + per-output variance    │
                                                        └──────────────────────────┘
```

- **Geometry encoder.** For NACA-4/CST, the cleanest input is the *parameter vector*
  (low-variance, no resampling artifacts): `(m,p,t)` or CST coefficients → small MLP. Encode
  both `g_before` and `Δg` (the response depends on the operating point, so you need the base
  shape, not just the morph). Keep a surface-coordinate 1D-CNN / PointNet variant in reserve
  for when you move past parametric families.
- **Condition encoder.** `(AoA, log Re[, Mach])` → MLP. Normalize; `log Re` matters.
- **Query embedding.** `q = f(enc(g_before), enc(c), enc(Δg))`. Start with a learned encoder;
  it should be trained so "similar response behavior" → nearby embeddings (contrastive option
  below).
- **Retrieval/memory module.** kNN over memory keys (FAISS; cosine or L2). Top-k values fed
  to the predictor. Start with hard top-k; later try attention-weighted (soft) retrieval.
- **Fusion.** Cross-attention from query to retrieved memory values (ICON-style) or gated
  fusion (multi-operator paper style). Make this an ablatable switch so you can disable
  retrieval cleanly.
- **Prediction head.** Multi-output: ΔCp on the fixed grid (treat as a function — predict on
  the grid or as a few basis coefficients), plus scalar ΔCl/ΔCd/ΔCm. Heteroscedastic head
  predicts mean + log-variance per output.
- **Uncertainty / OOD.** Three cheap signals, evaluated against each other:
  (1) retrieval distance to nearest memory key; (2) predicted variance; (3) deep-ensemble or
  MC-dropout disagreement. Wrap with **split conformal prediction** for calibrated intervals.

---

## 5. Memory representation

### Per memory item (one morph step)

```
{
  id, path_id, step_index,
  geometry_before,        # CST/NACA params (and resampled surface coords)
  geometry_after,
  delta_geometry,         # parameterization-consistent (same basis as before/after)
  AoA, Re, Mach,
  Cp_before[x], Cp_after[x], delta_Cp[x],   # on a fixed shared x/c grid
  Cl_before, Cl_after, delta_Cl,
  Cd_before, Cd_after, delta_Cd,
  Cm_before, Cm_after, delta_Cm,
  xfoil_converged (bool), n_iter, transition_loc_upper, transition_loc_lower,
  separation_flag,        # from XFOIL BL solution; treat as weak/approximate
  solver_fidelity,        # 'xfoil' | 'rans'
  model_error,            # residual when this item was last predicted (filled online)
  uncertainty_score       # predicted σ at time of storage
}
```

Notes / pitfalls:
- **delta_geometry must be in the same basis** as before/after, or deltas across families are
  not comparable. This is why parametric families come first.
- **Cp correspondence.** Morphing moves the surface, so resample all Cp to a *fixed* x/c grid
  (and keep upper/lower separate) before differencing, or ΔCp is meaningless.
- **separation_flag from XFOIL is approximate** — store it, label it weak, don't headline it.

### Similarity search & embedding

- **Baseline embedding (start here):** normalized concat of
  `[CST coeffs of g_before, log Re, AoA, direction(Δg)]`, L2 / cosine kNN. No learning.
- **Learned embedding (next):** the query encoder, optionally trained contrastively so items
  with similar Δy are neighbors (positive pairs = steps with small Δy distance; negatives =
  large). This is what justifies retrieval over raw-feature kNN.
- **Index:** FAISS flat for the MVP (dataset is small), IVF/HNSW later.
- **Leakage guard:** at OOD test time the memory must contain *only* training items; never
  retrieve from the test split. On ID test, expect retrieval to be near-trivial — that's the
  point of contrasting ID vs OOD.

---

## 6. Datasets & experiments

### Generation

- NACA-4 (MVP) → add CST/PARSEC families later for true geometry-family OOD.
- Random airfoil pairs → linear interpolation paths (in param space) → N steps.
- AoA grid × Re grid per geometry. Log XFOIL convergence; keep non-converged points labeled
  (they're useful negatives and OOD markers, not garbage to silently drop).

### Splits (each isolates one shift axis)

1. **In-distribution (ID):** random split over all paths/conditions. Sanity ceiling.
2. **Geometry-family OOD:** hold out a region of shape space (e.g. high camber, or — once CST
   is in — an entire parameterization family) at train time; test on it.
3. **AoA OOD:** train AoA ≤ 8°, test 10–14° (approaching stall — flag XFOIL reliability).
4. **Reynolds OOD:** train at one Re, test at 0.3× and 3–6× that Re.
5. **Morphing-path generalization:** train on short morphs, test on longer morphs and/or
   unseen morph *directions*; plus a **path-composition** test (does summing micro-deltas
   recover the endpoint of a path it never saw whole?).
6. **Near-stall (stretch, RANS-validated):** only trust this once you have RANS labels.

---

## 7. Baselines

Ordered by how much they matter to the paper:

1. **Non-retrieval delta model** — identical encoders/head, retrieval disabled. *This is the
   baseline that decides whether retrieval is a contribution (H2).* Build it first.
2. **Absolute surrogate + subtraction** — `geometry+condition → y`, then `Δy = y₂ − y₁`.
   Tests H1.
3. **kNN-only delta** — predict the nearest neighbor's stored Δy, no learned predictor.
   Shows learning adds value beyond raw retrieval.
4. **Plain MLP** — vanilla regression, no delta framing, no retrieval. Floor.
5. **DeepONet / FNO** — relevant once you predict ΔCp as a function or move to RANS fields;
   not needed for the scalar-coefficient MVP. Include for the field-prediction phase.

Every model shares encoders, normalization, training budget, and the *same retrieval pool*
where applicable, so differences are attributable to the mechanism, not the plumbing.

---

## 8. Metrics

- **Cp error:** relative L2 over the x/c grid; plus error at the suction peak (where it
  matters for stall).
- **Cl/Cd/Cm error:** MAE, and Cd in **drag counts** (1 count = 1e-4) — reviewers expect this.
- **Delta-prediction error:** error on ΔCp / ΔCl / ΔCd specifically, vs. the
  absolute-then-subtract baseline (H1).
- **OOD degradation:** ratio `error_OOD / error_ID` per axis. Lower is better; report per
  split, not averaged into one number.
- **Uncertainty calibration:** regression calibration curve / expected calibration error,
  conformal interval **coverage** vs. nominal, and correlation(predicted σ, actual error).
- **Retrieval usefulness:** Δerror (with vs without retrieval) as a function of OOD distance;
  sensitivity to k; soft vs hard retrieval; learned vs raw-feature embedding (H2).
- **Path-consistency (the differentiator):** endpoint-reconstruction error from summed
  micro-deltas; path-independence gap between two morph routes with shared endpoints (H3).

---

## 9. Roadmap

- **Phase 1 — Data generation.** NACA-4 sampler, morph-path generator, XFOIL runner with
  fixed config + convergence logging, Cp resampling to a shared grid, memory-item serializer.
  *Exit:* a clean dataset + a few sanity plots (Cp curves, Cl–α, drag polars look physical).
- **Phase 2 — Simple surrogate.** Absolute `geometry+condition → y` model + ID/OOD harness.
  *Exit:* reproducible baseline numbers and the split machinery.
- **Phase 3 — Morphing delta model (the MVP core).** Non-retrieval `(g,c,Δg) → Δy` with
  heteroscedastic head. *Exit:* H1 result (delta beats absolute-difference) + OOD table.
- **Phase 4 — Memory retrieval.** Add FAISS memory, hard then soft retrieval, raw then learned
  embedding; ablate cleanly against Phase 3. *Exit:* H2 result (or an honest negative result).
- **Phase 5 — OOD + uncertainty + path-consistency.** Full OOD matrix, conformal calibration,
  path-consistency loss/metric (H3, H4). Optional: RANS labels for near-stall.
- **Phase 6 — Writeup.** Benchmark + ablations + the one headline claim. Release code, data
  generator, and pretrained memory bank.

Rough effort for a solo researcher: Phases 1–3 are the bulk of the value and are achievable;
Phases 4–5 are where the paper is won or lost; treat RANS as optional scope.

---

## 10. Title, abstract, contributions

**Working title:**
*AeroMorph-Flow: Retrieval-Augmented Prediction of Aerodynamic Change Under Airfoil Morphing,
and When Memory Actually Helps Under Distribution Shift*

**Abstract (draft):**
Data-driven airfoil surrogates predict absolute aerodynamics from a fixed geometry, but design
and control are inherently incremental: shapes morph, and what matters is how the flow
*changes*. We study the prediction of aerodynamic deltas — ΔCp, ΔCl, ΔCd, ΔCm — induced by
small geometry morphs, framed as estimating a local operator d(flow)/d(geometry) along smooth
deformation paths. We pair this delta formulation with a retrieval module over a memory bank of
past morphing experiences and ask a question prior retrieval-for-physics work usually skips:
does retrieval beat an equally-conditioned model without it, and where? On a controlled
benchmark spanning geometry-family, camber/thickness, angle-of-attack, Reynolds-number, and
morph-magnitude shifts, we find [delta framing reduces error vs. absolute-difference baselines],
[retrieval's benefit is concentrated in geometry-family OOD and negligible in-distribution], and
[retrieval distance is a calibrated out-of-distribution signal]. We further introduce a
path-consistency criterion — summed micro-deltas should reconstruct path endpoints and be
approximately route-independent — as both a regularizer and an evaluation. Code, the data
generator, and the memory bank are released.

> Fill the bracketed claims with whatever you actually measure. If retrieval *doesn't* help,
> say so — a clean negative result on "RAG for physics" is more valuable than an overclaim.

**Candidate contributions (claim only what you can defend):**
1. A delta/local-operator formulation for morphing-airfoil aerodynamics, with evidence it beats
   absolute-then-subtract surrogates on Δ-quantities.
2. A controlled, multi-axis OOD benchmark for morphing-conditioned aerodynamic prediction
   (family, camber/thickness, AoA, Re, morph magnitude), released with the data generator.
3. An empirical characterization of *when retrieval helps* — including the regime where it does
   not — with retrieval distance evaluated as an OOD/uncertainty signal.
4. A path-consistency criterion for learned local flow operators, used as both loss and metric.

**Do not claim:** that the model "learns physics," that this is the first retrieval-augmented
scientific ML, or strong near-stall/separation results from XFOIL data.

---

## Appendix: concrete first-week tasks

1. NACA-4 coordinate generator + morph-path interpolator (param space).
2. XFOIL batch runner (subprocess), fixed config, convergence + transition logging.
3. Cp resampling to a shared x/c grid; memory-item dataclass + serialization (parquet/zarr).
4. ID/OOD split functions (start with random + one geometry-family holdout).
5. Phase-2 absolute MLP to validate the whole pipeline end-to-end before any delta/retrieval.
