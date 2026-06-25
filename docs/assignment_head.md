# OT Assignment Head — Graphical Explanation

Source:
- [assignment_head.py](../rtdetrv2_pytorch/src/zoo/rtdetr/assignment_head.py)
- [assignment_postprocessor.py](../rtdetrv2_pytorch/src/zoo/rtdetr/assignment_postprocessor.py)

## The problem it solves

RT-DETR predicts each of the `N` decoder queries **independently**. For tooth detection that's bad: the same FDI tooth number can land on several boxes, and nothing decides which box wins a class. The assignment head adds a **global** constraint — *each tooth class is used at most once* — in a fully differentiable way.

## The pipeline

```
 RT-DETR decoder output (per query, independent)
 ┌──────────────────────────────────────────────────────────┐
 │  pred_embed  [B,N,256]   pred_boxes [B,N,4]  pred_logits  │
 │     (emb_i)                  (box_i)          [B,N,C]     │
 └──────────────────────────────────────────────────────────┘
        │              │                 │
        └──────────────┼─────────────────┘
                       ▼
   ① TOKENIZE:  concat → [emb ‖ box ‖ logits]   in_dim = 256+4+C
                       │
              input_proj (Linear → dim) + LayerNorm
                       │   f_i  [B,N,dim]
                       │
        ┌──────────────┘   (+)  box_pos(box)   ← explicit positional encoding
        │                                        of 4 box coords; position is the
        │                                        dominant cue for tooth identity
        ▼
   ② CONTEXT (TransformerEncoder, 3 layers, self-attention)
        every query attends to every other query
                       │   g_i  [B,N,dim]   ← context-aware features
                       ▼
   ③ SCORE against learnable per-class prototypes
        scores =  g · class_protoᵀ / temp        (semantic match)
                + α · logits                     (RT-DETR's own prior)
                + β · box_pos · pos_protoᵀ / temp (geometry-vs-class prior)
                       │
                       ▼  cost matrix  S  [B, N, C]
```

### The cost matrix S

```
            class 0   class 1   ...   class C-1
          ┌───────────────────────────────────┐
 query 0  │  s00       s01      ...    s0,C-1  │
 query 1  │  s10       s11      ...            │
   ...    │   .                                │
 query N-1│  ...                       sN-1,C-1│
          └───────────────────────────────────┘
```

## ④ + ⑤ Dustbin augmentation + Sinkhorn

The matrix is padded with a **learnable dustbin** row and column (single scalar `bin_score`), then **Sinkhorn**-normalized in log space → a doubly-stochastic soft assignment `P`.

```
           c0   c1  ...  cN-1 │ DUSTBIN(absent class)
         ┌─────────────────────┼──────────┐
   q0    │                     │   ·       │  each REAL row sums to 1
   q1    │     real cells      │   ·       │
   ...   │       S             │   ·       │  ← query → dustbin column =
   qN-1  │                     │   ·       │     "this box is background/dropped"
         ├─────────────────────┼──────────┤
 DUSTBIN │  ·    ·   ·    ·     │  alpha    │
 (drop   └─────────────────────┴──────────┘
  query)   each REAL column sums to 1
           ↑ class → dustbin row = "this tooth is absent"
```

`log_optimal_transport` builds the `[B,N+1,C+1]` matrix; `log_sinkhorn_iterations` runs 50 iterations of alternating row/column log-normalization. The **column constraint *is* the "each class at most once" rule.** Output stored as `outputs['assignment_log']`.

## ⑥ Inference — the post-processor

Reads `assignment_log` and extracts at most one box per class:

```
 P = exp(log_P[:N,:C])
        │
        ├─ Hungarian (linear_sum_assignment on −log P)  → one query per class
        │   (or argmax fallback)
        ▼
 (q_idx, c_idx) pairs
        │
        ▼  DUSTBIN GATE — keep pair only if the real cell beats BOTH dustbins:
            real ≥ log_P[q, DUSTBIN_col]   (better than being a dropped box)
          AND real ≥ log_P[DUSTBIN_row, c]  (better than the tooth being absent)
        │
        ▼
 final: labels, boxes, scores   (≤ one per class, missing teeth correctly absent)
```

The dustbin gate is what lets missing teeth stay missing — without it, Hungarian is forced to assign *every* class, turning absent teeth into guaranteed false positives.

## Key design choices (the three cost terms)

| Term | Source | Why |
|------|--------|-----|
| `g · class_protoᵀ` | context features vs learned class prototype | semantic identity after queries see each other |
| `α · logits` | RT-DETR's own predictions | head **corrects** rather than relearns detection |
| `β · box_pos · pos_protoᵀ` | box position vs learned canonical tooth location | explicit anatomical-position prior (dominant cue for FDI tooth ID) |

All temperatures, `α`, `β`, `bin_score`, prototypes, and the encoder are learnable. Sinkhorn runs in fp32 even under AMP for stability.

Reference for the matching layer: SuperGlue (Sarlin et al., CVPR 2020).
