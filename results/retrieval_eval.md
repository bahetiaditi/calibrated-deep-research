# Retrieval evaluation (R1-R4)

30 queries with relevant judgments, 997 total judgments.

```
     configuration                 nDCG@10    rec@20   rec@20!     MRR
----------------------------------------------------------------------
R1   Dense only                     0.6092    0.6331    0.6652  0.7259
R2   BM25 only                      0.6128    0.6569    0.7166  0.7457
R3   RRF fusion                     0.6459    0.6651    0.7095  0.7777
R4   RRF + cross-encoder rerank     0.6675    0.6651    0.7095  0.8128

rec@20   counts grades 1 and 2 — saturates when the pool is small relative to k
rec@20!  counts grade 2 only — the discriminating one on this label set
```

All expected orderings hold.
