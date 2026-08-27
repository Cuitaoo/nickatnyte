# Boundary MRR Diagnosis

Date: 2026-08-27 (fable-model branch, Task 4)

## Question

`iteration2-comparison.md` reported boundary MRR falling `0.696111 -> 0.612778`
across the selective-evidence change while boundary Hit@10 held at 1.0. Was a
ranking defect introduced?

## Method

The evaluator is deterministic offline, so per-session ranks are exactly
reproducible. A scratch worktree was created at `7b32af3` (the last commit
before the selective-evidence implementation in this lineage — identical code
to the "Codex adaptive branch" row) and the full offline evaluation was rerun
there, then compared per boundary session against the current branch
(`docs/evaluations/fable-task3-offline.json`).

## Result: no regression against the real code ancestor

| Session | Ancestor (rank, hit turn) | Current (rank, hit turn) |
|---|---|---|
| public_0035 | (9, 8) | (9, 5) — earlier |
| public_0041 | (1, 5) | (1, 4) — earlier |
| public_0050 | (4, 1) | (4, 1) |
| public_0104 | (2, 3) | (2, 3) |
| public_0112 | (1, 4) | (1, 4) |
| public_0131 | (10, 2) | (10, 2) |
| public_0169 | (1, 10) | (1, 10) |
| public_0180 | (1, 9) | (5, 4) — 5 turns earlier, 4 ranks lower |
| public_0187 | (miss) | (1, 9) — new hit |
| public_0192 | (1, 3) | (1, 3) |

Boundary aggregate: Hit@10 `0.9 -> 1.0`, MRR `0.596111 -> 0.616111`,
MTTC improves. The only rank loss (public_0180) buys a five-turn-earlier hit,
which the technical score rewards through both MTTC and the earlier session
end.

## Why the comparison doc showed a drop

Rerunning `7b32af3` reproduces the "Codex adaptive branch" row of the
comparison table exactly (Hit@10 0.870, MRR 0.507536), not the "Verified
pre-change branch" row (0.875 / 0.502952 / boundary MRR 0.696111). The
pre-change row therefore came from a branch snapshot whose code is not any
ancestor of this branch. The apparent boundary-MRR regression was a
cross-branch comparison artifact, not a change introduced by the
selective-evidence work.

## Remaining boundary headroom (not defects)

`public_0131` (rank 10) and `public_0035` (rank 9) hit deep in the top 10;
these are in scope for the weight-tuning and reranking tasks, not evidence
handling.

## Reproduction

```bash
git worktree add --detach /tmp/prechange 7b32af3
cd /tmp/prechange
OPENAI_ENABLED=false python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl \
  --output /tmp/prechange-results.json
```

Then compare the `sessions` arrays of the two result files on
`scenario_type == "boundary"`.
