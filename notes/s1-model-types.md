# Other S1 model types (field notes)

S1 = any input -> one pass -> bounded typed answer with probabilities. Defined by output type.
Clef: pick from named options. Clairic: read values out (spans, numbers, dates, lists, ranges, absent).

## Missing output types

| Type | Question | Signal |
|---|---|---|
| Subset | which of these apply | Clef SATA-Bench 33.8 |
| Ranking | order these by X | rerankers score one at a time |
| Pairwise preference | A or B by criterion given per call | reward models fix the criterion |
| Locate | which line/step/item in the input | new |
| Calibrated quantity | p10/p50/p90 of a number | new |
| Match | pair items in A with items in B | new |
| Relations | typed (subject, relation, object) | Clef ACOS 33 |
| Change | two states -> what changed | new |
| State over turns | running typed state | Clef SGD 43.8 |
| Point/box | where on the image | not in schema API form |

## New inputs

audio (end of turn, intent without transcript), time series, tables (in-context), code facts, traces,
network data, video.

## Agent-loop roles

continue/stop/ask, router, context pruner, memory gate, action admission, abstain/out-of-scope,
instruction source (user vs data).

## Gaps in Clef's own scores

- Hard reasoning (GPQA 48, BBH 74): "think a little, then decide".
- Subsets, relations, state over turns (above).
- Narrow domains (POP909 16, ChessBench 25): specialist S1s.
- Out-of-scope needs size (CLINC: 97 vs 67 for flash): a tiny dedicated OOS model.

## Build patterns

S1 + exact circuit (Handwired); compiled cascades (Clairic -> Clef -> code); self-checking S1; small students
taught by Clef.

## Top three

1. Ranking + pairwise preference with per-call criteria.
2. Context pruner for agents.
3. Locate + change.
