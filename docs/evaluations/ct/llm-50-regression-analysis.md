# Qwen 50-Sample Regression Analysis

## Scope

This compares the same stratified 50-session set under:

- offline deterministic preference parsing; and
- deterministic Qwen state updates (`qwen3:4b-q4_K_M`, temperature 0,
  reasoning disabled), with the no-preference evidence gate enabled.

Vector retrieval and the local cross-encoder were enabled in both runs. The
online candidate reranker was disabled. All 17 regressed sessions were replayed
turn by turn. Every replayed turn successfully used the LLM, so none of these
regressions came from a timeout, malformed call, or deterministic fallback.

## Main Finding

The dominant problem is a contract mismatch between the new state updater and
the existing ranker. The deterministic parser and retrieval weights were tuned
together. Qwen often captures the shopper's meaning correctly, but places the
same evidence in a different field:

- structured `feature` versus unstructured `search_terms`;
- one compound material versus all detected component materials;
- concise category phrase versus duplicated lexical evidence; or
- no duplicate search term when the value is already a preference.

Those fields do not have equal ranking behavior. Structured attributes receive
confirmed-preference boosts and feed route-specific retrieval. Search terms can
also enter the exact-phrase and relaxed routes. Therefore, semantically similar
states can produce different candidate pools, clarification questions, and
cross-encoder inputs.

## Per-Session Diagnosis

| Sample | Baseline -> LLM | What Qwen did that hurt |
|---|---|---|
| `public_0002` | T5 R4 -> T7 R5 | Shortened `Accessories Belts` to `belts`, omitted `100% Leather`, and stored `Imported` as a search term instead of a feature. The weaker evidence encoding kept the target out until two more no-preference turns. |
| `public_0003` | T5 R1 -> T8 R1 | Treated `ignore my earlier preference ... Water Resistant` as `merge`, preserving the stale `Stainless Steel Band` material. It should have replaced the earlier preference. The stale constraint also caused low-value brand/size/use-case/style questions before `other` revealed battery and date details. |
| `public_0007` | T2 R1 -> T4 R2 | Extracted only `polyester` from `75% Polyester, 20% Rayon, 5% Spandex`; the fallback extracted all three materials. This changed candidate disagreement, asked color before feature, and delayed `Imported; Pull On closure`. |
| `public_0008` | T2 R5 -> T2 R6 | Reduced `Bras Everyday Bras` to category `bras` and moved `everyday bras` into search terms. The constraints were otherwise correct, but the changed category/exact-phrase routing moved the target down one rank. |
| `public_0016` | T8 R1 -> T9 R1 | Stored `Rubber sole` as a search term rather than a confirmed feature. This changed the question policy so it asked `style` before `other`, delaying the distinctive shaft measurement by one turn. |
| `public_0018` | T1 R3 -> T6 R3 | Kept only concise category/material evidence and stored `Pull On closure` as a search term rather than a feature. The fallback duplicated more lexical evidence. The target dropped from the early candidate list and reappeared only after several no-preference messages changed the latest-message route. |
| `public_0020` | T1 R6 -> T4 R4 | The extracted category and material were valid, but Qwen's compact search-term representation produced a different candidate pool. It needed color and `Imported` before recall recovered. Rank improved, but the three-turn delay cost more than the rank gain. |
| `public_0021` | T2 R1 -> T4 R1 | Extracted `polyester` but not `spandex` as a structured material from `98% Polyester, 2% Spandex`. The fallback represented both. The system then needed the later `Imported; Zipper closure` answer. |
| `public_0027` | T2 R4 -> T2 R6 | Correctly captured cotton and zipper closure but dropped `Imported` entirely and represented closure differently from the fallback. The target stayed a hit but fell two ranks. |
| `public_0031` | T8 R1 -> T8 R2 | Captured `Imported` but dropped `Zipper closure`. At the final composition answer, both systems had cotton/polyester evidence, but the missing closure signal left the target at rank 2. |
| `public_0036` | T2 R1 -> T3 R1 | Qwen correctly kept only `pink` as color and placed the long marketing sentence in search terms. The fallback incorrectly stored the whole sentence as a color value, and that accidental confirmed-attribute boost found the target before `leather` was disclosed. This is a baseline artifact, not a desirable production behavior. |
| `public_0037` | T4 R2 -> T4 R3 | Misclassified `Imported` as a material alongside cotton. `Imported` is generic feature metadata, not a material, so the bad structured constraint reduced ranking quality. |
| `public_0040` | T5 R1 -> T4 R7 | The LLM-driven state caused the question policy to ask feature before color. It reached the target one turn earlier, but `Imported` remained a search term rather than a feature and the target was only rank 7. The evaluator ends immediately, so it never received the later chance to reach rank 1. |
| `public_0042` | T4 R1 -> T2 R2 | Misclassified `Imported` as material and returned the target at rank 2 immediately after color. The baseline waited for chronograph/calendar details and returned rank 1. Two saved turns were worth less than the reciprocal-rank loss under the competition formula. |
| `public_0045` | T3 R3 -> T3 R4 | Extracted the right material and features, but also duplicated generic `Imported` and `Button closure` into search terms. That altered exact-phrase/relaxed retrieval and moved the target down one rank. |
| `public_0047` | T3 R1 -> T3 R2 | Kept polyester and pink as structured preferences but omitted the fallback's duplicate `100% Polyester` and `color: pink` search evidence. The cleaner state was semantically valid, but the current ranker rewards the duplicated lexical representation. |
| `public_0048` | T3 R2 -> T4 R2 | After the evaluator said no material preference, the changed candidate diagnostics made the LLM path ask color before feature. The fallback asked feature immediately. Both eventually ranked the target second, but Qwen spent one extra turn. |

## Root-Cause Groups

### 1. Evidence Encoding Does Not Match Ranking Weights

Affected most clearly:

`public_0002`, `public_0008`, `public_0018`, `public_0020`, `public_0027`,
`public_0031`, `public_0045`, `public_0047`.

The LLM should update semantic state, but a deterministic adapter should derive
retrieval fields from that state. The prompt should not be responsible for
knowing which duplicate representation happens to work with current weights.

### 2. Incomplete or Incorrect Attribute Classification

Examples:

- compound materials were incompletely expanded in `public_0007` and
  `public_0021`;
- closure/sole metadata was omitted or left unstructured in `public_0016`,
  `public_0018`, `public_0027`, and `public_0031`;
- `Imported` was incorrectly called material in `public_0037` and
  `public_0042`.

### 3. Override Semantics

`public_0003` is the clearest actual state-transition failure. Qwen preserved a
preference that the user explicitly said to ignore. A deterministic transition
guard should convert explicit `ignore earlier preference` language into a
same-product replacement when a new attribute value is supplied.

### 4. Question-Policy Side Effects

Different state representation changed candidate disagreement and therefore
question ordering in `public_0007`, `public_0016`, `public_0021`, `public_0040`,
and `public_0048`. These are not independent question-model failures; they are
downstream effects of the candidate pool and evidence fields.

### 5. Evaluator Versus Product Behavior

Two regressions should not be copied blindly into production fixes:

- `public_0036`: the baseline won through an incorrect giant color value.
- `public_0042`: the LLM found the target two turns earlier at rank 2, but the
  evaluator preferred waiting for rank 1.

## Recommended Fix Order

1. Add one deterministic canonicalization layer after either interpreter:
   expand all known materials in compound values; map closures, soles, wash
   instructions, and `Imported` to feature metadata; never map `Imported` to
   material.
2. Derive retrieval query terms deterministically from canonical state. Do not
   let the LLM choose ranking-field duplication.
3. Enforce explicit override transitions so ignored values are retired even if
   the model returns `merge`.
4. Recompute clarification value from the canonical candidate pool, with an
   earlier `other` path when standard facets no longer discriminate.
5. Add a confidence gate for early low-rank recommendations. This should be a
   product-facing uncertainty policy, not a hardcoded turn schedule.

The key conclusion is not that LLM state updates are inherently worse. The
current LLM output contract and the tuned retrieval contract are misaligned.
Fixing that boundary is more transferable than adding sample-specific prompt
examples or retuning weights around these 17 public sessions.

## Five-Case OpenAI API Check

Five clear Qwen interpretation failures were rerun through the configured
OpenAI model (`gpt-5.6-luna`) using the same strict state-update tool and
evidence validation. This tested interpretation only, not end-to-end ranking.

| Source | Result | OpenAI interpretation |
|---|---|---|
| `public_0003` | Failed | Captured `water resistant` as feature, but still preserved the ignored `stainless steel band` material. A deterministic override-retirement rule is still required. |
| `public_0007` | Passed | Extracted polyester, rayon, and spandex from the compound composition. Qwen had extracted only polyester. |
| `public_0037` | Passed | Preserved cotton and classified both `Imported` and `Pull On closure` as features. Qwen had classified `Imported` as material. |
| `public_0031` | Ranker-compatibility partial | Preserved cotton and captured `Zipper closure` as feature. It classified `Imported` as `other`, which is semantically defensible but differs from the evaluator/current ranker's `feature` convention. |
| `public_0016` | Ranker-compatibility partial | Preserved leather and captured `Rubber sole` as feature. It classified `Imported` as `other`, which is semantically defensible but differs from the evaluator/current ranker's `feature` convention. |

Measured totals:

- 2/5 exactly matched the current ranker-facing field convention;
- 4/5 were semantically acceptable: the two additional cases put `Imported`
  under `other`, while correctly classifying the distinctive closure/sole;
- 1/5 still failed the important override transition;
- average API latency was 3.242 seconds per update;
- token usage was 5,526 prompt and 851 completion tokens across five calls.

The evaluator maps unrecognized metadata such as `Imported` to `feature`, and
the current feature route participates in candidate retrieval. `other` is not
included in that route and can only help score a product after it is already a
candidate. This is why `other` is weaker in this implementation. It is not a
universal taxonomy rule; a production catalog would preferably model
`Imported` under an explicit origin/manufacturing field.

This supports a stronger hybrid design, but not unrestricted LLM ownership of
state. OpenAI was materially more reliable than Qwen on compound extraction and
gross attribute classification. A deterministic adapter can align four of
these five outputs with the competition's retrieval convention, while
transition guards must still enforce explicit removal of ignored preferences.
