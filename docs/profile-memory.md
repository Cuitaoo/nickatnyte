# Privacy-Safe Long-Term Profile Distillation

## Purpose

The official evaluator creates a random `session_id` and supplies an anonymized
aggregate `user_profile`. It does not expose a stable user identity. The Agent
therefore never links evaluator sessions or writes anonymous sessions into a
shared customer profile.

For production integration, the Agent emits structured, persistence-ready
profile updates. An authenticated application layer may write those updates
under its own opaque user key.

```text
validated state patch
  -> short-term ShoppingState
  -> observe explicit durable evidence
  -> ProfileUpdate (bounded durable evidence excerpt)
  -> authenticated application
  -> JsonProfileStore / production profile service
```

## Update Policy

An update is emitted only when all of these conditions hold:

1. The shopper uses explicit durable language such as `usually`, `always`,
   `generally`, `typically`, `I tend to`, or `my go-to`.
2. No transient marker such as `for this one`, `today`, `right now`, or `as a
   gift` is present.
3. The normal state reducer produced validated category-scoped preference
   evidence on the same turn.
4. The attribute is suitable for durable personalization: `material`, `color`,
   `style`, `brand`, `feature`, or `use_case`.

Ordinary evaluator wording such as `For that, what matters is: cotton` remains
session-only. Budget, size, agent recommendations, raw search terms, and
unvalidated model text are never persisted.

## Agent Integration

`Agent.profile_updates(session_id)` exposes observed updates without adding
fields to the official response contract:

```python
response = agent.respond(session_id, user_message, turn, top_k=10)
updates = agent.profile_updates(session_id)
```

The official evaluator ignores this method. A production wrapper that already
knows the authenticated user may persist the updates:

```python
store = JsonProfileStore("data/long_term_user_profile_updates.json")
store.apply_updates(authenticated_user_key, session_id, updates)
```

`user_key` must come from the application identity layer. Never substitute the
random evaluator `session_id` or infer identity from repeated profile contents.

## JSON Format

The generated file is intentionally gitignored. It contains normalized values,
confidence, support count, and a maximum-160-character evidence excerpt from
the sentence that triggered the update. It never stores full conversation
history:

```json
{
  "schema_version": 1,
  "users": {
    "demo_user_42": {
      "preferences": [
        {
          "category_scope": "jeans",
          "attribute": "style",
          "value": "relaxed fit",
          "polarity": "prefer",
          "confidence": 0.9,
          "support_count": 1,
          "sources": ["explicit_long_term"],
          "evidence": [
            {
              "session_id": "demo_session_1",
              "turn": 1,
              "source": "explicit_long_term",
              "evidence_excerpt": "I usually prefer relaxed-fit jeans."
            }
          ]
        }
      ]
    }
  }
}
```

Writes use a temporary file followed by an atomic replacement. Replaying the
same session evidence is idempotent; evidence from an independent session
increases support and confidence.

The excerpt provides auditability and lets a customer understand or correct a
stored preference. A real profile service should additionally apply its normal
PII-redaction, access-control, and retention policies before persistence.

## Demo

From the repository root:

```bash
python3 -m tools.demo_profile_memory
```

This writes `data/long_term_user_profile_updates.json` and prints both the
emitted delta and resulting stored profile. Use a second `--session-id` to
demonstrate repeated cross-session support:

```bash
python3 -m tools.demo_profile_memory --session-id demo_session_2
```

## Evaluation Safety

Profile observation does not mutate `ShoppingState.user_profile`, active
preferences, search terms, retrieval, ranking, clarification, recommendation
depth, or the response payload. It adds no model call. Consequently it has no
intended path into Hit@10, MRR, or MTTC.

The JSON adapter is a production demonstration and is not invoked by the local
or official evaluator.
