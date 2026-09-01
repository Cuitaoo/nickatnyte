# CT LLM Integration on Justin's Agent Improvements

## Source

- Base: `origin/justin/agent-improvements` at `32b25f6`
- Integration branch: `justin-agent-improvements-ct-copy`
- Justin's original branch is not modified or merged into.

## Added Pipeline

```text
deterministic safe-case parser
-> ambiguity/risk gate
-> optional constrained LLM interpretation
-> deterministic canonicalizer
-> deterministic state reducer
-> Justin's hybrid retrieval and staged buying path
-> local cross-encoder
-> confidence/orchestration/business passes
```

The LLM is bypassed for direct clarification answers, explicit no-preference
answers, labeled exact identifiers, and known no-state-change replies. Natural
free-form requests, corrections, unresolved references, and product overrides
remain eligible for model interpretation.

The model returns a typed `PreferencePatch`; it does not mutate production
state directly. The canonicalizer preserves the deterministic parser's ranking
representation, normalizes common catalog metadata, rejects attribute-like
categories, and preserves explicit correction scope. The existing state reducer
is the only component that applies the final transition.

## Preserved From Justin's Branch

- `StrategyDecision` turn-level orchestration records
- confidence-driven deferral release
- staged hard-constraint filtering for buying
- recommendation depth scheduling
- audience handling and evidence-based explanations
- profile and dual-track experiments with their existing feature gates

## Validation

```text
focused preference/agent tests: 63 passed
complete test suite:             303 passed
```

The branch-level tests use fake interpreters and do not require network access.
A real API evaluation must explicitly enable `OPENAI_ENABLED=true` and
`OPENAI_AMBIGUITY_GATE_ENABLED=true`; API failures fall back deterministically.
