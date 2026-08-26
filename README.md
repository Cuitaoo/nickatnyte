# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Set Up the Agent

Python 3.10 or later is recommended. Create an isolated environment and install the agent dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The agent uses LangChain and OpenAI for one structured preference update per shopper turn. Catalog search and product ranking remain local and deterministic; the model never generates product IDs.

To enable the model, copy the example configuration and put your own key in the local `.env` file:

```bash
cp .env.example .env
```

Never commit `.env` or paste its key into source code. This project intentionally does not load `.env` itself; export it into the current terminal before running:

```bash
set -a
source .env
set +a
```

Available settings:

- `OPENAI_API_KEY`: your private OpenAI API key.
- `OPENAI_MODEL`: model used for preference interpretation; defaults to `gpt-5.6-luna`.
- `OPENAI_TIMEOUT_SECONDS`: request timeout, clamped to 1–60 seconds.
- `OPENAI_MAX_RETRIES`: retry count, clamped to 0–3.
- `OPENAI_ENABLED`: set to `false` to guarantee no API calls.

If the key is missing, OpenAI is disabled, or a request fails, the agent automatically uses its deterministic preference parser.

## Run Tests and Evaluation

Run the offline tests without spending API credits:

```bash
OPENAI_ENABLED=false python3 -m unittest discover -s tests -v
```

Run the unchanged public evaluator in deterministic fallback mode:

```bash
OPENAI_ENABLED=false python3 -m evaluator.local_evaluator
```

Do not edit the evaluator or public labels when reporting your local score. The evaluator writes per-session results and aggregate metrics to the ignored `results.json` file.

### Bounded OpenAI Smoke Test

After exporting your key, use two turns to verify connectivity and memory before any larger paid run:

```bash
python3 - <<'PY'
from starter.agent import Agent

agent = Agent()
agent.reset("smoke", {"summary": "Prefers comfort.", "preference_tags": ["comfort"]})
print(agent.respond("smoke", "I need black leather hiking boots.", 1, 10))
print(agent.respond("smoke", "Actually, make that waterproof running shoes.", 2, 10))
agent.close()
PY
```

This smoke test makes at most two model requests. The full public evaluator can make as many as 2,000 requests (`200 sessions × 10 turns`). Before running it with OpenAI enabled, calculate expected input/output tokens from the smoke test, apply the model's current API prices, and get explicit team approval for that maximum spend.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

The current conversational lexical fallback scores Hit Rate@10 `0.805`, MRR
`0.440097`, MTTC `5.785`, and technical score `0.638829` on the same 200 public
sessions. These numbers use `OPENAI_ENABLED=false`; no API-backed full evaluation
has been run or claimed.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  competition adapter and turn coordinator
starter/llm_agent.py              one-call LangChain preference interpreter
starter/preference_tool.py        validated preference updates and fallback parser
starter/retrieval.py              local state-aware lexical retrieval
starter/questions.py              deterministic clarification policy
evaluator/local_evaluator.py      public-set simulator and scorer
requirements.txt                  Python dependencies
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
