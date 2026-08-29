"""Replay sessions that first hit at a given rank and show what outranked the target.

Answers: is the winner an exact score tie (decided by the alphabetical
parent_asin tiebreak), a near-duplicate, or a genuine ranking loss?
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from starter.agent import Agent
from starter.retrieval import CatalogRetriever

_LAST: dict = {}
_orig_search = CatalogRetriever.search
def _spy(self, state, latest_message, top_k):
    result = _orig_search(self, state, latest_message, top_k)
    _LAST["result"] = result
    return result
CatalogRetriever.search = _spy


def replay(agent, sample, products, categories, catalog_ids):
    sid = str(sample["sample_id"])
    agent.reset(sid, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    disclosed, boundary_used = set(), False
    override_applied = sample["scenario_type"] != "intent_override"
    msg = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
    for turn in range(1, MAX_TURNS + 1):
        resp = agent.respond(sid, msg, turn, TOP_K)
        ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
        if override_applied and target in ranked:
            return turn, ranked.index(target) + 1, ranked, _LAST.get("result")
        if turn == MAX_TURNS:
            return None, None, ranked, _LAST.get("result")
        ov = eff.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(ov.get("turn", 3)):
            override_applied = True
            if str(ov.get("new_value", "")):
                disclosed.add(str(ov["new_value"]))
            msg = str(ov.get("message", "Actually, please ignore my earlier preference."))
        else:
            msg, boundary_used = customer_reply(eff, resp.get("ask_attribute"), disclosed, boundary_used)
    return None, None, [], _LAST.get("result")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    a = ap.parse_args()

    want = {x["sample_id"] for x in json.load(open(a.results))["sessions"]
            if x.get("best_rank") == a.rank}
    catalog_ids, categories, products = catalog_index(a.catalog)
    samples = [s for s in load_jsonl(a.dataset) if s["sample_id"] in want]
    agent = Agent(a.catalog)

    verdicts = Counter()
    print(f"replaying {len(samples)} sessions that first hit at rank {a.rank}\n")
    for s in samples:
        turn, rank, ranked, res = replay(agent, s, products, categories, catalog_ids)
        tgt = str(s["ground_truth"]["parent_asin"])
        if rank != a.rank:
            verdicts["did-not-reproduce"] += 1
            continue
        by_id = {c.product_id: c for c in (res.candidates if res else ())}
        win, t = ranked[0], tgt
        cw, ct = by_id.get(win), by_id.get(t)
        meta = agent.retriever.metadata
        tw, tt = meta.get(win, {}).get("title", "")[:58], meta.get(t, {}).get("title", "")[:58]
        if cw and ct:
            gap = cw.score - ct.score
            tie = abs(gap) < 1e-9
            alpha = tie and win < t
            shared = len(set(tw.split()) & set(tt.split())) / max(1, len(set(tw.split()) | set(tt.split())))
            v = "EXACT-TIE(alphabetical)" if alpha else ("exact-tie" if tie else
                 ("near-duplicate" if shared > 0.5 else "genuine-loss"))
            verdicts[v] += 1
            print(f"{s['sample_id']} [{s['scenario_type']}] turn {turn}  gap {gap:+.5f}  {v}")
            print(f"   won  {win}  score {cw.score:.5f}  routes {len(cw.route_ranks)}  matched {sorted(cw.matched_attributes)}")
            print(f"        {tw}")
            print(f"   tgt  {t}  score {ct.score:.5f}  routes {len(ct.route_ranks)}  matched {sorted(ct.matched_attributes)}")
            print(f"        {tt}")
            print(f"   title overlap {shared:.2f}\n")
        else:
            verdicts["reranked-out-of-candidates"] += 1
    agent.close()
    print("VERDICTS:", dict(verdicts))


if __name__ == "__main__":
    main()
