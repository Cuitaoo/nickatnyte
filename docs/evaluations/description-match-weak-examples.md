# Description Match Weak-Set Replay

This replay tests the scoped change that includes product `description` in confirmed `material`, `color`, and `size` preference matching.

Run configuration:

```text
OPENAI_ENABLED=false
TECHJAM_VECTOR_ENABLED=true
TECHJAM_RERANK_ENABLED=true
TECHJAM_RERANK_WEIGHT=0.65
TECHJAM_RERANK_TOP_N=10
```

The replay uses the 10 sessions listed in `weak-public-set-examples.md`, not the full 200-session public set.

## Summary

| Session | Scenario | Before | After | Approx Delta |
|---|---:|---:|---:|---:|
| `public_0144` | intent_override | miss | miss | +0.0000 |
| `public_0154` | buying | miss | hit turn 2, rank 3 | +0.7800 |
| `public_0174` | buying | miss | miss | +0.0000 |
| `public_0175` | browsing | miss | miss | +0.0000 |
| `public_0198` | intent_override | hit turn 9, rank 7 | hit turn 9, rank 9 | -0.0095 |
| `public_0161` | buying | hit turn 9, rank 4 | hit turn 9, rank 4 | +0.0000 |
| `public_0126` | browsing | hit turn 6, rank 9 | hit turn 7, rank 9 | -0.0200 |
| `public_0035` | boundary | hit turn 5, rank 8 | hit turn 5, rank 8 | +0.0000 |
| `public_0087` | browsing | hit turn 5, rank 8 | hit turn 5, rank 8 | +0.0000 |
| `public_0137` | browsing | hit turn 6, rank 5 | hit turn 10, rank 1 | +0.1600 |

Net result on these weak examples: strongly positive, mostly because `public_0154` changes from a miss to a rank-3 hit.

## Key Fixed Case: `public_0154`

Target:

```text
B00CYNKSTE | Bestform Women's Wire Free Bra
```

The target has `cotton` and `white` only in `description`:

```text
100% Cotton cups are trimmed in a satin for a smooth fit.
Colors: White and Black.
```

Before the change, the target was retrieved but stayed around candidate rank 30+ because confirmed `material=cotton` and `color=white` were only checked against title/details/features.

After the change:

```text
T1 user: I'm looking for Bras Everyday Bras. A key requirement is: cotton.
   ask_attribute=color, target not in returned list
T2 user: For that, what matters is: color: white.
   ask_attribute=feature, target rank 3
```

Returned top 10 at the hit:

```text
1. B0023ZZAXW | Hanes 100% Cotton Lightly Lined Soft Cup 2-Pack, 34A-White/White
2. B005BT6C4I | Fruit of the Loom Women's 2pk Grey and White Cotton Stretch Extreme Comfort Bra
3. B00CYNKSTE | Bestform Women's Wire Free Bra <-- target
4. B086TZW76K | URATOT 6 Pieces Lace Bralettes for Women Girls Lace Daily Cami Bra
5. B0020A0TZY | Valmont Lacy Leisure Bra #23057, size 36B, white
6. B07H4N7BT9 | EUFANCE Bra Extenders Women's Comfortable Bra Extension Straps
7. B08B3CHZX3 | Generics Women's Sexy Floral Lace Front-Close Padded Bralette
8. B0857JPKMZ | Harley-Davidson Mens Screamin Eagle Vert White Short Sleeve T-Shirt
9. B0BVRKLSVP | Yacht & Smith Mens Wholesale Bulk Cotton Socks
10. B07T4VYHF4 | Marky G Apparel Men's Power Wash Tank
```

## Interpretation

This is a transferable fix because `description` is an official catalog field and is already used by FTS, vector text, and cross-encoder product text. The bug was that one final scoring stage ignored it for confirmed material/color/size preferences.

The weak-set result is promising but not enough by itself. A full 200-session eval is still needed because two weak sessions got slightly worse:

```text
public_0198: rank 7 -> rank 9
public_0126: turn 6 -> turn 7
```

If the full score improves or stays flat, this change is worth keeping because it fixes a real retrieval consistency issue.
