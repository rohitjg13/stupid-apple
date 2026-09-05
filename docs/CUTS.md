# Cut list

Published W4, executed W5 (Wed). Rohit decides. Ranked by demo value ÷ risk:
**first to go is at the top.**

| Order | Feature | Why it is cuttable | What we lose | Cost to cut |
|---|---|---|---|---|
| 1 | FINN shelf classifier (phase 4) | Never load-bearing; Sobel fill already detects stock-outs | A slide bullet | Zero — deck shows it as future work |
| 2 | Cloud sync | Off by default; the whole pitch is that it runs offline | Nothing on stage | Zero, `cloud.enabled: false` already |
| 3 | Multi-store view | One store in the demo | A dashboard tab | Small, `store_id` stays in the schema |
| 4 | Planogram diff beyond empty/low | Judges see the alert either way | Nuance in one alert | Small |
| 5 | Heatmap | Pretty, not decisive; costs tuning time | A visual in segment 2 | Medium — dashboard tile must be hidden cleanly |
| 6 | Live shelf segment | The only unrehearsable moment | The best 90 seconds of the demo | High — replace with shelf footage; only if it fails twice in W6 rehearsals |

**Not cuttable, ever:** the two contracts, the bus, `--source sim`, the laptop
fallback path, and the DPDP story. Those are what make the rest work.

**Rule for adding to this list:** anything proposed after W5 goes on here at
position 1, not into the plan.
