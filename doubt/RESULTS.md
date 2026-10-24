# doubt: results (2026-10-17)

Escalation-only Clef second opinion for regex auth detectors. Deployed as a Worker behind a caller key
(`wrangler secret put DOUBT_KEY`).

## Checks on the live deploy

| Check | Result |
|---|---|
| `GET /` | 200 |
| `POST /review` no key / wrong key | 401 / 401 |
| `POST /review` right key | 200 |
| Live URL vs direct REST, 16 cases | identical verdicts, max pEvery diff 0.0000 |

## Eval: 16 planted cases, real Clef (`@cf/cloudflare/clef`)

| | open handlers called "protected" | protected handlers called "open" |
|---|---|---|
| regex baseline (`baseline.mjs`) | 5 | 5 |
| regexes + doubt (escalation only) | **0** | 5 (doubt never clears, by design) |
| Clef alone (pEvery >= 0.6) | 0 | 0 (16/16) |
| always-doubt control | 0 | 7 |

Per case (pEvery = Clef's probability that every path verifies the caller; ctrl = same on line-shuffled code):

```
w-outbound-bearer        OPEN regex=prot -> open      pEvery=0.008 ctrl=0.016 ESCALATED
w-inbound-bearer         prot regex=prot -> protected pEvery=0.933 ctrl=0.163
w-reads-auth-forwards    OPEN regex=prot -> open      pEvery=0.019 ctrl=0.015 ESCALATED
w-access-jwt             prot regex=prot -> protected pEvery=0.898 ctrl=0.038
w-access-mention-only    OPEN regex=prot -> open      pEvery=0.029 ctrl=0.020 ESCALATED
hono-partial             OPEN regex=prot -> open      pEvery=0.041 ctrl=0.030 ESCALATED (choice=some)
hono-global              prot regex=open -> open      pEvery=0.969 ctrl=0.957
express-outbound         OPEN regex=open -> open      pEvery=0.008 ctrl=0.008
express-middleware       prot regex=open -> open      pEvery=0.956 ctrl=0.743
node-http-open           OPEN regex=open -> open      pEvery=0.009 ctrl=0.010
node-http-custom-header  prot regex=open -> open      pEvery=0.924 ctrl=0.537
deno-method-hole         OPEN regex=prot -> open      pEvery=0.040 ctrl=0.104 ESCALATED (choice=some)
bun-cookie-session       prot regex=open -> open      pEvery=0.935 ctrl=0.641
fastapi-dep              prot regex=open -> open      pEvery=0.895 ctrl=0.951
fastapi-outbound         OPEN regex=open -> open      pEvery=0.011 ctrl=0.009
w-rate-limit-only        OPEN regex=open -> open      pEvery=0.008 ctrl=0.010
```

## Limits

- 16 short, self-written, clearly labelled cases: a smoke test, not a benchmark. Next: ~30 real Worker entry files labelled by hand.
- The line-shuffle control is weak on short code (0.95 on two cases). Use a mismatched-contract control instead.
- Each review = 2 Clef calls of ~150-300 input tokens.

## Run

```sh
node eval.mjs                                   # mock, wiring only
CF_ACCOUNT_ID=... CF_API_TOKEN=... node eval.mjs # real Clef over REST
DOUBT_KEY=... node eval.mjs --url https://<your-worker>.workers.dev
```
