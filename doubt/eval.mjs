// node eval.mjs --url URL  (needs DOUBT_KEY) | CF_ACCOUNT_ID+CF_API_TOKEN (direct REST) | neither (mock wiring check)
import { cases } from './corpus/cases.mjs'
import { review, restAsker } from './src/doubt.mjs'
const url = process.argv.includes('--url') ? process.argv[process.argv.indexOf('--url') + 1] : null
let run, mode
if (url) { mode = 'live-url'; run = async (code) => { const r = await fetch(url + '/review', { method: 'POST', headers: { 'x-doubt-key': process.env.DOUBT_KEY ?? '', 'content-type': 'application/json' }, body: JSON.stringify({ code }) }); if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json() } }
else if (process.env.CF_ACCOUNT_ID && process.env.CF_API_TOKEN) { mode = 'direct-rest'; const ask = restAsker({ accountId: process.env.CF_ACCOUNT_ID, token: process.env.CF_API_TOKEN, model: process.env.MODEL }); run = (code) => review(code, ask) }
else { mode = 'MOCK'; const mock = async () => ({ inbound: { probabilities: { every: 0.34, some: 0.33, none: 0.33 } } }); run = (code) => review(code, mock) }
const rows = []
for (const c of cases) rows.push({ id: c.id, truth: c.protected, r: await run(c.code) })
let regexWrongClear = 0, finalWrongClear = 0, cost = 0
for (const { id, truth, r } of rows) {
  if (r.regex.protected && !truth) regexWrongClear++
  if (r.verdict === 'protected' && !truth) finalWrongClear++
  if (r.regex.protected && truth && r.verdict === 'open') cost++
  console.log(`${id.padEnd(24)} truth=${truth ? 'prot' : 'OPEN'} regex=${r.regex.protected ? 'prot' : 'open'} final=${r.verdict.padEnd(9)} pEvery=${r.model.pEvery.toFixed(3)} ctrl=${r.model.controlPEvery.toFixed(3)} choice=${r.model.choice ?? '-'}${r.escalated ? '  ESCALATED' : ''}`)
}
const modelRight = rows.filter(({ truth, r }) => (r.model.pEvery >= 0.6) === truth).length
console.log(`\nmode=${mode}`)
console.log(`open code called protected: regex ${regexWrongClear} -> with doubt ${finalWrongClear}`)
console.log(`protected code newly flagged by doubt (cost): ${cost}   (always-doubt control: 2)`)
console.log(`model alone right on ${modelRight}/16 (regex alone: 6/16)`)
