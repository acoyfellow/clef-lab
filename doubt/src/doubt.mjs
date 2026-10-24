// doubt: a second opinion for pattern-based gates. It may raise a finding, never clear one.
import { scanAuthPatterns } from '../baseline.mjs'

export const QUESTIONS = {
  inbound: {
    type: 'choice',
    instructions:
      'This is server source code that handles incoming requests. Decide whether the server verifies WHO IS CALLING it ' +
      'before it reaches secrets, stored data, or paid resources. Credentials used only to call another service, ' +
      'comments, logging, and rate limiting do not count as verifying the caller.',
    criteria: {
      every: 'every request path verifies the incoming caller first',
      some: 'some paths or methods verify the caller and others do not',
      none: 'no path verifies the incoming caller',
    },
  },
}

// Control: lines shuffled deterministically; structure broken, vocabulary identical.
// Known weakness: on short code the auth words survive shuffling, so control scores can be high.
export function scramble(code) {
  const lines = code.split('\n')
  let s = 0x9e3779b9
  for (let i = lines.length - 1; i > 0; i--) {
    s = Math.imul(s ^ (s >>> 15), 2246822519) >>> 0
    const j = s % (i + 1)
    ;[lines[i], lines[j]] = [lines[j], lines[i]]
  }
  return lines.join('\n')
}

export async function review(code, ask, { threshold = 0.6 } = {}) {
  const patterns = scanAuthPatterns(code)
  const regexProtected = patterns.length > 0
  const [real, control] = await Promise.all([ask(code, QUESTIONS), ask(scramble(code), QUESTIONS)])
  const p = real.inbound.probabilities, pc = control.inbound.probabilities
  const pEvery = p.every ?? 0
  const escalate = regexProtected && pEvery < threshold // escalation only
  return {
    regex: { protected: regexProtected, patterns },
    model: { pEvery, pSome: p.some ?? 0, pNone: p.none ?? 0, choice: real.inbound.choice, controlPEvery: pc.every ?? 0, beatsControl: pEvery > (pc.every ?? 0) },
    verdict: regexProtected && !escalate ? 'protected' : 'open',
    escalated: escalate,
  }
}

// Workers AI REST. Wire format: { state, questions } -> result.answers
export function restAsker({ accountId, token, model = '@cf/cloudflare/clef', fetchImpl = fetch }) {
  const url = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${model}`
  return async (state, questions) => {
    const r = await fetchImpl(url, { method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' }, body: JSON.stringify({ state, questions }) })
    const j = await r.json().catch(() => ({}))
    if (!r.ok || !j.result) throw new Error(`clef ${r.status}: ${JSON.stringify(j.errors ?? j).slice(0, 400)}`)
    return j.result.answers
  }
}

export const bindingAsker = (ai, model = '@cf/cloudflare/clef') => async (state, questions) => {
  const out = await ai.run(model, { state, questions })
  return (out.result ?? out).answers
}
