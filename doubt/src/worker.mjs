import { review, bindingAsker } from './doubt.mjs'

// Guards itself the way it asks others to: a caller key, checked before any model call.
export default {
  async fetch(req, env) {
    const url = new URL(req.url)
    if (url.pathname === '/') return new Response('doubt: POST /review {code} with x-doubt-key. Escalation-only second opinion for auth-pattern gates.\n')
    if (url.pathname !== '/review' || req.method !== 'POST') return new Response('not found', { status: 404 })
    if (!env.DOUBT_KEY || req.headers.get('x-doubt-key') !== env.DOUBT_KEY) return new Response('unauthorized', { status: 401 })
    const { code } = await req.json().catch(() => ({}))
    if (typeof code !== 'string' || code.length === 0 || code.length > 40_000) return new Response('code: 1..40000 chars', { status: 400 })
    try {
      return Response.json(await review(code, bindingAsker(env.AI)))
    } catch (e) {
      return Response.json({ error: String(e).slice(0, 300) }, { status: 502 })
    }
  },
}
