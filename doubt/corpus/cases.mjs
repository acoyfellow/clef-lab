// Planted cases. protected = every request path checks the caller before it reaches a secret or paid resource.
// Framework-agnostic: Workers, Hono, Express, Node http, Deno, Bun, FastAPI.
export const cases = [
  { id: 'w-outbound-bearer', protected: false, trap: 'Bearer used only to call out', code: `
export default { async fetch(req, env) {
  const body = await req.text()
  const r = await fetch('https://gateway.example.com/v1/chat', { method: 'POST',
    headers: { Authorization: \`Bearer \${env.GATEWAY_TOKEN}\` }, body })
  if (r.status === 401) return new Response('upstream unauthorized', { status: 502 })
  return r
} }` },
  { id: 'w-inbound-bearer', protected: true, code: `
export default { async fetch(req, env) {
  if (req.headers.get('Authorization') !== \`Bearer \${env.API_TOKEN}\`) return new Response('no', { status: 401 })
  return Response.json(await env.AI.run('@cf/meta/llama', { prompt: await req.text() }))
} }` },
  { id: 'w-reads-auth-forwards', protected: false, trap: 'reads Authorization only to forward it', code: `
export default { async fetch(req, env) {
  const auth = req.headers.get('Authorization') ?? \`Bearer \${env.FALLBACK_TOKEN}\`
  const r = await fetch('https://api.example.com/data', { headers: { Authorization: auth } })
  if (r.status === 401) console.log('upstream said unauthorized')
  return r
} }` },
  { id: 'w-access-jwt', protected: true, code: `
import { verifyAccessJwt } from './access'
export default { async fetch(req, env) {
  const jwt = req.headers.get('Cf-Access-Jwt-Assertion')
  if (!jwt || !(await verifyAccessJwt(jwt, env.AUD))) return new Response('forbidden', { status: 403 })
  return Response.json(await env.DB.prepare('select * from users').all())
} }` },
  { id: 'w-access-mention-only', protected: false, trap: 'Access header only in a comment and a log', code: `
// TODO: check Cf-Access-Jwt-Assertion here once Access is set up
export default { async fetch(req, env) {
  console.log('cf-access-jwt-assertion present?', !!req.headers.get('cf-access-jwt-assertion'))
  return Response.json(await env.AI.run('@cf/meta/llama', { prompt: await req.text() }))
} }` },
  { id: 'hono-partial', protected: false, trap: 'auth only on /admin; /chat open', code: `
import { Hono } from 'hono'
const app = new Hono()
app.use('/admin/*', async (c, next) => {
  if (c.req.header('x-api-key') !== c.env.ADMIN_KEY) return c.text('unauthorized', 401)
  await next()
})
app.post('/chat', async (c) => c.json(await c.env.AI.run('@cf/meta/llama', await c.req.json())))
app.get('/admin/stats', (c) => c.json({ ok: true }))
export default app` },
  { id: 'hono-global', protected: true, code: `
import { Hono } from 'hono'
import { bearerAuth } from 'hono/bearer-auth'
const app = new Hono()
app.use('*', async (c, next) => bearerAuth({ token: c.env.API_TOKEN })(c, next))
app.post('/chat', async (c) => c.json(await c.env.AI.run('@cf/meta/llama', await c.req.json())))
export default app` },
  { id: 'express-outbound', protected: false, trap: 'token used outbound; 401 handler is for upstream', code: `
const express = require('express')
const app = express()
app.post('/summarize', express.json(), async (req, res) => {
  const r = await fetch('https://api.openai.com/v1/responses', { method: 'POST',
    headers: { authorization: 'Bearer ' + process.env.OPENAI_KEY, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'gpt-x', input: req.body.text }) })
  if (r.status === 401) return res.status(500).send('bad upstream key')
  res.json(await r.json())
})
app.listen(3000)` },
  { id: 'express-middleware', protected: true, code: `
const express = require('express')
const app = express()
function requireKey(req, res, next) {
  if (req.get('x-api-key') !== process.env.CLIENT_KEY) return res.sendStatus(401)
  next()
}
app.use(requireKey)
app.post('/summarize', express.json(), async (req, res) => res.json(await callModel(req.body.text)))
app.listen(3000)` },
  { id: 'node-http-open', protected: false, trap: 'no auth words at all', code: `
import http from 'node:http'
http.createServer(async (req, res) => {
  const rows = await db.query('select email from customers')
  res.end(JSON.stringify(rows))
}).listen(8080)` },
  { id: 'node-http-custom-header', protected: true, trap: 'custom header a regex list would miss', code: `
import http from 'node:http'
import { timingSafeEqual } from 'node:crypto'
http.createServer(async (req, res) => {
  const got = Buffer.from(String(req.headers['x-internal-signature'] ?? ''))
  const want = Buffer.from(process.env.SHARED_SIG)
  if (got.length !== want.length || !timingSafeEqual(got, want)) { res.statusCode = 403; return res.end() }
  res.end(JSON.stringify(await db.query('select email from customers')))
}).listen(8080)` },
  { id: 'deno-method-hole', protected: false, trap: 'GET checks auth, POST does not', code: `
Deno.serve(async (req) => {
  if (req.method === 'GET') {
    if (req.headers.get('authorization') !== 'Bearer ' + Deno.env.get('TOKEN')) return new Response('', { status: 401 })
    return new Response(await kv.get(['report']))
  }
  await kv.set(['report'], await req.text())
  return new Response('saved')
})` },
  { id: 'bun-cookie-session', protected: true, trap: 'cookie session, no Authorization header', code: `
Bun.serve({ async fetch(req) {
  const sid = parseCookies(req.headers.get('cookie')).sid
  const user = sid && (await sessions.get(sid))
  if (!user) return Response.redirect('/login', 302)
  return Response.json(await billing.invoicesFor(user.id))
} })` },
  { id: 'fastapi-dep', protected: true, code: `
from fastapi import FastAPI, Depends, Header, HTTPException
import os
app = FastAPI()
def key(x_api_key: str = Header(...)):
    if x_api_key != os.environ["CLIENT_KEY"]:
        raise HTTPException(status_code=401)
@app.post("/embed", dependencies=[Depends(key)])
async def embed(body: dict):
    return await model.embed(body["text"])` },
  { id: 'fastapi-outbound', protected: false, trap: 'Authorization built for an outbound call', code: `
from fastapi import FastAPI
import httpx, os
app = FastAPI()
@app.post("/embed")
async def embed(body: dict):
    async with httpx.AsyncClient() as c:
        r = await c.post("https://api.example.com/embed", json=body,
                         headers={"Authorization": f"Bearer {os.environ['VENDOR_KEY']}"})
    if r.status_code == 401:
        return {"error": "unauthorized upstream"}
    return r.json()` },
  { id: 'w-rate-limit-only', protected: false, trap: 'rate limiting is not authentication', code: `
export default { async fetch(req, env) {
  const { success } = await env.RATE_LIMITER.limit({ key: req.headers.get('cf-connecting-ip') })
  if (!success) return new Response('slow down', { status: 429 })
  return Response.json(await env.AI.run('@cf/meta/llama', { prompt: await req.text() }))
} }` },
]
