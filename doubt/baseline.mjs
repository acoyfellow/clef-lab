// Baseline: a typical regex auth detector (bearer/env, Authorization + 401, Access JWT header, API key, signed webhooks).
export function scanAuthPatterns(source) {
  const patterns = []
  if (/Bearer\s*\$\{[^}]*env\.\w+/.test(source) || /`Bearer\s+\$\{[^}]*env\./.test(source) || /["']Bearer\s+["']\s*\+\s*env\./.test(source)) patterns.push('bearer-token')
  const readsAuthHeader = /headers\.get\(["']Authorization["']\)/.test(source) || /headers\.get\(["']authorization["']\)/.test(source)
  const has401 = /\b401\b/.test(source) || /\b403\b/.test(source) || /\bunauthorized\b/i.test(source)
  if (readsAuthHeader && has401 && !patterns.includes('bearer-token')) patterns.push('authorization-header')
  if (/Cf-Access-Jwt-Assertion/i.test(source)) patterns.push('cf-access-jwt')
  if ((/headers\.get\(["']x-api-key["']\)/i.test(source) || /headers\.get\(["']api-key["']\)/i.test(source) || /\.req\.header\(["']x-api-key["']\)/i.test(source) || /\.req\.header\(["']api-key["']\)/i.test(source)) && has401) patterns.push('api-key-header')
  if ((/["']x-hub-signature(-256)?["']/i.test(source) || /["']stripe-signature["']/i.test(source) || /["']x-slack-signature["']/i.test(source) || /["']x-shopify-hmac-sha256["']/i.test(source)) && has401) patterns.push('signed-webhook')
  return patterns
}
export const baselineSaysProtected = (code) => scanAuthPatterns(code).length > 0
