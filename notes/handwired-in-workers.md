# Handwired in a Worker (field notes)

chenglou/handwired json-search, run in workerd via `wrangler dev`.

| Path | Result |
|---|---|
| WebGPU | `navigator.gpu` undefined |
| generated-JS backend | blocked: `new Function` is disallowed |
| executor backend | works: 2,000 docs 23.6 s; counts match RegExp |
| generated JS captured at build time, bundled | works: 2,000 docs 1.17-2.9 s; counts match RegExp |

RegExp did the same 2,000 docs in ~1 ms. In a Worker the value is guarantees (no catastrophic backtracking,
bit-identical outputs), not speed. 100k docs of cells (~205 MB) exceed Worker memory.
