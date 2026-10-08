# Security

Needle stores private documents and spends API credit on every question, so the server treats
every caller as untrusted until they sign in, and treats every document as untrusted input to the
language models.

## Threat model

| Asset | Threat | Control |
| --- | --- | --- |
| Documents, conversations, settings | Anyone who can reach the port reads, deletes, or wipes them | All `/api` routes require a session or the access token (`security.py`) |
| OpenRouter credit | Unauthenticated or runaway use | Sign-in required; per-client rate limits on chat, uploads, refresh, and login |
| Sessions | Cookie theft or cross-site requests | HttpOnly, SameSite=Strict, Secure on HTTPS; writes need the `X-Needle-CSRF` header and a same-origin `Origin` |
| The browser | XSS through document names, passages, or model output | Everything rendered is HTML-escaped; strict CSP (`script-src 'self'`, no inline script); stored files are only ever served as attachments with fixed, non-executable types |
| The server | Malicious uploads | Extension allowlist, content sniffing that must match the extension, 50 MB streamed cap, zip-bomb limits for Office files, PDF page cap, encrypted PDFs refused, sanitised display names, files stored under generated ids |
| Answers | Prompt injection inside documents | Instruction-like sentences are stripped from passages (whole passage dropped if mostly instructions); passages are wrapped as untrusted data and cannot close their wrapper tag; a second model checks every draft; numbers, quotes, and citations are verified deterministically |
| Configuration | Information disclosure | `/health` returns only `{"status": "ok"}`; the API schema is off unless `NEEDLE_ENABLE_API_DOCS=true`; unknown `Host` headers are refused |

## Signing in

- Set `NEEDLE_ACCESS_TOKEN` (16+ characters). If unset, a random token is generated on first
  start, printed once in the server console, and stored in `backend/secrets/access_token`
  (git-ignored, file mode 600).
- The token is compared in constant time. Login attempts are limited to 10 per 5 minutes per client.
- Sessions are HMAC-signed cookies valid for `NEEDLE_SESSION_HOURS` (12). Changing
  `NEEDLE_SESSION_SECRET` (or deleting `backend/secrets/session_secret`) signs everyone out.
- API clients can send `Authorization: Bearer <token>` instead of using a session.
- `NEEDLE_AUTH_DISABLED=true` exists for local development only.

Needle is single-tenant: everyone with the token sees the same workspace. There are no per-user
permissions. If you need them, put Needle behind an identity-aware proxy (for example an SSO
gateway) and keep the token secret.

## Public demo mode

`NEEDLE_DEMO_MODE=true` turns Needle into a read-only public demo (see `docs/deploy-railway.md`).
Visitors need no token but can only read the sample corpus and ask questions; every other route
returns 403 and the owner token still unlocks full control. Each visitor sees only their own
conversations (a random HttpOnly cookie id), other visitors' question text is never shown in
analytics, questions are limited to 500 characters, and per-client rate limits plus a global daily
question cap (`NEEDLE_DEMO_DAILY_QUESTIONS`, default 300) bound spend. Because visitors cannot
upload, hostile documents cannot be introduced; the only prompt-injection text is the deliberate
sample in the eval corpus. Set a credit limit on the OpenRouter key as the final backstop.

## Deploying

- `python main.py` binds `127.0.0.1`. To serve a network, set `NEEDLE_HOST=0.0.0.0` and list your
  hostname in `NEEDLE_ALLOWED_HOSTS`.
- Terminate TLS in front of Needle and set `NEEDLE_COOKIE_SECURE=true`.
- Behind a proxy, run uvicorn with `--proxy-headers` so rate limits apply per real client
  (`render.yaml` does this).
- Keep `.env` and `backend/secrets/` out of version control (both are git-ignored).

## Dependencies

`backend/requirements.txt` pins every direct dependency. CI runs `pip-audit` on each push.
Four chromadb advisories are ignored there on purpose: PYSEC-2026-311, -3813, -3814, and -3815
affect Chroma's HTTP server mode (its auth providers and remotely supplied embedding configs).
Needle embeds Chroma in-process and never starts that server. Re-check them when upgrading.

## Known limits

- Injection filtering is pattern-based. It catches common phrasings and the checker model is a
  second line of defence, but a determined, novel phrasing can get past the patterns. Documents
  that discuss prompt attacks may have example sentences stripped.
- Document text is sent to OpenRouter (and, for Jev and the answer models, to the providers it
  routes to). Do not index material that must not leave your machine unless you replace the
  hosted models.
- Data at rest (SQLite, Chroma, uploaded originals) is not encrypted by Needle. Use disk
  encryption on the host.

## Reporting

Report vulnerabilities privately to the repository owner rather than in a public issue.
