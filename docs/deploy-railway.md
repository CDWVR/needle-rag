# Deploying the public demo on Railway

This puts Needle online as a **read-only public demo**: visitors browse a set of fictional sample
documents and ask questions; only you (with the access token) can upload, delete, or change
settings. No private data is public, and spend is capped.

## What visitors can and cannot do

| Visitors can | Visitors cannot |
| --- | --- |
| Read the sample documents and their passages | Upload, replace, or delete documents |
| Ask questions (6 a minute, 40 an hour per client, 500 characters each) | Change settings, refresh or roll back the index, reset the workspace |
| See their own conversations and rate answers | See anyone else's conversations or question text |
| View aggregate analytics and the eval result | Download originals or export analytics |

The whole demo shares a **daily question cap** (default 300, counted in UTC). With the default
models a question costs well under a cent, so the cap bounds the daily spend; also set a credit
limit on the OpenRouter key (below).

## Before you start

1. A Railway account with a payment method (the Hobby plan is enough).
2. An OpenRouter key with a **credit limit** (OpenRouter dashboard, Keys, set a limit such as $5).
   That limit is your hard stop if anything goes wrong.
3. This repository on GitHub (already the case).

## Steps

1. In Railway, **New Project, Deploy from GitHub repo**, and pick `needle-rag` (branch `master`
   once the demo PR is merged). Railway reads `railway.toml` and builds the `Dockerfile`.
2. **Add a volume** to the service and mount it at `/data`. Without it, the index, conversations,
   and the owner token vanish on every redeploy.
3. Set these **variables** on the service:

   | Variable | Value |
   | --- | --- |
   | `OPENROUTER_API_KEY` | your key |
   | `NEEDLE_DEMO_MODE` | `true` |
   | `NEEDLE_ACCESS_TOKEN` | a long random string (32+ characters); this is *your* owner sign-in |
   | `NEEDLE_COOKIE_SECURE` | `true` |
   | `NEEDLE_DEMO_DAILY_QUESTIONS` | optional, default `300` |

   For the Voice button, add `OPENROUTER_STT_API_KEY`: a **second** OpenRouter key with its own
   credit limit of **$1 or more**: OpenRouter rejects audio requests (HTTP 402) when the key has less
   than $0.50 of limit left. The app's own daily cap is what keeps spending near a cent a day. Clips are capped at 15 seconds and the app stops
   transcribing once it has spent `STT_DAILY_USD` (default `0.01`) in a UTC day. Without the key the
   Voice button falls back to the main key, and it is hidden if there is no key at all.

   `NEEDLE_SESSION_SECRET` is generated on first start and kept on the volume. The public domain
   and Railway's health checker are allowed automatically.
4. **Generate a domain** under Settings, Networking. Open it: the sample documents index in the
   background for about a minute after the first start, then the sample questions work.
5. To manage the demo, click the avatar (top right), choose owner sign-in, and enter your token.

## Checks after deploying

- `https://<your-domain>/health` returns `{"status":"ok"}`.
- A private browser window can ask a question but has no Upload button or Settings page.
- `curl -X POST https://<your-domain>/api/upload -H "X-Needle-CSRF: 1"` returns 403 ("disabled in
  the public demo"), not an upload prompt.
- Ask five questions quickly from one browser: the seventh within a minute is refused.

## Costs and limits

- **Railway:** the service needs roughly 1 GB of RAM while indexing and less when idle. Check the
  usage graph after the first day.
- **OpenRouter:** about $0.0004 per question in the hermetic eval. At 300 questions a day that is
  roughly $0.12.
- **Abuse:** per-client limits depend on Railway passing the real client address
  (`--proxy-headers`, already set in the image).

## Updating and resetting

- Pushing to the connected branch redeploys. The volume keeps data across deploys.
- To reload the sample documents from scratch, delete the volume contents (or the volume) and
  redeploy; they are re-seeded on start.
- To change your owner token, edit `NEEDLE_ACCESS_TOKEN` and redeploy. To also sign out every
  existing owner session, delete `secrets/session_secret` from the volume (Railway shell) or set
  a new `NEEDLE_SESSION_SECRET`.

## Running the same image locally

```bash
docker build -t needle-demo .
docker run --rm -p 8000:8000 -v needle-data:/data \
  -e OPENROUTER_API_KEY=... -e NEEDLE_DEMO_MODE=true \
  -e NEEDLE_ACCESS_TOKEN=change-me-to-something-long \
  needle-demo
```

Then open http://localhost:8000.
