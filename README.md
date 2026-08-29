# pi-nvidia

Pi extension for **NVIDIA NIM** — the hosted inference platform behind
[build.nvidia.com](https://build.nvidia.com).

Wires four hosted (serverless) capabilities against NVIDIA's two API bases,
plus a placeholder for Riva ASR/TTS (self-hosted, not wired).

## Setup

```bash
export NVIDIA_NIM_API_KEY=sk-...        # from https://build.nvidia.com (free tier)
```

Install like any pi extension:

```bash
npm install pi-nvidia
```

Then restart pi so the `nvidia` provider is loaded.

## Capabilities

| Capability | Endpoint | Status |
|---|---|---|
| Chat (text) | `POST https://integrate.api.nvidia.com/v1/chat/completions` | ✅ live-verified |
| Vision (VLM, image→text) | same, `image_url` content part | ✅ live-verified (described a test image correctly) |
| Embeddings | `POST https://integrate.api.nvidia.com/v1/embeddings` | ✅ endpoint verified; pi's openai-compat core calls it for embedding models |
| Image generation | `POST https://ai.api.nvidia.com/v1/genai/<namespace>/<model>` | ✅ route confirmed (GET→405); generation body per NVIDIA docs |
| **Riva ASR / TTS** | `{NVIDIA_RIVA_URL}/v2/asrapi/recognize` · `.../synthesize` | ⚠️ placeholder only — see below |

### Two API bases

NVIDIA splits its hosted models across two hosts:

- **Unified OpenAI-compatible base** — `https://integrate.api.nvidia.com/v1`
  serves chat, vision, and embeddings (addressed by `model` in the request
  body). This is what the extension uses for everything except images.
- **Per-model genai base** — `https://ai.api.nvidia.com/v1/genai/<namespace>/<model>`
  serves image-generation microservices. The model id (e.g.
  `black-forest-labs/flux.1-schnell`) is in the **URL path**, not the body; the
  body is `{"prompt": "...", "steps": 4, "seed": 0}`.

### Image generation

Generated images are saved under `.pi/generated-images/` and reported as a
clickable link (OSC 8 hyperlink). The handler tolerates the common response
shapes: `{data:[{url}]}`, `{data:[{b64_json}]}`, `{images:[{image}]}`,
`{image}`, or a raw image body.

### Riva ASR / TTS — placeholder

`parakeet` / `canary` (ASR) and `chatterbox` (TTS) are **self-hosted NIMs**.
There is no fixed hosted endpoint — probing
`ai.api.nvidia.com/v1/{riva,asr,tts,nemo,speech,genai}/<model>` all return 404.

To wire them later, set `NVIDIA_RIVA_URL` to your deployed Riva/NIM instance and
call:

- **ASR** — `POST {NVIDIA_RIVA_URL}/v2/asrapi/recognize`
  with `{"audio": "<base64>", "config": {"language_code": "en-US", ...}}`
- **TTS** — `POST {NVIDIA_RIVA_URL}/v2/ttsapi/synthesize`
  with `{"text": "...", "language_code": "...", "voice": {...}}`

`streamRivaASR` / `streamRivaTTS` exist as stubs but are **not dispatched**.

## Commands

Three commands show what a provider usually advertises. Tab-completion is
provided after `/nvidia-…`.

### `/nvidia-capabilities [vision|image|embed|reasoning|tools]`

Per-model capability table; the optional filter narrows to models that have that
capability.

```
/nvidia-capabilities
/nvidia-capabilities vision
```

| Model | Type | Vision | Image | Embed | Reasoning | Tools |
|---|---|:---:|:---:|:---:|:---:|:---:|
| openai/gpt-oss-20b | chat | — | — | — | — | ✓ |
| nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1 | embed | — | — | ✓ | — | — |
| black-forest-labs/flux.1-schnell | imggen | — | ✓ | — | — | — |
| meta/llama-3.2-11b-vision-instruct | vlm | ✓ | — | — | — | ✓ |

### `/nvidia-context [context|max] [asc|desc]`

Context window and max output tokens, sortable. Defaults to `context asc`.

```
/nvidia-context max desc
```

### `/nvidia-usage`

Opens the build.nvidia.com usage page in your browser. There is deliberately no
data table — **NVIDIA exposes no usage/quota API.** Probed 2026-08-29:

| Endpoint | Result |
|---|---|
| `/v1/usage` · `/v1/quotas` · `/v1/accounts` · `/v1/rate-limits` · `/v1/billing` | 404 |
| `build.nvidia.com/v1/analytics/budget_usage` | `text/html` (SPA, not JSON) |
| `build.nvidia.com/v1/analytics/cli-summary` | `text/html` (SPA, not JSON) |
| chat-completion response headers | no rate-limit headers |

Usage/quota is only visible in the build.nvidia.com web UI (session login, not
the `nvapi-…` API key).

### Where the capability and context numbers come from

`GET /v1/models` and `GET /v1/models/{id}` both return only
`{ id, object, created, owned_by }` — NVIDIA publishes **no** context window,
max tokens, modality, or pricing metadata. So:

- **Context / max output** — from each model's NIM docs page, which embeds an
  OpenAPI schema (`max_tokens.maximum` is authoritative, e.g. 8192 for the
  Llama 3.2 Vision models), plus model cards. Unlisted models fall back to the
  registered `131072 / 65536` defaults, and the table says so. Image models show
  `—` because they take `steps`, not tokens.
- **Capabilities** — `vision` / `image` / `embed` are derived from the model id
  (VLM set, image-gen set, `embed` in the name) and are reliable. `reasoning` /
  `tools` come from model cards and are conservative: live verification was
  blocked because `integrate.api.nvidia.com` was returning HTTP 500 for every
  request while this was written.

## Notes

- The seed list is a representative subset; the full chat/VLM/embedding catalog
  is fetched from `/v1/models` in the background and cached on disk.
- Image-generation models are merged from a fixed list (they are not in the
  unified `/v1/models` catalog).
- If you also use `pi-cn-free-model-providers`, it registers its own `nvidia`
  provider (chat only). Load one or the other to avoid an id clash.
