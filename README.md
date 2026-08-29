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

## Notes

- The seed list is a representative subset; the full chat/VLM/embedding catalog
  is fetched from `/v1/models` in the background and cached on disk.
- Image-generation models are merged from a fixed list (they are not in the
  unified `/v1/models` catalog).
- If you also use `pi-cn-free-model-providers`, it registers its own `nvidia`
  provider (chat only). Load one or the other to avoid an id clash.
