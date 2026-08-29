// NVIDIA NIM provider — https://build.nvidia.com / https://docs.api.nvidia.com/nim
//
// Hosted (serverless) base: https://integrate.api.nvidia.com/v1
//   - chat + vision (VLM): POST /v1/chat/completions   (image_url content for VLM)
//   - embeddings:          POST /v1/embeddings         (handled by pi's openai-compat core)
// Image generation (separate host + genai segment):
//   - POST https://ai.api.nvidia.com/v1/genai/<namespace>/<model>
//   - body { prompt, steps, seed }   (no `model` field; the model is in the URL path)
// Auth: NVIDIA_NIM_API_KEY env var (Bearer).
//
// Riva (ASR/TTS: parakeet / canary / chatterbox) is NOT wired — see the
// placeholder at the bottom. Riva models are self-hosted NIMs; probing
// ai.api.nvidia.com/{riva,asr,tts,nemo,speech,genai}/<model> all return 404,
// so there is no fixed hosted endpoint to call.

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { Image, Markdown } from "@earendil-works/pi-tui";

const UNIFIED_BASE = "https://integrate.api.nvidia.com/v1";
const IMAGE_GENAI_BASE = "https://ai.api.nvidia.com/v1/genai";
const API_KEY_ENV = "NVIDIA_NIM_API_KEY";

// Models served by the per-model "genai" host. These are NOT listed in the
// unified /v1/models catalog (which is chat/VLM/embedding only).
const IMAGE_MODELS = new Set([
  "black-forest-labs/flux.1-schnell",
  "black-forest-labs/flux.1-dev",
  "stabilityai/stable-diffusion-3.5-large",
  "stabilityai/stable-diffusion-3-medium",
  "stabilityai/stable-diffusion-xl",
  "google/diffusiongemma-26b-a4b-it",
]);

const VLM_MODELS = new Set([
  "meta/llama-3.2-11b-vision-instruct",
  "meta/llama-3.2-90b-vision-instruct",
  "microsoft/phi-3-vision-128k-instruct",
  "adept/fuyu-8b",
  "microsoft/kosmos-2",
  "nvidia/vila",
  "nvidia/neva-22b",
]);

// Models available immediately on startup (a representative subset; the full
// catalog is fetched from /v1/models in the background).
const SEED = [
  // chat
  "openai/gpt-oss-20b",
  "nvidia/nemotron-3-nano-30b-a3b",
  "moonshotai/kimi-k3",
  "nvidia/llama-3.1-nemotron-51b-instruct",
  // vision (VLM)
  "meta/llama-3.2-11b-vision-instruct",
  // embeddings
  "nvidia/nemotron-3-embed-1b",
  "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
  // image generation (genai host)
  "black-forest-labs/flux.1-schnell",
];

// Convert an absolute path to a clickable Markdown link. The TUI renders
// `[label](url)` as an OSC 8 hyperlink, so the saved file opens in one click.
function fileLink(p, label = p) {
  return `[${label}](${pathToFileURL(String(p)).href})`;
}

let appendNvidiaImage = null;

const openAICompletionsApi = await (async () => {
  try {
    return (await import("@earendil-works/pi-ai/api/openai-completions.lazy")).openAICompletionsApi;
  } catch {
    return (await import("@earendil-works/pi-ai")).openAICompletionsApi;
  }
})();

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------

function isEmbeddingId(id) {
  return /embed|embedqa/i.test(id);
}
function isVLMId(id) {
  return VLM_MODELS.has(id) || (/vision/i.test(id) && !isEmbeddingId(id));
}
function isImageId(id) {
  return IMAGE_MODELS.has(id);
}

function convertModel(model) {
  const id = typeof model?.id === "string" ? model.id : String(model?.id ?? "");
  const imageModel = isImageId(id);
  const embedding = isEmbeddingId(id);
  const vlm = !embedding && isVLMId(id);

  return {
    id,
    name: id,
    reasoning: false,
    input: vlm ? ["text", "image"] : ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072,
    maxTokens: 65536,
    // Private metadata (pi ignores unknown fields); consumed by streamNvidia.
    nvidiaImageModel: imageModel,
    nvidiaEmbeddingModel: embedding,
  };
}

async function fetchModels(baseUrl, signal) {
  const apiKey = process.env[API_KEY_ENV];
  const headers = {};
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const res = await fetch(`${baseUrl}/models`, { headers, redirect: "follow", signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

  const payload = await res.json();
  // OpenAI /v1/models returns { data: [{ id, ... }] }
  const data = Array.isArray(payload?.data) ? payload.data : Array.isArray(payload) ? payload : [];
  return data.filter((m) => m && m.id).map(convertModel);
}

// ---------------------------------------------------------------------------
// Image generation (genai host)
// ---------------------------------------------------------------------------

function latestTextPrompt(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  const user = [...messages].reverse().find((message) => message?.role === "user");
  if (!user) return "";
  if (typeof user.content === "string") return user.content;
  return (user.content ?? [])
    .filter((part) => part?.type === "text")
    .map((part) => part.text ?? "")
    .join("\n");
}

async function saveGeneratedImage(image, modelId) {
  const directory = join(process.cwd(), ".pi", "generated-images");
  await mkdir(directory, { recursive: true });
  const mime = image?.mime_type ?? image?.mimeType ?? "image/png";
  const extension = mime.includes("jpeg") ? "jpg" : mime.includes("webp") ? "webp" : "png";
  // model.id contains "/" and ".", which are unsafe in filenames.
  const safe = modelId.replace(/[\/.]/g, "-");
  const filePath = join(directory, `${safe}-${Date.now()}.${extension}`);

  const raw = image?.raw;
  let buf;
  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
    buf = Buffer.from(raw);
  } else if (typeof image?.b64_json === "string") {
    buf = Buffer.from(image.b64_json, "base64");
  } else if (typeof image?.image === "string") {
    buf = Buffer.from(image.image, "base64");
  } else if (image?.url) {
    const download = await fetch(image.url);
    if (!download.ok) throw new Error(`Unable to download generated image: HTTP ${download.status}`);
    buf = Buffer.from(await download.arrayBuffer());
  } else {
    throw new Error("NVIDIA image API returned no usable image data");
  }

  await writeFile(filePath, buf);
  return { filePath, mimeType: mime };
}

function streamImageGeneration(model, context, options) {
  const stream = createAssistantMessageEventStream();
  const output = {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "pending",
    timestamp: Date.now(),
  };

  (async () => {
    try {
      stream.push({ type: "start", partial: output });
      const prompt = latestTextPrompt(context);
      if (!prompt) throw new Error("Image generation requires a text prompt");
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) throw new Error(`${API_KEY_ENV} env var is required for image generation`);

      const endpoint = `${IMAGE_GENAI_BASE}/${model.id}`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, steps: 4, seed: 0 }),
        signal: options?.signal,
      });

      const contentType = response.headers.get("content-type") ?? "";
      let image;
      if (contentType.includes("application/json")) {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload?.error?.message ?? payload?.detail ?? `NVIDIA image API HTTP ${response.status}`);
        }
        // Tolerate the common NVIDIA response shapes:
        //   { data: [{ url | b64_json | image }] }
        //   { images: [{ image: "base64" }] }
        //   { image: "base64" }
        image = payload?.data?.[0] ?? payload?.images?.[0] ?? (payload?.image ? { image: payload.image } : payload);
      } else {
        if (!response.ok) throw new Error(`NVIDIA image API HTTP ${response.status}`);
        image = { raw: await response.arrayBuffer() };
      }

      const saved = await saveGeneratedImage(image, model.id);
      appendNvidiaImage?.({ path: saved.filePath, mimeType: saved.mimeType });
      const text = `Generated image saved to: ${fileLink(saved.filePath)}`;
      output.content.push({ type: "text", text });
      stream.push({ type: "text_start", contentIndex: 0, partial: output });
      stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: output });
      stream.push({ type: "text_end", contentIndex: 0, content: text, partial: output });
      output.stopReason = "stop";
      stream.push({ type: "done", reason: "stop", message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}

function streamNvidia(model, context, options) {
  if (model.nvidiaImageModel) return streamImageGeneration(model, context, options);
  return openAICompletionsApi().streamSimple(model, context, options);
}

// ---------------------------------------------------------------------------
// Riva (ASR / TTS) — PLACEHOLDER, NOT WIRED
// ---------------------------------------------------------------------------
// Riva models (parakeet / canary ASR, chatterbox TTS) are self-hosted NIMs.
// Probing https://ai.api.nvidia.com/v1/{riva,asr,tts,nemo,speech,genai}/<model>
// all return 404, so there is no fixed hosted endpoint. To wire them, point
// NVIDIA_RIVA_URL at your deployed Riva/NIM instance and call:
//   ASR: POST {NVIDIA_RIVA_URL}/v2/asrapi/recognize
//        body {"audio":"<base64>","config":{"language_code":"en-US",...}}
//   TTS: POST {NVIDIA_RIVA_URL}/v2/ttsapi/synthesize
//        body {"text":"...","language_code":"...","voice":{...}}
function streamRivaASR(_model, _context, _options) {
  throw new Error("Riva ASR is a placeholder: set NVIDIA_RIVA_URL and POST {base}/v2/asrapi/recognize");
}
function streamRivaTTS(_model, _context, _options) {
  throw new Error("Riva TTS is a placeholder: set NVIDIA_RIVA_URL and POST {base}/v2/ttsapi/synthesize");
}
// Intentionally NOT dispatched from streamNvidia — placeholders only.
void streamRivaASR;
void streamRivaTTS;

// ---------------------------------------------------------------------------
// Extension entry point (synchronous — no network on startup)
// ---------------------------------------------------------------------------

export default function (pi) {
  const baseUrl = UNIFIED_BASE;
  const apiKeyEnv = API_KEY_ENV;
  appendNvidiaImage = (image) => pi.appendEntry("nvidia-generated-image", image);
  pi.registerEntryRenderer?.("nvidia-generated-image", (entry, _options, theme) => {
    const image = entry.data ?? {};
    // pi passes an entry-renderer `theme` that lacks `fallbackColor()`, which
    // `Image.render` calls. Wrap it so inline previews render and never throw.
    const imageTheme = theme && typeof theme.fallbackColor === "function"
      ? theme
      : { fallbackColor: (s) => (theme && theme.fg ? theme.fg("toolOutput", s) : s) };
    try {
      const data = readFileSync(image.path).toString("base64");
      return new Image(data, image.mimeType || "image/png", imageTheme, { maxWidthCells: 80, maxHeightCells: 30 });
    } catch {
      return new Markdown(`Generated image unavailable: ${fileLink(image.path ?? "unknown path")}`, 1, 0, theme);
    }
  });

  pi.registerProvider("nvidia", {
    name: "NVIDIA NIM",
    baseUrl,
    // Keep this as an env reference even when the variable is absent. Pi can
    // then mark the provider as unconfigured instead of using a placeholder key.
    apiKey: `$${apiKeyEnv}`,
    api: "openai-completions",
    streamSimple: streamNvidia,
    models: SEED.map((id) => convertModel({ id })),

    async refreshModels({ signal, stored, publish, allowNetwork }) {
      const cachedModels = Array.isArray(stored?.models) ? stored.models : undefined;
      const seedModels = SEED.map((id) => convertModel({ id }));

      if (allowNetwork === false || signal.aborted) {
        return cachedModels?.length ? cachedModels : seedModels;
      }

      let models;
      try {
        const fetched = await fetchModels(UNIFIED_BASE, signal);
        // Image-generation models live on the separate genai host and are not
        // in the unified /v1/models catalog — merge them in explicitly.
        const seen = new Set(fetched.map((m) => m.id));
        for (const id of IMAGE_MODELS) {
          if (!seen.has(id)) fetched.push(convertModel({ id }));
        }
        models = fetched;
      } catch {
        return cachedModels?.length ? cachedModels : seedModels;
      }

      if (models.length > 0) {
        await publish({ persist: { provider: "nvidia", models } });
        return models;
      }

      return cachedModels?.length ? cachedModels : seedModels;
    },
  });
}
