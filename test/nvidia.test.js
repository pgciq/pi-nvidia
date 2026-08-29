import assert from "node:assert/strict";
import test from "node:test";
import extension from "../extensions/nvidia.ts";

function getProviderConfig() {
  let config;
  extension({
    registerProvider(_name, providerConfig) {
      config = providerConfig;
    },
    registerEntryRenderer() {},
    appendEntry() {},
  });
  return config;
}

test("registers the nvidia provider with the unified base and env key", () => {
  const config = getProviderConfig();
  assert.equal(config.name, "NVIDIA NIM");
  assert.equal(config.baseUrl, "https://integrate.api.nvidia.com/v1");
  assert.equal(config.apiKey, "$NVIDIA_NIM_API_KEY");
  assert.equal(config.api, "openai-completions");
  assert.equal(typeof config.streamSimple, "function");
  assert.equal(typeof config.refreshModels, "function");
});

test("flags image, embedding, and VLM seed models correctly", () => {
  const { models } = getProviderConfig();
  assert.ok(Array.isArray(models) && models.length > 0);

  const image = models.find((m) => m.nvidiaImageModel);
  assert.ok(image, "an image model is seeded");
  assert.deepEqual(image.input, ["text"]);

  const embed = models.find((m) => m.nvidiaEmbeddingModel);
  assert.ok(embed, "an embedding model is seeded");
  assert.deepEqual(embed.input, ["text"]);

  const vlm = models.find((m) => m.input.includes("image"));
  assert.ok(vlm, "a VLM model is seeded");
  assert.deepEqual(vlm.input, ["text", "image"]);
});

test("uses a valid model array when refresh falls back to the cache", async () => {
  const config = getProviderConfig();
  const cachedModels = [{
    id: "cached-model",
    name: "Cached model",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 4096,
  }];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: { provider: "nvidia", models: cachedModels },
      publish: async () => true,
    });
    assert.deepEqual(result, cachedModels);
    assert.ok(Array.isArray(result));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("uses the seed list when offline with no cache", async () => {
  const config = getProviderConfig();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: undefined,
      publish: async () => true,
    });
    assert.ok(Array.isArray(result));
    assert.ok(result.length > 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
