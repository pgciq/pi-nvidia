import assert from "node:assert/strict";
import test from "node:test";
import extension from "../extensions/nvidia.ts";

function loadExtension() {
  let config;
  const commands = new Map();
  extension({
    registerProvider(_name, providerConfig) {
      config = providerConfig;
    },
    registerCommand(name, options) {
      commands.set(name, options);
    },
    registerEntryRenderer() {},
    appendEntry() {},
  });
  return { config, commands };
}

function getProviderConfig() {
  return loadExtension().config;
}

// Invoke a command handler in "print" mode and capture what it would print.
// Pi's registry stamps `provider` onto each model at registration time; convertModel()
// does not, so the fake registry does it here.
async function runCommand(commands, config, name, args = "") {
  const printed = [];
  const originalLog = console.log;
  console.log = (line) => printed.push(String(line));
  try {
    await commands.get(name).handler(args, {
      mode: "print",
      hasUI: false,
      modelRegistry: {
        getAvailable: () => config.models.map((m) => ({ ...m, provider: "nvidia" })),
      },
    });
  } finally {
    console.log = originalLog;
  }
  return printed.join("\n");
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

test("registers the capabilities, context, and usage commands", () => {
  const { commands } = loadExtension();
  for (const name of ["nvidia-capabilities", "nvidia-context", "nvidia-usage"]) {
    assert.ok(commands.has(name), `${name} is registered`);
    assert.equal(typeof commands.get(name).handler, "function", `${name} has a handler`);
    assert.equal(typeof commands.get(name).description, "string");
  }
});

test("capability table renders seed models with the expected flags", async () => {
  const { config, commands } = loadExtension();
  const table = await runCommand(commands, config, "nvidia-capabilities");

  assert.match(table, /# NVIDIA NIM model capabilities/);
  assert.match(table, /\| Model \| Type \| Vision \| Image \| Embed \| Reasoning \| Tools \|/);

  const imageRow = table.split("\n").find((l) => l.includes("flux.1-schnell"));
  assert.match(imageRow, /\| imggen \| — \| ✓ \| — \| — \| — \|/);

  const embedRow = table.split("\n").find((l) => l.includes("nemoretriever-1b-vlm-embed-v1"));
  assert.match(embedRow, /\| embed \| — \| — \| ✓ \| — \| — \|/);

  const vlmRow = table.split("\n").find((l) => l.includes("llama-3.2-11b-vision-instruct"));
  assert.match(vlmRow, /\| vlm \| ✓ \| — \| — \| — \| ✓ \|/);

  const chatRow = table.split("\n").find((l) => l.includes("gpt-oss-20b"));
  assert.match(chatRow, /\| chat \| — \| — \| — \| — \| ✓ \|/);
});

test("capability filter narrows to matching models", async () => {
  const { config, commands } = loadExtension();
  const table = await runCommand(commands, config, "nvidia-capabilities", "embed");

  assert.match(table, /\(filter: embed\)/);
  // Every row must be an embedding model; no chat/vlm/imggen rows leak through.
  const rows = table
    .split("\n")
    .filter((l) => /^\| [^|]+ \| (chat|vlm|embed|imggen) \|/.test(l));
  assert.ok(rows.length > 0, "the filter matched at least one model");
  for (const row of rows) assert.match(row, /\| embed \|/);
});

test("context table reflects the documented max_tokens caps", async () => {
  const { config, commands } = loadExtension();
  const table = await runCommand(commands, config, "nvidia-context");

  assert.match(table, /# NVIDIA NIM context & max output/);
  // max_tokens.maximum === 8192 in the NIM OpenAPI schema for the 3.2V models.
  assert.match(table, /llama-3\.2-11b-vision-instruct \| vlm \| 128K \| 8K \|/);
  // Embedding models cap input at 512 tokens.
  assert.match(table, /nemoretriever-1b-vlm-embed-v1 \| embed \| 512 \| 512 \|/);
  // Nemotron-51B-Instruct caps output at 4096.
  assert.match(table, /llama-3\.1-nemotron-51b-instruct \| chat \| 128K \| 4K \|/);
  // Image models take `steps`, not tokens.
  const imageRow = table.split("\n").find((l) => l.includes("flux.1-schnell"));
  assert.match(imageRow, /\| imggen \| — \| — \|/);
});

test("context table sorts by max output descending", async () => {
  const { config, commands } = loadExtension();
  const table = await runCommand(commands, config, "nvidia-context", "max desc");

  assert.match(table, /sorted by maxTokens desc/);
  const rows = table
    .split("\n")
    .filter((l) => /^\| [^|]+ \| (chat|vlm|embed|imggen) \|/.test(l))
    .map((l) => {
      const raw = l.split("|").filter(Boolean).pop().trim();
      if (raw === "—" || raw === "-" || raw === "n/a") return 0;
      return raw.endsWith("K") ? Number(raw.slice(0, -1)) * 1000 : Number(raw);
    });
  // First row is the widest output; the list is monotonically non-increasing.
  assert.ok(rows[0] >= 8000, `top row has a large max output (got ${rows[0]})`);
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i - 1] >= rows[i]);
});
