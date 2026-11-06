const retired = new Set(["anticipator", "apprentice", "apprentice-v2", "apprentice-v3", "burst-focus", "burst-large", "burst-medium", "burst-micro", "burst-small", "burst-tiny", "challenger", "contender-v3", "counter-attacker-v4", "handcrafted-nano-v4", "handcrafted-small-v4", "prism-nano-v4", "selfplay", "selfplay-v2", "selfplay-v3", "strategist-v3"])

export function modelFile(name: string): URL {
  return new URL(retired.has(name)
    ? `./fixtures/models/${name}.onnx`
    : `../public/models/${name}.onnx`, import.meta.url)
}
