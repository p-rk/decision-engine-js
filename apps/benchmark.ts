import {
  prepareQuestions,
  type ModelManifest,
} from "../packages/core/src/index.js";
import {
  HuggingFaceTokenizer,
  NodeOnnxRuntime,
  type NodeRuntimeOptions,
} from "../packages/node/src/index.js";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const modelDirectory = resolve("models/laya-english-fp32");
const manifest = JSON.parse(
  await readFile(resolve(modelDirectory, "manifest.json"), "utf8"),
) as ModelManifest;
const tokenizer = await HuggingFaceTokenizer.load(
  resolve(modelDirectory, "tokenizer"),
);
const questions = {
  decision: {
    type: "choice" as const,
    instructions: "Who should handle this?",
    criteria: ["billing", "technical", "sales"],
  },
};

const preparationStart = performance.now();
for (let index = 0; index < 100; index += 1) {
  prepareQuestions(
    tokenizer,
    "I was billed twice. Please refund the duplicate.",
    questions,
    manifest.inference.maxLength,
    manifest.inference.headMaxLength,
  );
}
const preparationAverageMs =
  (performance.now() - preparationStart) / 100;
console.log(`Preparation average: ${preparationAverageMs.toFixed(3)} ms`);

const prepared = prepareQuestions(
  tokenizer,
  "I was billed twice. Please refund the duplicate.",
  questions,
  manifest.inference.maxLength,
  manifest.inference.headMaxLength,
);
const configurations: Array<{
  name: string;
  options: NodeRuntimeOptions;
}> = [
  { name: "default", options: {} },
  {
    name: "4 threads sequential",
    options: {
      intraOpNumThreads: 4,
      interOpNumThreads: 1,
      executionMode: "sequential",
    },
  },
  {
    name: "6 threads sequential",
    options: {
      intraOpNumThreads: 6,
      interOpNumThreads: 1,
      executionMode: "sequential",
    },
  },
  {
    name: "8 threads sequential",
    options: {
      intraOpNumThreads: 8,
      interOpNumThreads: 1,
      executionMode: "sequential",
    },
  },
  {
    name: "10 threads sequential",
    options: {
      intraOpNumThreads: 10,
      interOpNumThreads: 1,
      executionMode: "sequential",
    },
  },
];

for (const configuration of configurations) {
  const loadStart = performance.now();
  const runtime = await NodeOnnxRuntime.create(
    resolve(modelDirectory, manifest.files.model),
    manifest,
    configuration.options,
  );
  const loadMs = performance.now() - loadStart;
  const timings: number[] = [];
  try {
    for (let run = 0; run < 4; run += 1) {
      const start = performance.now();
      await runtime.run(prepared.batch);
      timings.push(performance.now() - start);
    }
  } finally {
    await runtime.dispose();
  }
  const warm = timings.slice(1);
  const average =
    warm.reduce((total, timing) => total + timing, 0) / warm.length;
  console.log(
    `${configuration.name}: load=${loadMs.toFixed(1)} ms, ` +
      `cold=${(timings[0] ?? 0).toFixed(1)} ms, ` +
      `warm avg=${average.toFixed(1)} ms, ` +
      `runs=${warm.map((timing) => timing.toFixed(1)).join(", ")}`,
  );
}
