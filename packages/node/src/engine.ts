import {
  createPrediction,
  parseManifest,
  prepareQuestions,
  type ModelManifest,
  type Prediction,
  type Questions,
  type State,
} from "@decision-engine/core";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import {
  NodeOnnxRuntime,
  type NodeRuntimeOptions,
} from "./runtime.js";
import { HuggingFaceTokenizer } from "./tokenizer.js";

export interface LoadOptions {
  manifest?: string;
  runtime?: NodeRuntimeOptions;
}

export class DecisionEngine {
  private disposed = false;

  private constructor(
    readonly manifest: ModelManifest,
    private readonly tokenizer: HuggingFaceTokenizer,
    private readonly runtime: NodeOnnxRuntime,
  ) {}

  static async load(
    modelDirectory: string,
    options: LoadOptions = {},
  ): Promise<DecisionEngine> {
    const directory = resolve(modelDirectory);
    const manifestPath = resolve(
      directory,
      options.manifest ?? "manifest.json",
    );
    let manifestValue: unknown;
    try {
      manifestValue = JSON.parse(await readFile(manifestPath, "utf8"));
    } catch (error) {
      throw new Error(`Failed to read model manifest at ${manifestPath}`, {
        cause: error,
      });
    }
    const manifest = parseManifest(manifestValue);
    const tokenizerDirectory = dirname(
      resolve(directory, manifest.files.tokenizer),
    );
    const modelPath = resolve(directory, manifest.files.model);

    const tokenizer = await HuggingFaceTokenizer.load(tokenizerDirectory);
    const runtime = await NodeOnnxRuntime.create(
      modelPath,
      manifest,
      options.runtime,
    );
    return new DecisionEngine(manifest, tokenizer, runtime);
  }

  async predict(state: State, questions: Questions): Promise<Prediction> {
    if (this.disposed) {
      throw new Error("Cannot predict with a disposed DecisionEngine");
    }
    const prepared = prepareQuestions(
      this.tokenizer,
      state,
      questions,
      this.manifest.inference.maxLength,
      this.manifest.inference.headMaxLength,
    );
    const output = await this.runtime.run(prepared.batch);
    return createPrediction(
      questions,
      output,
      this.manifest,
      prepared.inputTokens,
    );
  }

  async dispose(): Promise<void> {
    if (!this.disposed) {
      this.disposed = true;
      await this.runtime.dispose();
    }
  }
}
