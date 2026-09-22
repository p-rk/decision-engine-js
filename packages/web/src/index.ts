import type {
  ModelManifest,
  ModelRuntime,
  PreparedBatch,
  RawModelOutput,
} from "@decision-engine/core";
import * as ort from "onnxruntime-web";

function toRows(tensor: ort.Tensor, expectedRows: number): number[][] {
  const values = Array.from(tensor.data as ArrayLike<number>, Number);
  const width = tensor.dims[1];
  if (
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== expectedRows ||
    width === undefined
  ) {
    throw new Error(
      `Expected a rank-2 output with ${expectedRows} rows, got [${tensor.dims.join(", ")}]`,
    );
  }
  return Array.from({ length: expectedRows }, (_, row) =>
    values.slice(row * width, (row + 1) * width),
  );
}

export class WebOnnxRuntime implements ModelRuntime {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly manifest: ModelManifest,
  ) {}

  static async create(
    modelUrl: string,
    manifest: ModelManifest,
  ): Promise<WebOnnxRuntime> {
    let session: ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
    } catch (webGpuError) {
      console.warn(
        "WebGPU initialization failed; falling back to WebAssembly.",
        webGpuError,
      );
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }
    return new WebOnnxRuntime(session, manifest);
  }

  async run(batch: PreparedBatch): Promise<RawModelOutput> {
    const names = this.manifest.inference.inputs;
    const feeds: Record<string, ort.Tensor> = {
      [names.inputIds]: new ort.Tensor("int64", batch.inputIds, [
        batch.batchSize,
        batch.sequenceLength,
      ]),
      [names.attentionMask]: new ort.Tensor("bool", batch.attentionMask, [
        batch.batchSize,
        batch.sequenceLength,
      ]),
      [names.markerPositions]: new ort.Tensor(
        "int64",
        batch.markerPositions,
        [batch.batchSize, batch.optionCount],
      ),
      [names.markerMask]: new ort.Tensor("bool", batch.markerMask, [
        batch.batchSize,
        batch.optionCount,
      ]),
      [names.questionTypes]: new ort.Tensor("int64", batch.questionTypes, [
        batch.batchSize,
      ]),
    };
    const outputs = await this.session.run(feeds);
    const outputNames = this.manifest.inference.outputs;
    const logits = outputs[outputNames.logits];
    const actionLogits = outputs[outputNames.actionLogits];
    if (!logits || !actionLogits) {
      throw new Error("ONNX model did not return the outputs declared in its manifest");
    }
    return {
      logits: toRows(logits, batch.batchSize),
      actionLogits: toRows(actionLogits, batch.batchSize),
    };
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}

export type {
  ModelManifest,
  ModelRuntime,
  PreparedBatch,
  RawModelOutput,
} from "@decision-engine/core";
