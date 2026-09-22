import type { ModelManifest } from "./types.js";

export function parseManifest(value: unknown): ModelManifest {
  if (typeof value !== "object" || value === null) {
    throw new Error("Model manifest must be an object");
  }
  const manifest = value as Partial<ModelManifest>;
  if (manifest.schemaVersion !== 1) {
    throw new Error(`Unsupported model manifest version: ${manifest.schemaVersion}`);
  }
  if (manifest.model?.format !== "onnx") {
    throw new Error("Model manifest must describe an ONNX model");
  }
  if (
    !manifest.files?.model ||
    !manifest.files.tokenizer ||
    !manifest.files.config
  ) {
    throw new Error("Model manifest is missing required files");
  }
  if (
    !manifest.inference ||
    manifest.inference.maxLength < 1 ||
    manifest.inference.headMaxLength < 1
  ) {
    throw new Error("Model manifest has invalid inference limits");
  }
  if (
    !manifest.calibration ||
    manifest.calibration.temperatures.length !== 3
  ) {
    throw new Error("Model manifest requires three calibration temperatures");
  }
  return manifest as ModelManifest;
}
