import { describe, expect, it } from "bun:test";

import {
  clampTemperature,
  collateItems,
  confidenceFromProbabilities,
  createPrediction,
  buildSequence,
  prepareQuestions,
  renderOptions,
  type ModelManifest,
} from "./index.js";

const tokenizer = {
  clsTokenId: 101,
  maskToken: "[MASK]",
  maskTokenId: 103,
  padTokenId: 0,
  sepTokenId: 102,
  encode: (text: string) =>
    Array.from(text, (character) => character.codePointAt(0) ?? 0),
};

const manifest: ModelManifest = {
  schemaVersion: 1,
  model: { id: "test/laya", format: "onnx", precision: "int8" },
  files: {
    model: "model.onnx",
    tokenizer: "tokenizer.json",
    config: "config.json",
  },
  inference: {
    maxLength: 512,
    headMaxLength: 192,
    inputs: {
      inputIds: "input_ids",
      attentionMask: "attention_mask",
      markerPositions: "marker_pos",
      markerMask: "marker_mask",
      questionTypes: "qtype",
    },
    outputs: { logits: "logits", actionLogits: "action" },
  },
  calibration: {
    temperatures: [1, 1, 1],
    temperaturesByOptions: {},
  },
};

describe("calibration", () => {
  it("clamps unsafe temperatures", () => {
    expect(clampTemperature(0.1)).toBe(0.5);
    expect(clampTemperature(7)).toBe(5);
    expect(clampTemperature(Number.NaN)).toBe(1);
  });

  it("returns zero confidence for a uniform distribution", () => {
    expect(confidenceFromProbabilities([0.5, 0.5])).toBeCloseTo(0);
  });
});

describe("questions", () => {
  it("renders labeled choice descriptions", () => {
    expect(
      renderOptions({
        type: "choice",
        instructions: "Route this",
        criteria: { billing: "payments", technical: "bugs" },
      }),
    ).toEqual(["billing: payments", "technical: bugs"]);
  });

  it("uses default text for blank boolean criteria", () => {
    expect(
      renderOptions({
        type: "noul",
        instructions: "Is this true?",
        criteria: { false: "", true: "" },
      }),
    ).toEqual([
      "false: no, the statement does not hold",
      "true: yes, the statement holds",
    ]);
  });
});

describe("prediction", () => {
  it("creates a calibrated choice response", () => {
    const prediction = createPrediction(
      {
        department: {
          type: "choice",
          instructions: "Route this",
          criteria: ["billing", "technical", "sales"],
        },
      },
      { logits: [[4, 1, 0]], actionLogits: [[3, 0]] },
      manifest,
      28,
    );

    expect(prediction.answers.department).toMatchObject({
      type: "choice",
      choice: "billing",
      action: { act_probability: 0.9526 },
    });
    expect(prediction.usage.input_tokens).toBe(28);
  });
});

describe("preparation", () => {
  it("places one mask marker before each option", () => {
    const prepared = buildSequence(
      tokenizer,
      "refund please",
      {
        type: "choice",
        instructions: "Route this",
        criteria: ["billing", "technical", "sales"],
      },
      512,
      192,
    );
    expect(prepared.markers).toHaveLength(3);
    expect(prepared.markers.map((marker) => prepared.ids[marker])).toEqual([
      103, 103, 103,
    ]);
    expect(prepared.ids[0]).toBe(101);
    expect(prepared.ids.at(-1)).toBe(102);
  });

  it("pads question rows and preserves token counts", () => {
    const batch = collateItems(
      [
        { ids: [1, 2, 3], markers: [1], questionType: 0 },
        { ids: [4, 5], markers: [0, 1], questionType: 2 },
      ],
      99,
    );
    expect(batch.sequenceLength).toBe(3);
    expect(Array.from(batch.inputIds)).toEqual([1n, 2n, 3n, 4n, 5n, 99n]);
    expect(Array.from(batch.attentionMask)).toEqual([1, 1, 1, 1, 1, 0]);
  });

  it("pads prepared questions to the configured model context", () => {
    const prepared = prepareQuestions(
      tokenizer,
      "short state",
      {
        route: {
          type: "choice",
          instructions: "Route this",
          criteria: ["billing", "sales"],
        },
      },
      64,
      32,
    );
    expect(prepared.batch.sequenceLength).toBe(64);
    expect(prepared.inputTokens).toBeLessThan(64);
  });
});
