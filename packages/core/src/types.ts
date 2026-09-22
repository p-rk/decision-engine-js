export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export type State = string | JsonValue[] | { [key: string]: JsonValue };

export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: string[] | Record<string, JsonValue>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: JsonValue[];
}

export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: {
    false?: JsonValue;
    true?: JsonValue;
  };
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Record<string, Question>;

export interface ActionResult {
  act_probability: number;
}

export interface ChoiceAnswer {
  type: "choice";
  confidence: number;
  action: ActionResult;
  choice: string;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type: "score";
  confidence: number;
  action: ActionResult;
  score: number;
  legend: Record<string, JsonValue>;
  probabilities: Record<string, number>;
}

export interface NoulAnswer {
  type: "noul";
  confidence: number;
  action: ActionResult;
  noul: number;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface Prediction {
  model: string;
  answers: Record<string, Answer>;
  usage: {
    input_tokens: number;
    output_tokens: 0;
  };
}

export interface RawModelOutput {
  logits: readonly (readonly number[])[];
  actionLogits: readonly (readonly number[])[];
}

export interface PreparedBatch {
  batchSize: number;
  sequenceLength: number;
  optionCount: number;
  inputIds: BigInt64Array;
  attentionMask: Uint8Array;
  markerPositions: BigInt64Array;
  markerMask: Uint8Array;
  questionTypes: BigInt64Array;
}

export interface Tokenizer {
  clsTokenId: number;
  maskToken: string;
  maskTokenId: number;
  padTokenId: number;
  sepTokenId: number;
  encode(text: string): number[];
}

export interface ModelRuntime {
  run(batch: PreparedBatch): Promise<RawModelOutput>;
  dispose(): Promise<void>;
}

export interface ModelManifest {
  schemaVersion: 1;
  model: {
    id: string;
    revision?: string;
    responseName?: string;
    format: "onnx";
    precision: "fp32" | "fp16" | "int8" | "uint8";
  };
  files: {
    model: string;
    tokenizer: string;
    config: string;
    sha256?: Record<string, string>;
  };
  inference: {
    maxLength: number;
    headMaxLength: number;
    inputs: {
      inputIds: string;
      attentionMask: string;
      markerPositions: string;
      markerMask: string;
      questionTypes: string;
    };
    outputs: {
      logits: string;
      actionLogits: string;
    };
  };
  calibration: {
    temperatures: [number, number, number];
    temperaturesByOptions: Record<string, number>;
  };
}
