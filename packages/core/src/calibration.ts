const MIN_TEMPERATURE = 0.5;
const MAX_TEMPERATURE = 5;

export function clampTemperature(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(MAX_TEMPERATURE, Math.max(MIN_TEMPERATURE, value));
}

export function softmax(values: readonly number[]): number[] {
  if (values.length === 0) {
    throw new Error("Cannot compute softmax for an empty array");
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("Model returned non-finite logits");
  }
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function confidenceFromProbabilities(
  probabilities: readonly number[],
): number {
  if (probabilities.length < 2) {
    return 1;
  }
  const entropy = -probabilities.reduce(
    (sum, probability) =>
      sum + probability * Math.log(Math.max(probability, 1e-12)),
    0,
  );
  return Math.max(
    0,
    Math.min(1, 1 - entropy / Math.log(probabilities.length)),
  );
}

export function temperatureBucket(
  type: "choice" | "score" | "noul",
  optionCount: number,
): string {
  const size =
    optionCount <= 2
      ? "2"
      : optionCount <= 5
        ? "3-5"
        : optionCount <= 10
          ? "6-10"
          : "11+";
  return `${type}:${size}`;
}
