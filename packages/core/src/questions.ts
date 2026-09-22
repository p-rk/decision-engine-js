import type { JsonValue, Question, State } from "./types.js";

export const QUESTION_TYPES = {
  choice: 0,
  score: 1,
  noul: 2,
} as const;

function renderCriterion(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function serializeState(state: State): string {
  return typeof state === "string" ? state : JSON.stringify(state);
}

export function labelsForQuestion(question: Question): string[] {
  if (question.type === "choice") {
    return Array.isArray(question.criteria)
      ? [...question.criteria]
      : Object.keys(question.criteria);
  }
  if (question.type === "score") {
    return question.criteria.map((_, index) => String(index));
  }
  return ["false", "true"];
}

export function renderOptions(question: Question): string[] {
  if (question.type === "choice") {
    if (Array.isArray(question.criteria)) {
      return [...question.criteria];
    }
    return Object.entries(question.criteria).map(([label, description]) =>
      description === null || description === ""
        ? label
        : `${label}: ${renderCriterion(description)}`,
    );
  }

  if (question.type === "score") {
    return question.criteria.map(
      (criterion, index) => `level ${index}: ${renderCriterion(criterion)}`,
    );
  }

  const configuredFalse = question.criteria?.false;
  const configuredTrue = question.criteria?.true;
  const falseDescription =
    configuredFalse === null ||
    configuredFalse === undefined ||
    configuredFalse === ""
      ? "no, the statement does not hold"
      : configuredFalse;
  const trueDescription =
    configuredTrue === null ||
    configuredTrue === undefined ||
    configuredTrue === ""
      ? "yes, the statement holds"
      : configuredTrue;
  return [
    `false: ${renderCriterion(falseDescription)}`,
    `true: ${renderCriterion(trueDescription)}`,
  ];
}

export function validateQuestion(question: Question): void {
  if (!question.instructions.trim()) {
    throw new Error("Question instructions must not be blank");
  }

  if (question.type === "choice") {
    const labels = labelsForQuestion(question);
    if (labels.length === 0) {
      throw new Error("Choice criteria must not be empty");
    }
    if (new Set(labels).size !== labels.length) {
      throw new Error("Choice labels must be unique");
    }
  } else if (question.type === "score" && question.criteria.length === 0) {
    throw new Error("Score criteria must not be empty");
  }
}
