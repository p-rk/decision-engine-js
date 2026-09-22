import {
  clampTemperature,
  confidenceFromProbabilities,
  softmax,
  temperatureBucket,
} from "./calibration.js";
import { labelsForQuestion, validateQuestion } from "./questions.js";
import type {
  Answer,
  ModelManifest,
  Prediction,
  Questions,
  RawModelOutput,
} from "./types.js";

function round(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function temperatureFor(
  question: Questions[string],
  optionCount: number,
  manifest: ModelManifest,
): number {
  const typeIndex = question.type === "choice" ? 0 : question.type === "score" ? 1 : 2;
  const bucket = temperatureBucket(question.type, optionCount);
  return clampTemperature(
    manifest.calibration.temperaturesByOptions[bucket] ??
      manifest.calibration.temperatures[typeIndex],
  );
}

export function createPrediction(
  questions: Questions,
  output: RawModelOutput,
  manifest: ModelManifest,
  inputTokens: number,
): Prediction {
  const entries = Object.entries(questions);
  if (
    output.logits.length !== entries.length ||
    output.actionLogits.length !== entries.length
  ) {
    throw new Error("Model output batch size does not match the question count");
  }

  const answers: Record<string, Answer> = {};
  entries.forEach(([questionId, question], row) => {
    validateQuestion(question);
    const labels = labelsForQuestion(question);
    const logits = output.logits[row];
    const actionLogits = output.actionLogits[row];
    if (!logits || !actionLogits) {
      throw new Error(`Model output is missing row ${row}`);
    }

    const temperature = temperatureFor(question, labels.length, manifest);
    const probabilities = softmax(
      logits.slice(0, labels.length).map((value) => value / temperature),
    );
    const actionProbability = softmax(actionLogits)[0];
    if (actionProbability === undefined) {
      throw new Error("Action head returned no probabilities");
    }
    const base = {
      confidence: round(confidenceFromProbabilities(probabilities)),
      action: { act_probability: round(actionProbability) },
    };

    if (question.type === "choice") {
      const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
      const selected = labels[selectedIndex];
      if (selected === undefined) {
        throw new Error("Choice prediction did not select a label");
      }
      answers[questionId] = {
        type: "choice",
        ...base,
        choice: selected,
        probabilities: Object.fromEntries(
          labels.map((label, index) => [label, round(probabilities[index] ?? 0)]),
        ),
      };
    } else if (question.type === "score") {
      answers[questionId] = {
        type: "score",
        ...base,
        score: round(
          probabilities.reduce(
            (total, probability, index) => total + probability * index,
            0,
          ),
        ),
        legend: Object.fromEntries(
          question.criteria.map((criterion, index) => [String(index), criterion]),
        ),
        probabilities: Object.fromEntries(
          probabilities.map((probability, index) => [
            String(index),
            round(probability),
          ]),
        ),
      };
    } else {
      const trueProbability = probabilities[1] ?? 0;
      answers[questionId] = {
        type: "noul",
        ...base,
        noul: round(trueProbability),
        confidence: round(Math.max(trueProbability, 1 - trueProbability)),
      };
    }
  });

  return {
    model: manifest.model.responseName ?? manifest.model.id,
    answers,
    usage: { input_tokens: inputTokens, output_tokens: 0 },
  };
}
