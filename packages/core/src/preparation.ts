import {
  QUESTION_TYPES,
  renderOptions,
  serializeState,
  validateQuestion,
} from "./questions.js";
import type {
  PreparedBatch,
  Question,
  Questions,
  State,
  Tokenizer,
} from "./types.js";

export interface PreparedItem {
  ids: number[];
  markers: number[];
  questionType: number;
}

export interface PreparedQuestions {
  batch: PreparedBatch;
  inputTokens: number;
}

function cleanMaskToken(text: string, tokenizer: Tokenizer): string {
  return text.replaceAll(tokenizer.maskToken, " ");
}

export function buildSequence(
  tokenizer: Tokenizer,
  state: State,
  question: Question,
  maxLength = 512,
  headMaxLength = 192,
): PreparedItem {
  validateQuestion(question);
  const options = renderOptions(question);
  const instructionIds = tokenizer.encode(
    `${question.type} question: ${cleanMaskToken(question.instructions, tokenizer)}`,
  );
  let optionIds = options.map((option) => [
    tokenizer.maskTokenId,
    ...tokenizer
      .encode(` ${cleanMaskToken(option, tokenizer)}`)
      .slice(0, 48),
  ]);
  let optionBudget =
    headMaxLength -
    optionIds.reduce((total, option) => total + option.length, 0);
  if (optionBudget < 16) {
    const perOption = Math.max(
      4,
      Math.floor((headMaxLength - 16) / Math.max(1, optionIds.length)),
    );
    optionIds = optionIds.map((option) => option.slice(0, perOption));
    optionBudget =
      headMaxLength -
      optionIds.reduce((total, option) => total + option.length, 0);
  }

  const ids = [
    tokenizer.clsTokenId,
    ...instructionIds.slice(0, Math.max(8, optionBudget)),
    tokenizer.sepTokenId,
  ];
  const markers: number[] = [];
  for (const option of optionIds) {
    markers.push(ids.length);
    ids.push(...option);
  }
  ids.push(tokenizer.sepTokenId);

  const availableStateTokens = Math.max(0, maxLength - ids.length - 1);
  const stateIds = tokenizer
    .encode(cleanMaskToken(serializeState(state), tokenizer))
    .slice(0, availableStateTokens);
  ids.push(...stateIds, tokenizer.sepTokenId);

  const truncatedIds = ids.slice(0, maxLength);
  const validMarkers = markers.filter((marker) => marker < maxLength);
  if (validMarkers.length !== options.length) {
    throw new Error(
      `Question options exceed headMaxLength=${headMaxLength}`,
    );
  }
  return {
    ids: truncatedIds,
    markers: validMarkers,
    questionType: QUESTION_TYPES[question.type],
  };
}

export function collateItems(
  items: readonly PreparedItem[],
  padTokenId: number,
  padToLength?: number,
): PreparedBatch {
  if (items.length === 0) {
    throw new Error("At least one question is required");
  }
  const longestItem = Math.max(...items.map((item) => item.ids.length));
  if (padToLength !== undefined && padToLength < longestItem) {
    throw new Error(
      `padToLength=${padToLength} is shorter than the longest item (${longestItem})`,
    );
  }
  const sequenceLength = padToLength ?? longestItem;
  const optionCount = Math.max(...items.map((item) => item.markers.length));
  const inputIds = new BigInt64Array(items.length * sequenceLength);
  inputIds.fill(BigInt(padTokenId));
  const attentionMask = new Uint8Array(items.length * sequenceLength);
  const markerPositions = new BigInt64Array(items.length * optionCount);
  const markerMask = new Uint8Array(items.length * optionCount);
  const questionTypes = new BigInt64Array(items.length);

  items.forEach((item, row) => {
    item.ids.forEach((tokenId, column) => {
      inputIds[row * sequenceLength + column] = BigInt(tokenId);
      attentionMask[row * sequenceLength + column] = 1;
    });
    item.markers.forEach((position, column) => {
      markerPositions[row * optionCount + column] = BigInt(position);
      markerMask[row * optionCount + column] = 1;
    });
    questionTypes[row] = BigInt(item.questionType);
  });

  return {
    batchSize: items.length,
    sequenceLength,
    optionCount,
    inputIds,
    attentionMask,
    markerPositions,
    markerMask,
    questionTypes,
  };
}

export function prepareQuestions(
  tokenizer: Tokenizer,
  state: State,
  questions: Questions,
  maxLength: number,
  headMaxLength: number,
): PreparedQuestions {
  const items = Object.values(questions).map((question) =>
    buildSequence(tokenizer, state, question, maxLength, headMaxLength),
  );
  return {
    batch: collateItems(items, tokenizer.padTokenId, maxLength),
    inputTokens: items.reduce((total, item) => total + item.ids.length, 0),
  };
}
