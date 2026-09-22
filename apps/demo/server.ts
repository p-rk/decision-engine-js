import { DecisionEngine } from "../../packages/node/src/index.js";
import { resolve } from "node:path";

const modelDirectory = resolve(
  process.env.MODEL_DIR ??
    resolve(import.meta.dir, "../../models/laya-english-fp32"),
);
const indexFile = Bun.file(resolve(import.meta.dir, "public/index.html"));
const port = Number(process.env.PORT ?? 3000);

let modelStatus: "idle" | "loading" | "ready" | "error" = "idle";
let modelLoadMs: number | null = null;
let agentPromise: Promise<DecisionEngine> | null = null;

function roundMilliseconds(value: number): number {
  return Math.round(value * 100) / 100;
}

function loadAgent(): Promise<DecisionEngine> {
  if (!agentPromise) {
    modelStatus = "loading";
    const startedAt = performance.now();
    const loading = DecisionEngine.load(modelDirectory)
      .then((agent) => {
        modelLoadMs = roundMilliseconds(performance.now() - startedAt);
        modelStatus = "ready";
        console.log(`Model loaded in ${modelLoadMs} ms`);
        return agent;
      })
      .catch((error) => {
        modelStatus = "error";
        agentPromise = null;
        console.error("Model loading failed", error);
        throw error;
      });
    agentPromise = loading;
    return loading;
  }
  return agentPromise;
}

type DecisionType = "choice" | "score" | "noul";

interface DecisionInput {
  state: string;
  question: string;
  type: DecisionType;
  criteria: string[] | { false?: string; true?: string };
}

function readText(value: unknown, name: string, maxLength: number): string {
  const text = typeof value === "string" ? value.trim() : "";
  if (!text) {
    throw new Error(`${name} is required`);
  }
  if (text.length > maxLength) {
    throw new Error(`${name} must be ${maxLength.toLocaleString()} characters or fewer`);
  }
  return text;
}

function parseCriteria(
  value: unknown,
  type: DecisionType,
): DecisionInput["criteria"] {
  if (type === "noul") {
    if (value === undefined) {
      return {};
    }
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new Error("Noul criteria must be an object");
    }
    const criteria = value as Record<string, unknown>;
    const falseDescription =
      typeof criteria.false === "string" ? criteria.false.trim() : "";
    const trueDescription =
      typeof criteria.true === "string" ? criteria.true.trim() : "";
    if (falseDescription.length > 500 || trueDescription.length > 500) {
      throw new Error("Noul descriptions must be 500 characters or fewer");
    }
    return {
      ...(falseDescription && { false: falseDescription }),
      ...(trueDescription && { true: trueDescription }),
    };
  }

  const criteria = Array.isArray(value)
    ? value.map((item) => (typeof item === "string" ? item.trim() : ""))
    : [];
  const maximum = type === "score" ? 10 : 20;
  if (criteria.length < 2 || criteria.length > maximum) {
    throw new Error(`Provide between 2 and ${maximum} criteria`);
  }
  if (criteria.some((criterion) => !criterion)) {
    throw new Error("Criteria must not be blank");
  }
  if (new Set(criteria).size !== criteria.length) {
    throw new Error("Criteria must be unique");
  }
  if (criteria.some((criterion) => criterion.length > 200)) {
    throw new Error("Criteria must be 200 characters or fewer");
  }
  return criteria;
}

function parseDecisionInput(value: unknown): DecisionInput {
  if (typeof value !== "object" || value === null) {
    throw new Error("Request body must be a JSON object");
  }
  const body = value as Record<string, unknown>;
  if (
    body.type !== undefined &&
    body.type !== "choice" &&
    body.type !== "score" &&
    body.type !== "noul"
  ) {
    throw new Error("Question type must be choice, score, or noul");
  }
  const type: DecisionType = body.type ?? "choice";
  return {
    state: readText(body.state, "State text", 20_000),
    question: readText(body.question, "Question", 1_000),
    type,
    criteria: parseCriteria(body.criteria ?? body.choices, type),
  };
}

function questionForInput(input: DecisionInput) {
  if (input.type === "noul") {
    return {
      type: "noul" as const,
      instructions: input.question,
      criteria: input.criteria as { false?: string; true?: string },
    };
  }
  return {
    type: input.type,
    instructions: input.question,
    criteria: input.criteria as string[],
  };
}

async function predict(input: DecisionInput) {
  const waitStartedAt = performance.now();
  const agent = await loadAgent();
  const modelWaitMs = performance.now() - waitStartedAt;
  const inferenceStartedAt = performance.now();
  const result = await agent.predict(input.state, {
    decision: questionForInput(input),
  });
  return {
    result,
    timing: {
      modelLoadMs,
      modelWaitMs: roundMilliseconds(modelWaitMs),
      inferenceMs: roundMilliseconds(performance.now() - inferenceStartedAt),
    },
  };
}

function isInputError(message: string): boolean {
  return [
    "Request body",
    "State text",
    "Question",
    "Question type",
    "Provide",
    "Criteria",
    "Noul",
    "Batch",
  ].some((prefix) => message.startsWith(prefix));
}

const server = Bun.serve({
  port,
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/") {
      return new Response(indexFile, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    if (request.method === "GET" && url.pathname === "/api/health") {
      return Response.json({
        status: "ok",
        model: {
          status: modelStatus,
          loadMs: modelLoadMs,
        },
      });
    }

    if (
      request.method === "POST" &&
      (url.pathname === "/api/decide" || url.pathname === "/api/classify")
    ) {
      const requestStartedAt = performance.now();
      try {
        const prediction = await predict(
          parseDecisionInput(await request.json()),
        );
        return Response.json({
          ...prediction,
          timing: {
            ...prediction.timing,
            totalMs: roundMilliseconds(performance.now() - requestStartedAt),
          },
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Decision failed";
        if (!isInputError(message)) {
          console.error("Decision failed", error);
        }
        return Response.json(
          { error: message },
          { status: isInputError(message) ? 400 : 500 },
        );
      }
    }

    if (request.method === "POST" && url.pathname === "/api/batch") {
      const requestStartedAt = performance.now();
      try {
        const body = await request.json();
        if (typeof body !== "object" || body === null) {
          throw new Error("Request body must be a JSON object");
        }
        const record = body as Record<string, unknown>;
        if (!Array.isArray(record.states)) {
          throw new Error("Batch states must be an array");
        }
        if (record.states.length < 1 || record.states.length > 50) {
          throw new Error("Batch must contain between 1 and 50 states");
        }
        const states = record.states.map((state) =>
          readText(state, "State text", 20_000),
        );
        const input = parseDecisionInput({ ...record, state: states[0] });
        const items = [];
        for (const state of states) {
          const prediction = await predict({ ...input, state });
          items.push({
            state,
            result: prediction.result,
            inferenceMs: prediction.timing.inferenceMs,
          });
        }
        return Response.json({
          items,
          timing: {
            modelLoadMs,
            totalMs: roundMilliseconds(performance.now() - requestStartedAt),
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Batch failed";
        if (!isInputError(message)) {
          console.error("Batch failed", error);
        }
        return Response.json(
          { error: message },
          { status: isInputError(message) ? 400 : 500 },
        );
      }
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  },
});

console.log(`Decision Engine JS demo: http://localhost:${server.port}`);
console.log(`Model directory: ${modelDirectory}`);
void loadAgent().catch(() => {
  // loadAgent logs the full error and the health endpoint exposes the failed status.
});

async function shutdown(): Promise<void> {
  server.stop();
  if (agentPromise) {
    try {
      const agent = await agentPromise;
      await agent.dispose();
    } catch {
      // The model-load error was already reported by loadAgent.
    }
  }
}

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());
