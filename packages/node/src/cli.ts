#!/usr/bin/env node

import { resolve } from "node:path";

import { DecisionEngine } from "./engine.js";

function usage(): never {
  console.error(
    "Usage: decision-engine --model <model-directory> <customer message>",
  );
  process.exit(2);
}

const args = process.argv.slice(2);
const modelFlag = args.indexOf("--model");
const modelArgument = modelFlag === -1 ? undefined : args[modelFlag + 1];
if (!modelArgument) {
  usage();
}
const modelDirectory = resolve(modelArgument);
args.splice(modelFlag, 2);
const text = args.join(" ").trim();
if (!text) {
  usage();
}

console.error("Loading model...");
const agent = await DecisionEngine.load(modelDirectory);
try {
  const result = await agent.predict(text, {
    department: {
      type: "choice",
      instructions: "Who should handle this?",
      criteria: ["billing", "technical", "sales"],
    },
  });
  console.log(JSON.stringify(result, null, 2));
} finally {
  await agent.dispose();
}
