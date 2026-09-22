import { createHash } from "node:crypto";
import { mkdir, rename, rm } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";

const repository = "p-rk/decision-engine-laya-onnx";
const revision =
  process.env.DECISION_ENGINE_MODEL_REVISION ??
  "521f99e060d76993e99f8f79154b735a5c2cb824";
const destination = resolve(
  process.env.MODEL_DIR ?? "models/laya-english-fp32",
);
const force = process.argv.includes("--force");
const files = [
  "manifest.json",
  "model.fp32.onnx",
  "rl_agent_config.json",
  "encoder/config.json",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
] as const;

interface DownloadResult {
  sha256: string;
  bytes: number;
}

interface Manifest {
  files?: {
    model?: string;
    sha256?: Record<string, string>;
  };
}

function remoteUrl(path: string): string {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  return `https://huggingface.co/${repository}/resolve/${revision}/${encodedPath}`;
}

function localPath(path: string): string {
  const result = resolve(destination, path);
  if (!result.startsWith(`${destination}${sep}`)) {
    throw new Error(`Refusing unsafe model path: ${path}`);
  }
  return result;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

async function sha256(path: string): Promise<DownloadResult> {
  const file = Bun.file(path);
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of file.stream()) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  return { sha256: hash.digest("hex"), bytes };
}

async function download(path: string): Promise<DownloadResult> {
  const target = localPath(path);
  const partial = `${target}.partial`;
  await mkdir(dirname(target), { recursive: true });
  await rm(partial, { force: true });

  console.log(`Downloading ${path}...`);
  const response = await fetch(remoteUrl(path), { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download ${path}: ${response.status} ${response.statusText}`,
    );
  }

  const expectedBytes = Number(response.headers.get("content-length") ?? 0);
  const writer = Bun.file(partial).writer();
  const hash = createHash("sha256");
  let bytes = 0;
  let nextProgress = 100 * 1024 * 1024;

  try {
    for await (const chunk of response.body) {
      writer.write(chunk);
      hash.update(chunk);
      bytes += chunk.byteLength;
      if (expectedBytes > 100 * 1024 * 1024 && bytes >= nextProgress) {
        console.log(
          `  ${formatBytes(bytes)} / ${formatBytes(expectedBytes)}`,
        );
        nextProgress += 100 * 1024 * 1024;
      }
    }
    await writer.end();
    if (expectedBytes && bytes !== expectedBytes) {
      throw new Error(
        `Incomplete download for ${path}: expected ${expectedBytes} bytes, received ${bytes}`,
      );
    }
    await rename(partial, target);
  } catch (error) {
    writer.end();
    await rm(partial, { force: true });
    throw error;
  }

  return { sha256: hash.digest("hex"), bytes };
}

async function ensureFile(
  path: string,
  expectedSha256?: string,
): Promise<void> {
  const target = localPath(path);
  const existing = Bun.file(target);
  if (!force && (await existing.exists())) {
    if (!expectedSha256) {
      console.log(`Using existing ${path}`);
      return;
    }
    console.log(`Verifying existing ${path}...`);
    const checked = await sha256(target);
    if (checked.sha256 === expectedSha256) {
      console.log(`Verified ${path} (${formatBytes(checked.bytes)})`);
      return;
    }
    console.log(`Checksum mismatch; downloading a clean ${path}`);
  }

  const downloaded = await download(path);
  if (expectedSha256 && downloaded.sha256 !== expectedSha256) {
    await rm(target, { force: true });
    throw new Error(
      `Checksum mismatch for ${path}: expected ${expectedSha256}, received ${downloaded.sha256}`,
    );
  }
  console.log(`Installed ${path} (${formatBytes(downloaded.bytes)})`);
}

console.log(`Decision Engine JS model: ${repository}@${revision}`);
console.log(`Destination: ${destination}`);

await ensureFile("manifest.json");
const manifest = (await Bun.file(localPath("manifest.json")).json()) as Manifest;
const modelPath = manifest.files?.model;
if (!modelPath || modelPath !== "model.fp32.onnx") {
  throw new Error("The downloaded manifest references an unexpected model file");
}
const modelSha256 = manifest.files?.sha256?.[modelPath];
if (!modelSha256) {
  throw new Error("The downloaded manifest does not provide a model SHA-256");
}

for (const path of files) {
  if (path !== "manifest.json") {
    await ensureFile(path, path === modelPath ? modelSha256 : undefined);
  }
}

console.log("\nSetup complete. Start the playground with: bun run demo");
