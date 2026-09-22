# Contributing

Thank you for improving Decision Engine JS.

## Development

Requirements:

- Bun 1.4+
- Node.js 20+
- Python 3.12 and `uv` for training or model conversion changes

Install dependencies and run the standard checks:

```bash
bun install
bun test
bun run typecheck
bun run build
bun run validate:data
bun run test:training
```

Keep changes focused and include tests for behavior changes. Do not commit generated models,
training runs, virtual environments, or ONNX files.

## Pull requests

Describe the problem, the chosen approach, and the validation performed. Mention model or output
compatibility changes explicitly and include before/after measurements for performance changes.
