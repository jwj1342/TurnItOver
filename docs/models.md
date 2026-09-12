# Model API configuration

The model client supports text and static PNG/JPEG/WebP input through four REST protocols:

| Provider value | Endpoint | Default key variable |
|---|---|---|
| `openai` | OpenAI Responses `/v1/responses` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages `/v1/messages` | `ANTHROPIC_API_KEY` |
| `gemini` | Gemini `generateContent` | `GEMINI_API_KEY` |
| `openai_compatible` | Chat Completions `/v1/chat/completions` | `COMPATIBLE_API_KEY` |
| `openrouter` | OpenRouter `/api/v1/chat/completions` | `OPENROUTER_API_KEY` |

Protocol sources: [OpenAI image input](https://developers.openai.com/api/docs/guides/images-vision),
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[Gemini image input](https://ai.google.dev/gemini-api/docs/image-understanding).
OpenAI-compatible services differ in supported models/parameters; this adapter implements the basic
Chat Completions text/image subset, not every extension. Gemini here means the API-key Developer API,
not Vertex AI OAuth. No streaming, tool calls, native video or automatic retry is implemented.

## Configure locally

Copy `.env.example` to `.env` if it does not already exist, then fill in your own keys and exact model IDs.
The template deliberately does not choose potentially unavailable model names.
`generator`, `judge`, and `diagnosis` can use independent providers, models, endpoints and key variables.
For a single provider, set all three role provider fields to that provider; its API key can be shared.
Set only the roles you actually use. `models-check` reports all roles, so unused roles can remain unready.

`.env` is read by model commands only. Shell environment variables override file values, including empty values.
Supported syntax: `KEY=value`, single/double quotes, optional `export`, comments. No variable interpolation,
multiline values, shell execution, or sourcing is performed. Keep `.env` private (mode 600).
Do not place secrets in model IDs or URLs. Remote endpoints require HTTPS; localhost HTTP supports vLLM tunnels.

```bash
source scripts/setup_env.sh
python -m turnitover models-check
```

This checks local configuration and prints key presence, never the key. It does not prove credentials work,
model availability, vision support or quota. Unconfigured roles produce exit code 1.

## One model call

```bash
python -m turnitover model-call --role judge \
  --prompt-file my-prompt.txt --image reference.jpg --image candidate.png \
  --out output/judge-example
```

An actual call sends the prompt and supplied images to the configured service and may incur API cost.
Use `--dry-run` to prepare inputs without network access or keys. Each invocation makes at most one HTTP request;
timeouts and provider errors are surfaced without retries. API keys, headers and endpoints are not saved in output.
The folder contains input copies, the prompt, response text, returned model ID, usage, finish reason and latency.
Token accounting preserves each provider's native fields; no cross-provider price or usage normalization is claimed.

## Minimal image-to-program baseline

```bash
python -m turnitover reconstruct --image reference.jpg --out output/photo-baseline
python -m turnitover preview --program output/photo-baseline/program.ts --out output/photo-baseline-preview
```

`reconstruct` makes one generator call with the versioned prompt in `docs/prompts/reconstruct.txt`, extracts the
TypeScript factory and checks syntax with esbuild. It does not execute model output automatically. `preview`
then loads it into the browser ABI and exports media. Compile success is not ABI validity or reconstruction quality.
Multiple `--image` arguments are interpreted as views of the same object; no camera estimation is performed.
This is a single-pass baseline, **not an integration or reproduction of img2threejs, Articulate-Anything or Real2Code**.
The judge role is used by the active/fixed/random [verifier](verifier.md). The diagnosis role is independently
callable but is not yet wired into a repair loop. Local Qwen verification can bypass APIs with `--local-model`.

Remote APIs need a network-enabled node. Nibi compute nodes have no internet: run API calls on a login node,
then render saved programs through Slurm as needed. Local model servers require a reachable endpoint/tunnel.
Video input requires a separate frame-sampling or upstream reconstruction pipeline; passing an MP4 as `--image` is unsupported.

## Verification status

Provider wire formats, secret redaction, role overrides, response parsing and dry-run behavior have offline tests.
Live provider authentication and model quality require user-supplied credentials and a real reference image.
