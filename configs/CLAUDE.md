# Configs

YAML / TOML configs. One file per concern.

## Conventions

- Naming: `<purpose>-<scope>.yaml` (e.g. `train-stack-x.yaml`)
- Required keys at top, optional with defaults below
- Comment any value that is load-bearing or empirically tuned
- Reference paths via env vars or `${PROJECT_ROOT}/...`, not absolute

## Validation

- Schema-validate at load time (Pydantic / dataclass)
- Fail fast on missing required keys

## Anti-patterns

- Don't bury tuning constants deep in code — surface them here
- Don't duplicate values across configs — use a `_base.yaml` + override pattern
- Don't commit secrets/tokens — use env-var substitution
- Don't mix runtime and training/build configs in the same file
