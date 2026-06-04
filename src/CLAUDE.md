# src/ — Python source

## Style

- `from __future__ import annotations` everywhere
- Type hints on all public functions
- Google-style docstrings for non-trivial functions
- Logging via `logging`, never `print()`
- Immutable configs: `@dataclass(frozen=True)` or Pydantic `BaseModel`

## Imports

```python
# 1. stdlib
# 2. third-party
# 3. local (src.*)
```

## Anti-patterns

- No global mutable state — pass explicitly
- No bare `except:` — catch specific exceptions
- No hardcoded paths — load from configs/env
- No `Any` unless truly justified — use `TypeAlias` or generics
