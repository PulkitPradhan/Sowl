# LEARN.md — "what I learned" (plain words, max 5 lines per phase)

- **Phase 0:** In `src/` layout the pyproject `name` (`sowl`) is the distribution name, NOT the import name — hatchling builds an empty wheel unless told `[tool.hatch.build.targets.wheel] packages = ["src/ag"]`. `uv sync` installs your own project editable into `.venv`, which is why `import ag` just works in tests. `uv.lock` pins exact versions; commit it (apps, unlike libraries). TDD even works for scaffolding: wrote the import test first, watched all 13 fail with exactly `ModuleNotFoundError`, then made them pass.
