# Sowl

Local-first, general-purpose personal AI agent (daemon + thin CLI & Web harness), built to run on low-spec machines without GPU and scale up with hardware.

## Quickstart

### 1. Installation & Environment
```bash
# Sync dependencies
uv sync

# Run test suite and linter
uv run pytest
uv run ruff check .
```

### 2. Daemon Management
```bash
# Start daemon in background or foreground
agentd start

# Check daemon status
ag daemon status

# Stop daemon
ag daemon stop
```

### 3. Interactive Web Harness
Start the local-first web UI at `http://127.0.0.1:3180`:
```bash
# Start web harness (auto-attaches to or starts agentd, opens default browser)
ag web

# Run on a custom port without auto-opening browser
ag web --port 3200 --no-open
```

### 4. CLI Execution
```bash
# Check connectivity
ag ping

# Hardware diagnostics & tier recommendation
ag doctor

# Execute an autonomous task
ag run "Read notes.txt and tell me what it says"

# Manage model profiles
ag model list
ag model use tier2_gpu

# Session management
ag session list
ag session new --title "Research Session"
ag session show <session_id>
```

### 5. Evaluation Harness
```bash
# Run 20-task quick eval with Laya gate
ag eval --profile tier2_gpu --laya rules --quick

# Compare two evaluation runs
ag eval compare evals/results/run_a.json evals/results/run_b.json
```

### 6. Laya Decision & Safety Gate
Sowl protects against prompt injection, ungrounded actions, and hallucinated identifiers via the Laya Gate (`src/ag/laya/`):
- **Provenance & Taint Tracking**: Audits all identifier arguments (emails, URLs, file paths). Identifiers originating from untrusted tool outputs are flagged as tainted and require explicit human confirmation showing source origin.
- **URL & Contacts Safety**: Restricts non-HTTP(S) protocols, flags IP literals/punycode/private networks, and grounds contact lookups via `contacts.toml`.
- **Monotonic Layer B Judge**: Optional secondary LLM judge that can strictly escalate safety decisions (allow -> confirm/block) without loosening Layer A deterministic rules.

