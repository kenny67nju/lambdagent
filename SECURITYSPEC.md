# LambdAgent Security Specification

## Overview

LambdAgent is the computational kernel for agent execution. Security is critical because:
- Agents execute LLM-generated tool calls (shell commands, file operations, browser control)
- The ReAct loop amplifies risk — one compromised step can cascade
- Multi-agent constructs (Channel, GroupChat) create inter-agent attack surfaces
- YAML configs are compiled to executable Lambda terms

## Threat Model

```
Malicious Input            Compromised LLM           Untrusted Config
      |                        |                        |
      v                        v                        v
  [Lam.apply()]           [ReAct Loop]             [from_config()]
  - Prompt injection       - Tool abuse              - YAML injection
  - Resource exhaustion    - Infinite loops           - Unsafe tool refs
  - Data exfiltration      - Secret extraction        - Resource bomb
                           - Command injection
```

## Security Architecture

### 1. Sandbox & Process Isolation

**Three-level isolation model:**

| Level | Mechanism | Use Case |
|-------|-----------|----------|
| L0 | In-process (no isolation) | Development, trusted agents |
| L1 | Subprocess + resource limits | Production, semi-trusted |
| L2 | Container (planned) | Untrusted agents, multi-tenant |

**SandboxPolicy presets:**
- `strict()`: no network, no subprocess, 64MB memory, 10s timeout
- `default()`: no network, no subprocess, 256MB memory, 30s timeout
- `permissive()`: network allowed, 512MB memory, 60s timeout

**Requirements:**
- Tool functions MUST be wrapped in `SandboxedTool` for production
- Resource limits MUST be enforced via POSIX `resource.setrlimit()`
- Input data to sandboxed processes MUST be serialized via JSON, NOT string interpolation

**macOS subprocess blocking (fixed 2026-03-28):**

Previously, `RLIMIT_NPROC` failed silently on macOS (`except: pass`), meaning
`allow_subprocess=False` had zero effect on Apple systems.

Fix: `ResourceLimiter.apply()` now detects the failure and falls back to
Python-level monkey-patching inside the child process:
- `subprocess.Popen/run/call/check_call/check_output` → `PermissionError`
- `os.system/popen` → `PermissionError`
- `os.execl/execle/execlp/execlpe/execv/execve/execvp/execvpe` → `PermissionError`

**Limitations of the macOS fallback:**
- This is a Python-level guard, NOT a kernel boundary
- Native C extensions or ctypes `fork()` calls bypass it
- It blocks casual subprocess usage from Python code within the sandbox
- For true process isolation on macOS, use L2 container sandbox (planned)

### 2. YAML Config Security

**Compilation (`from_config`):**
- MUST use `yaml.safe_load()` (never `yaml.load()`) — DONE
- Config MUST be validated against schema before compilation
- `maxSteps` MUST have an upper bound (default: 100, max: 1000)
- `maxTokens` MUST have an upper bound
- Tool references MUST be resolved from a whitelist, not arbitrary imports

**Linting (`lint.py`):**
- L001-L016 static analysis rules enforce config correctness
- SHOULD add security-specific lint rules:
  - S001: Detect unbounded maxSteps
  - S002: Detect missing guard on tool-using agents
  - S003: Detect overly permissive sandbox policy

### 3. ToolGateway — Runtime Permission Enforcement

> **Implementation**: `lambdagent/tool_gateway.py` (introduced 2026-03-28)
>
> Before this module, `guard.dangerousCommandBlock` and `guard.highRiskConfirmation`
> were declared in YAML but **never checked at runtime**. The ToolGateway closes this gap.

**Architecture:**

```
LLM output (tool_call)
       │
       ▼
┌─────────────────────┐
│  GatedTool.apply()   │   ← wraps every Tool compiled from YAML
├─────────────────────┤
│ 1. classify(tool, x) │   → RiskLevel: SAFE/LOW/MEDIUM/HIGH/CRITICAL
│ 2. check(policy)     │   → Action: ALLOW/BLOCK/CONFIRM/LOG_ONLY
│ 3. execute or block  │
│ 4. truncate output   │   → maxOutputLength enforcement
│ 5. audit log         │   → every call recorded with timestamp
└─────────────────────┘
```

**Risk Classification (5 levels):**

| Level | Examples | Default Action |
|-------|---------|----------------|
| CRITICAL | `rm -rf /`, `curl\|sh`, `cat ~/.ssh/*`, fork bomb | Always BLOCK |
| HIGH | `sudo`, `rm -rf`, `pip install`, `kill -9`, `git push --force` | BLOCK (or CONFIRM if `highRiskConfirmation`) |
| MEDIUM | `mv`, `sed -i`, `git rebase`, `cp -r`, `tee /` | LOG_ONLY (or BLOCK if `dangerousCommandBlock`+`blockMedium`) |
| LOW | `file write`, MCP remote calls, browser open | ALLOW + log |
| SAFE | `ls`, `cat`, `git status`, `terminate`, `screenshot` | ALLOW |

**YAML guard → GatewayPolicy mapping:**

```yaml
guard:
  dangerousCommandBlock: true    # → policy.block_dangerous = True
  highRiskConfirmation: true     # → policy.confirm_high_risk = True
  maxOutputLength: 3000          # → policy.max_output_length = 3000 (enforced!)
```

**Compilation flow (`fromconfig/compiler.py`):**

1. `build_agent()` reads `guard` config → calls `_build_gateway(guard_cfg)`
2. `_build_gateway()` creates `ToolGateway(GatewayPolicy.from_guard_config(...))`
3. Gateway injected into overrides as `_tool_gateway`
4. `_compile_tools()` wraps every `Tool(...)` via `gateway.wrap(tool)` → `GatedTool`
5. At runtime, every tool call passes through `GatedTool.apply()` → classify → check → execute/block

**Blocked tool calls return an error message** (not an exception), so the agent can see the block reason and adjust:
```
[BLOCKED] Tool 'shell': CRITICAL: rm -rf /
```

**Audit log**: Every tool call (allowed or blocked) is recorded with:
- timestamp, tool name, input preview, risk level, action taken, result preview, duration

**Confirmation flow** (when `highRiskConfirmation: true`):
- HIGH-risk calls trigger `policy.confirm_callback(tool, input, reason)`
- If callback returns `False` (or no callback set) → BLOCK
- Callback can be injected via `overrides["_confirm_callback"]` at compile time

### 3b. Guard Construct (Output Validation)

**Guard construct:**
- `Guard(agent, predicate, max_retries)` validates agent output
- SHOULD be used on all tool-calling agents in production
- Predicate functions MUST NOT have side effects
- `maxOutputLength` is now enforced: outputs exceeding the limit trigger retry or fallback truncation

**Tool timeout:**
- All Tool calls MUST have configurable timeout (default: 30s)
- Timeout MUST be enforced at process level, not just asyncio

**Dangerous commands (pattern library in `tool_gateway.py`):**
- **CRITICAL patterns** (22 rules): `rm -rf /`, `dd if=`, `mkfs`, fork bomb, `curl|sh`, credential access (`~/.ssh`, `~/.aws`, `/etc/shadow`), netcat listeners
- **HIGH patterns** (18 rules): `sudo`, `rm -r`, `chmod`, `kill -9`, `pip install`, `brew`, `docker rm`, `shutdown`, `git push --force`
- **MEDIUM patterns** (10 rules): `mv`, `cp -r`, `sed -i`, `git checkout/merge/rebase`, `tee /`
- All patterns compiled once at import time (`re.compile` with `IGNORECASE`)

### 4. Memory & State Security

**Memory construct:**
- `Memory(agent, store)` persists conversation state
- Memory stores MUST be scoped per agent/tenant (no cross-agent access)
- Memory size MUST be bounded (default: 20 entries, configurable)
- Sensitive data in memory SHOULD be encrypted at rest

**Checkpoint:**
- `checkpoint.py` serializes execution state to JSON
- Checkpoint files MUST NOT contain raw secrets
- Checkpoint files SHOULD have restricted file permissions (0600)

### 5. Multi-Agent Security

**Channel construct:**
- `Channel` enables inter-agent message passing
- Agents MUST only access channels they are explicitly connected to
- Channel messages SHOULD be validated by receiving agent's Guard

**GroupChat:**
- `GroupChat` runs multiple agents in a shared conversation
- Each agent MUST have its own Context (no shared mutable state)
- Moderator agent SHOULD enforce topic boundaries

**Handoff:**
- `Handoff` delegates to other agents at runtime
- Target agents MUST be from a pre-approved registry
- No dynamic import or arbitrary agent instantiation

### 6. LLM Provider Security

- API keys MUST be read from environment variables, never hardcoded
- Provider selection MUST be explicit (no auto-detection from untrusted input)
- LLM responses MUST be treated as untrusted input
- Token usage MUST be tracked and bounded per execution

### 7. Prompt Injection Defense

- System prompts SHOULD include injection resistance instructions
- Tool call parsing MUST use structured extraction (regex on JSON blocks)
- LLM output MUST be validated before execution (parse_and_execute pattern)
- `done` tool MUST be the only way to terminate ReAct loop

---

## Implementation Status

### Completed (2026-03-28)

| ID | Item | Implementation |
|----|------|---------------|
| LSEC-02 | macOS `RLIMIT_NPROC` fallback | `sandbox.py`: `ResourceLimiter._block_subprocess_python()` monkey-patches subprocess/os.exec* |
| LSEC-03 | Fix `pyproject.toml` build-backend | Fixed |
| LSEC-08 | Dangerous command blocklist | `tool_gateway.py`: 50+ regex patterns across CRITICAL/HIGH/MEDIUM levels |
| **LSEC-18** | **ToolGateway — runtime permission enforcement** | `tool_gateway.py`: `GatedTool` wraps all Tools, classifies risk, blocks/allows/confirms |
| **LSEC-19** | **guard.dangerousCommandBlock enforcement** | `compiler.py` → `_build_gateway()` → `_compile_tools()` wraps all Tools |
| **LSEC-20** | **guard.highRiskConfirmation enforcement** | `tool_gateway.py`: HIGH-risk calls routed to confirm_callback |
| **LSEC-21** | **guard.maxOutputLength enforcement** | `compiler.py` `_compile_guard()` now reads field; `GatedTool` truncates output |
| **LSEC-22** | **Tool call audit logging** | `tool_gateway.py` `AuditLog`: every call recorded with risk/action/duration |

### Gap Analysis — What Was Spec'd vs What Was Enforced

| YAML Field | Before (2026-03-27) | After (2026-03-28) |
|-----------|---------------------|---------------------|
| `guard.dangerousCommandBlock` | Declared, **never read** | Read by `_build_gateway()`, enforced via `GatedTool` |
| `guard.highRiskConfirmation` | Declared, **never read** | Read by `_build_gateway()`, triggers confirm flow |
| `guard.maxOutputLength` | Declared, **never read** | Read by `_compile_guard()` + `GatedTool.truncate_output()` |
| `guard.validator` | Read, enforced via `Guard` | Unchanged (was already working) |
| `guard.retry` | Read, enforced via `Guard` | Unchanged |
| `mcp.policy.mode` | Declared, **never read** | Still not enforced (only `retryOnFail` used) |
| `sandbox.allow_subprocess` (macOS) | Silent fail | Python-level monkey-patch fallback |

---

## TODO List

### P0 - Immediate (Critical/High)

- [x] **LSEC-01**: Fix sandbox code injection — pass input via JSON file, not string interpolation in `sandbox.py`
- [x] **LSEC-02**: Enforce resource limit warnings — macOS fallback via monkey-patch (2026-03-28)
- [x] **LSEC-03**: Fix `pyproject.toml` build-backend (DONE)

### P1 - Week 1 (High)

- [x] **LSEC-04**: Add security lint rules (S001-S003) to `fromconfig/lint.py`
- [x] **LSEC-05**: Add `maxSteps` upper bound validation in compiler (cap at 1000)
- [x] **LSEC-06**: Add Tool timeout enforcement at process level
- [x] **LSEC-07**: Validate tool references against whitelist in `from_config()`
- [x] **LSEC-08**: Add dangerous command blocklist to shell tool execution → `tool_gateway.py` (2026-03-28)
- [x] **LSEC-18**: ToolGateway runtime permission enforcement → `tool_gateway.py` (2026-03-28)
- [x] **LSEC-19**: guard.dangerousCommandBlock enforcement → `compiler.py` (2026-03-28)
- [x] **LSEC-20**: guard.highRiskConfirmation enforcement → `tool_gateway.py` (2026-03-28)
- [x] **LSEC-21**: guard.maxOutputLength enforcement → `compiler.py` + `tool_gateway.py` (2026-03-28)
- [x] **LSEC-22**: Tool call audit logging → `tool_gateway.py` (2026-03-28)
- [x] **LSEC-23**: Enforce `mcp.policy.mode` (auto/force/intelligence/disable) at runtime

### P2 - Month 1 (Medium)

- [ ] **LSEC-09**: Encrypt checkpoint files at rest
- [ ] **LSEC-10**: Scope Memory stores per agent/tenant
- [ ] **LSEC-11**: Add Channel access control (agent can only access declared channels)
- [x] **LSEC-12**: Add prompt injection resistance to default system prompts
- [ ] **LSEC-13**: Add token usage hard limits per execution context
- [ ] **LSEC-24**: ToolGateway path-based ACL — enforce `allowed_read_paths`/`allowed_write_paths` from SandboxPolicy
- [ ] **LSEC-25**: ToolGateway network ACL — block outbound network from tools when `SandboxPolicy.network=False`
- [ ] **LSEC-26**: Integrate ToolGateway audit log with AgentPaaS observability (OpenTelemetry spans)

### P3 - Ongoing (Low)

- [ ] **LSEC-14**: L2 container sandbox implementation
- [ ] **LSEC-15**: Formal verification of sandbox escape resistance
- [ ] **LSEC-16**: Fuzz testing for YAML config parser
- [ ] **LSEC-17**: Security audit of multi-agent constructs (Channel, GroupChat, AsyncPar)
- [ ] **LSEC-27**: ToolGateway rate limiting — cap tool calls per minute per agent
- [ ] **LSEC-28**: macOS sandbox-exec(1) integration for kernel-level process isolation
