# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email **qinliu@nju.edu.cn** with subject line starting `[lambdagent-security]`.

I will respond within 7 days and coordinate disclosure.

You can also use GitHub's [Private Vulnerability Reporting](https://github.com/kenny67nju/lambdagent/security/advisories/new).

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ active |
| < 0.1   | ❌        |

## Scope

In scope:
- `lambdagent.fromconfig.compiler` YAML → Term compiler (including the `guard.validator` safe-eval sandbox)
- `lambdagent.sandbox` process isolation and resource limits
- `lambdagent.tool_gateway` permission gateway
- `lambdagent.builtin_tools.shell_tools` / `code_tools` (any command-injection vector)
- `lambdagent.mcp_client` / `mcp_server` (RPC parsing, transport handling)
- `lambdagent.providers.*` (API key handling, credential leakage)

Out of scope:
- Vulnerabilities in third-party LLM provider APIs (report to those providers)
- Vulnerabilities in optional MCP servers users connect to
- Issues that require physical or local-file access to the developer's machine
- DoS via resource exhaustion when the user explicitly disables sandbox limits

See `SECURITYSPEC.md` for the detailed security model.
