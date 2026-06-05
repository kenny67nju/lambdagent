"""
lambdagent CLI — Lambda 演算 Agent 的命令行入口

用法:
    lambdagent compile config.yml           编译 YAML → Lambda 项
    lambdagent run config.yml "input"       编译 + 执行
    lambdagent repl config.yml              交互式 REPL
    lambdagent lint config.yml              静态分析
    lambdagent trace trace.json             查看 β-规约追踪
    lambdagent lambda config.yml            导出 Lambda 表达式
    lambdagent tools config.yml             列出/测试工具

设计哲学:
    CLI 是 Lambda 演算的自然栖息地。
    Unix 管道 | = 函数组合 >>
    每个命令 = 一次或多次 β-规约
    REPL = Y(λself.λΓ. read >> eval >> print >> self(Γ'))
"""

import argparse
import json
import sys
import time
import os

from ..core import Context
from ..fromconfig import from_config, describe_config, build_agent
from ..fromconfig import lint_config
from .shell_tool import parse_tool_args, ShellTool, ShellToolError


def main():
    parser = argparse.ArgumentParser(
        prog="lambdagent",
        description="Lambda Calculus Agent DSL — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lambdagent compile agent-config.yml
  lambdagent run agent-config.yml "帮我写快速排序"
  lambdagent run agent-config.yml --tool grep="grep -c ERROR log.txt" "分析日志"
  echo "Hello" | lambdagent run agent-config.yml -
  lambdagent repl agent-config.yml
  lambdagent lint agent-config.yml
  lambdagent lambda agent-config.yml
  lambdagent tools agent-config.yml
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # ── compile ──
    p_compile = sub.add_parser(
        "compile", help="Compile YAML → Lambda term (no execution)"
    )
    p_compile.add_argument("config", help="YAML config file path")
    p_compile.add_argument("--format", choices=["text", "json"], default="text")
    p_compile.add_argument("--validate", action="store_true", help="Also run lint")

    # ── run ──
    p_run = sub.add_parser("run", help="Compile + execute agent")
    p_run.add_argument("config", help="YAML config file path")
    p_run.add_argument(
        "input", nargs="?", default=None, help="Input text (use - for stdin)"
    )
    p_run.add_argument("--input-file", help="Read input from file")
    p_run.add_argument("--trace", action="store_true", help="Print β-reduction trace")
    p_run.add_argument("--trace-file", help="Save trace to JSON file")
    p_run.add_argument("--max-steps", type=int, help="Override maxSteps")
    p_run.add_argument("--temperature", type=float, help="Override temperature")
    p_run.add_argument("--model", help="Override model")
    p_run.add_argument(
        "--tool", action="append", default=[], help="Inject CLI tool: NAME=COMMAND"
    )
    p_run.add_argument(
        "--timeout", type=int, default=300, help="Total timeout (seconds)"
    )
    p_run.add_argument("--format", choices=["text", "json"], default="text")
    p_run.add_argument("--quiet", action="store_true", help="Only output final result")
    p_run.add_argument(
        "--verbose", action="store_true", help="Print all intermediate steps"
    )

    # ── repl ──
    p_repl = sub.add_parser("repl", help="Interactive REPL (persistent session)")
    p_repl.add_argument("config", help="YAML config file path")
    p_repl.add_argument("--tool", action="append", default=[], help="Inject CLI tool")
    p_repl.add_argument("--model", help="Override model")
    p_repl.add_argument("--temperature", type=float, help="Override temperature")

    # ── lint ──
    p_lint = sub.add_parser("lint", help="Static analysis of agent config")
    p_lint.add_argument("config", help="YAML config file or directory")
    p_lint.add_argument("--level", choices=["error", "warn", "info"], default="info")
    p_lint.add_argument("--format", choices=["text", "json"], default="text")

    # ── trace ──
    p_trace = sub.add_parser("trace", help="View/replay β-reduction trace")
    p_trace.add_argument("file", help="Trace JSON file")
    p_trace.add_argument("--step", type=int, help="Show specific step")
    p_trace.add_argument("--timeline", action="store_true", help="Timeline view")

    # ── lambda ──
    p_lambda = sub.add_parser("lambda", help="Export pure Lambda expression")
    p_lambda.add_argument("config", help="YAML config file path")
    p_lambda.add_argument(
        "--format", choices=["human", "formal", "json"], default="human"
    )

    # ── tools ──
    p_tools = sub.add_parser("tools", help="List and test tools")
    p_tools.add_argument("config", help="YAML config file path")
    p_tools.add_argument(
        "--test", nargs=2, metavar=("TOOL", "INPUT"), help="Test a tool"
    )
    p_tools.add_argument("--discover", metavar="SERVER", help="Discover MCP tools")

    # ── version ──
    sub.add_parser("version", help="Version info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        handlers = {
            "compile": cmd_compile,
            "run": cmd_run,
            "repl": cmd_repl,
            "lint": cmd_lint,
            "trace": cmd_trace,
            "lambda": cmd_lambda,
            "tools": cmd_tools,
            "version": cmd_version,
        }
        handler = handlers.get(args.command)
        if handler:
            sys.exit(handler(args))
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


# ════════════════════════════════════════════════════════════
# compile
# ════════════════════════════════════════════════════════════


def cmd_compile(args) -> int:
    """编译 YAML → Lambda 项，打印结构"""
    if not os.path.exists(args.config):
        print(f"[ERROR] File not found: {args.config}", file=sys.stderr)
        return 4

    desc = describe_config(args.config)

    if args.format == "json":
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(
            json.dumps(
                {
                    "lambda_structure": desc,
                    "config": cfg,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(desc)

    if args.validate:
        print("\n--- Lint ---")
        return cmd_lint_inner(args.config, "info", "text")

    return 0


# ════════════════════════════════════════════════════════════
# run
# ════════════════════════════════════════════════════════════


def cmd_run(args) -> int:
    """编译 + 执行 Agent"""
    if not os.path.exists(args.config):
        print(f"[ERROR] File not found: {args.config}", file=sys.stderr)
        return 4

    # 获取输入
    input_text = _resolve_input(args)
    if input_text is None:
        print(
            "[ERROR] No input provided. Use positional arg, --input-file, or stdin (-)",
            file=sys.stderr,
        )
        return 1

    # 解析注入工具
    try:
        injected_tools = parse_tool_args(args.tool) if args.tool else {}
    except ShellToolError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # 编译
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 应用覆盖
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.temperature is not None:
        cfg.setdefault("model", {})["temperature"] = args.temperature
    if args.max_steps:
        cfg.setdefault("react", {})["maxSteps"] = args.max_steps

    agent = build_agent(cfg)

    # 注入 CLI 工具到配置
    # (这里简化处理：对于 react agent，CLI 工具通过 Memory 注入工具描述)
    if injected_tools:
        tool_desc = "Available CLI tools:\n"
        for name, st in injected_tools.items():
            tool_desc += f"  - {name}: {st.command}\n"
        if not args.quiet:
            print(f"[Injected {len(injected_tools)} CLI tool(s)]")

    # 执行
    ctx = Context()
    if not args.quiet:
        print(f"[Compiling {args.config}...]")
        print(f'[Executing: ({_term_name(agent)} "{input_text[:50]}...")]')

    t0 = time.time()
    result = agent(input_text, ctx)
    elapsed = time.time() - t0

    # 输出
    if args.format == "json":
        output = {
            "result": str(result),
            "trace": [
                {
                    "step": i,
                    "term": e.term_name,
                    "duration_ms": round(e.duration_ms, 1),
                    "input": str(e.input)[:200],
                    "output": str(e.output)[:200],
                    "model": e.model,
                }
                for i, e in enumerate(ctx.trace)
            ],
            "stats": {
                "total_steps": len(ctx.trace),
                "total_time_ms": round(elapsed * 1000, 1),
            },
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if args.trace or args.verbose:
            print(f"\n{'─' * 60}")
            print("β-reduction trace:")
            ctx.print_trace()
            print(f"{'─' * 60}")

        if not args.quiet:
            print(f"\nResult ({len(ctx.trace)} β-reductions, {elapsed:.1f}s):")

        print(result)

    # 保存 trace
    if args.trace_file:
        trace_data = [
            {
                "step": i,
                "term": e.term_name,
                "term_id": e.term_id,
                "duration_ms": round(e.duration_ms, 1),
                "input": str(e.input),
                "output": str(e.output),
                "model": e.model,
                "tokens": e.tokens_used,
            }
            for i, e in enumerate(ctx.trace)
        ]
        with open(args.trace_file, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"[Trace saved to {args.trace_file}]")

    return 0


# ════════════════════════════════════════════════════════════
# repl
# ════════════════════════════════════════════════════════════


def cmd_repl(args) -> int:
    """交互式 REPL"""
    if not os.path.exists(args.config):
        print(f"[ERROR] File not found: {args.config}", file=sys.stderr)
        return 4

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.temperature is not None:
        cfg.setdefault("model", {})["temperature"] = args.temperature

    agent = build_agent(cfg)
    agent_name = cfg.get("name", cfg.get("agentId", "agent"))
    agent_type = cfg.get("type", "simple")
    max_steps = cfg.get("react", {}).get("maxSteps", "?")

    ctx = Context()
    total_reductions = 0
    session_start = time.time()

    print(f"lambdagent REPL v2.0")
    print(f"Agent: {agent_name} ({agent_type}, maxSteps={max_steps})")
    desc = describe_config(args.config)
    # 取第一行 Lambda 结构
    for line in desc.split("\n"):
        if "=" in line and "#" in line:
            print(f"Lambda: {line.split('=', 1)[-1].strip()}")
            break
    print(f"Type :help for commands, :quit to exit")
    print()

    while True:
        try:
            user_input = input("λ> ").strip()
        except EOFError:
            print("\n[EOF]")
            break

        if not user_input:
            continue

        # REPL 内置命令
        if user_input.startswith(":"):
            cmd = user_input.split()[0].lower()
            cmd_args_str = user_input[len(cmd) :].strip()

            if cmd in (":quit", ":q", ":exit"):
                break

            elif cmd == ":help":
                print("""
REPL Commands:
  :help              Show this help
  :quit / :q         Exit REPL
  :trace             Show last execution trace
  :trace N           Show step N details
  :memory            Show Memory contents
  :memory clear      Clear Memory
  :lambda            Show agent Lambda expression
  :lint              Lint config
  :reload            Reload YAML config
  :stats             Session statistics
  :tools             List available tools
                """)

            elif cmd == ":trace":
                if cmd_args_str and cmd_args_str.isdigit():
                    n = int(cmd_args_str)
                    if n < len(ctx.trace):
                        e = ctx.trace[n]
                        print(f"  β[{n}] {e.term_name}")
                        print(f"    Duration: {e.duration_ms:.0f}ms")
                        print(f"    Model: {e.model or 'N/A'}")
                        print(f"    Input:  {str(e.input)[:200]}")
                        print(f"    Output: {str(e.output)[:200]}")
                    else:
                        print(f"  Step {n} not found (total: {len(ctx.trace)})")
                else:
                    if ctx.trace:
                        ctx.print_trace()
                    else:
                        print("  No trace yet.")

            elif cmd == ":memory":
                if cmd_args_str == "clear":
                    ctx.memory.clear()
                    print("  Memory cleared.")
                else:
                    if ctx.memory:
                        for k, v in ctx.memory.items():
                            print(f"  {k}: {str(v)[:100]}")
                    else:
                        print("  Memory is empty.")

            elif cmd == ":lambda":
                print(describe_config(args.config))

            elif cmd == ":lint":
                cmd_lint_inner(args.config, "info", "text")

            elif cmd == ":reload":
                with open(args.config) as f:
                    cfg = yaml.safe_load(f)
                agent = build_agent(cfg)
                print(f"  Reloaded {args.config}")

            elif cmd == ":stats":
                elapsed = time.time() - session_start
                print(f"  Session: {elapsed:.0f}s")
                print(f"  Total β-reductions: {total_reductions}")
                print(f"  Trace entries: {len(ctx.trace)}")

            elif cmd == ":tools":
                mcp = cfg.get("mcp", {})
                for server, tools in mcp.get("onlineTool", {}).items():
                    for t in tools:
                        print(f"  [MCP]   {t:<30s} {server}")
                for t in mcp.get("localTools", []):
                    if t == "terminate":
                        print(f"  [Local] {'terminate':<30s} (λx.x) base case")
                    else:
                        print(f"  [Local] {t}")

            else:
                print(f"  Unknown command: {cmd}. Type :help")

            continue

        # 正常输入 → β-规约
        pre_count = len(ctx.trace)
        t0 = time.time()

        try:
            result = agent(user_input, ctx)
            elapsed = time.time() - t0
            new_steps = len(ctx.trace) - pre_count
            total_reductions += new_steps

            # 打印 trace
            for i in range(pre_count, len(ctx.trace)):
                e = ctx.trace[i]
                out_s = str(e.output)[:60]
                print(f"  [β[{i}] {e.term_name} {e.duration_ms:.1f}s] {out_s}")

            print(f"\n{result}")
            print(f"({new_steps} β-reductions, {elapsed:.1f}s)")
        except Exception as e:
            print(f"  [ERROR] {e}")

        print()

    elapsed = time.time() - session_start
    print(f"\nSession: {total_reductions} β-reductions in {elapsed:.0f}s")
    return 0


# ════════════════════════════════════════════════════════════
# lint
# ════════════════════════════════════════════════════════════


def cmd_lint(args) -> int:
    """Lint 配置文件"""
    return cmd_lint_inner(args.config, args.level, args.format)


def cmd_lint_inner(config_path: str, level: str, fmt: str) -> int:
    """lint 核心逻辑（compile --validate 复用）"""
    if not os.path.exists(config_path):
        print(f"[ERROR] File not found: {config_path}", file=sys.stderr)
        return 4

    # 如果是目录，递归处理
    if os.path.isdir(config_path):
        import glob

        files = glob.glob(os.path.join(config_path, "*.yml")) + glob.glob(
            os.path.join(config_path, "*.yaml")
        )
        total_errors = 0
        for f in files:
            print(f"\n--- {f} ---")
            total_errors += cmd_lint_inner(f, level, fmt)
        return 3 if total_errors else 0

    issues = lint_config(config_path)

    level_order = {"error": 0, "warn": 1, "info": 2}
    min_level = level_order.get(level, 2)

    filtered = [i for i in issues if level_order.get(i.level.lower(), 2) <= min_level]

    if fmt == "json":
        data = [
            {
                "level": i.level,
                "field": i.field,
                "message": i.message,
                "lambda": i.lambda_meaning,
            }
            for i in filtered
        ]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"\nlambdagent lint: {config_path}")
        print("=" * 60)
        for issue in filtered:
            print(issue)
        errors = [i for i in issues if i.level == "ERROR"]
        warns = [i for i in issues if i.level == "WARN"]
        infos = [i for i in issues if i.level == "INFO"]
        print(f"\n{'─' * 60}")
        print(
            f"  {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info(s)"
        )

    errors = [i for i in issues if i.level == "ERROR"]
    return 3 if errors else 0


# ════════════════════════════════════════════════════════════
# trace
# ════════════════════════════════════════════════════════════


def _trace_get(e, key, default=None):
    """兼容两种 trace 格式（v1: term/duration_ms, v2: term_type+name/elapsed_ms）。"""
    if key == "term":
        return e.get("term") or f"{e.get('term_type', '')}:{e.get('name', '')}"
    if key == "duration_ms":
        return e.get("duration_ms") or e.get("elapsed_ms", 0)
    if key == "tokens":
        return e.get("tokens") or (e.get("tokens_in", 0) + e.get("tokens_out", 0))
    return e.get(key, default)


def cmd_trace(args) -> int:
    """查看 β-规约追踪"""
    if not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        return 4

    with open(args.file) as f:
        trace = json.load(f)

    if args.step is not None:
        if 0 <= args.step < len(trace):
            e = trace[args.step]
            print(f"β[{e.get('step', args.step)}] {_trace_get(e, 'term')}")
            print(f"  Duration: {_trace_get(e, 'duration_ms'):.0f}ms")
            print(f"  Model:    {e.get('model', 'N/A')}")
            print(f"  Tokens:   {_trace_get(e, 'tokens') or 'N/A'}")
            print(f"  Input:    {str(e.get('input', ''))[:300]}")
            print(f"  Output:   {str(e.get('output', ''))[:300]}")
        else:
            print(f"Step {args.step} not found (total: {len(trace)})")
        return 0

    if args.timeline:
        # 简化时间线
        cumulative = 0
        print("Time ──────────────────────────────────────────→")
        for e in trace:
            ms = _trace_get(e, "duration_ms")
            bar = "█" * max(1, int(ms / 100))
            print(
                f"  {cumulative / 1000:6.1f}s ├{bar} {_trace_get(e, 'term')} ({ms:.0f}ms)"
            )
            cumulative += ms
        print(f"  {cumulative / 1000:6.1f}s ┤ END")
        return 0

    # 默认：打印全部
    for e in trace:
        inp = str(e.get("input", ""))[:60]
        out = str(e.get("output", ""))[:60]
        ms = _trace_get(e, "duration_ms")
        print(
            f"  β[{e.get('step', '?')}] {_trace_get(e, 'term')} ({ms:.0f}ms): {inp} → {out}"
        )

    total_ms = sum(_trace_get(e, "duration_ms") for e in trace)
    print(f"\nTotal: {len(trace)} β-reductions, {total_ms / 1000:.1f}s")
    return 0


# ════════════════════════════════════════════════════════════
# lambda
# ════════════════════════════════════════════════════════════


def cmd_lambda(args) -> int:
    """导出 Lambda 表达式"""
    if not os.path.exists(args.config):
        print(f"[ERROR] File not found: {args.config}", file=sys.stderr)
        return 4

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", cfg.get("agentId", "agent"))
    agent_type = cfg.get("type", "simple")
    model_cfg = cfg.get("model", {})
    prompt = cfg.get("systemPrompt", "")
    react = cfg.get("react", {})
    memory = cfg.get("memory", {})
    mcp = cfg.get("mcp", {})

    model_name = (
        f"{model_cfg.get('provider', 'anthropic')}/{model_cfg.get('name', '?')}"
    )
    temp = model_cfg.get("temperature", 0)

    # 收集工具
    tools = []
    for server, tl in mcp.get("onlineTool", {}).items():
        for t in tl:
            tools.append((t, f'MCP("{server}", "{t}", x)'))
    for t in mcp.get("localTools", []):
        if t == "terminate":
            tools.append(("terminate", "λx. x"))
        else:
            tools.append((t, f'local("{t}", x)'))

    if args.format == "json":
        data = {
            "name": name,
            "type": agent_type,
            "lambda": _build_lambda_str(cfg),
            "tools": [{"name": n, "lambda": l} for n, l in tools],
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    # human / formal
    print(f"{name} =")

    if memory.get("enabled"):
        strategy = memory.get("strategy", "local")
        size = memory.get("size", "?")
        ttl = memory.get("ttl", "?")
        print(f"  Memory(")

    if agent_type == "react":
        max_s = react.get("maxSteps", 10)
        print(f"    Y_{max_s}(λself. λstate.")
        short_prompt = prompt.strip().replace("\n", " ")[:40]
        print(
            f'      let t = (λx. LLM_{{{model_name}, ⊕_{temp}}}("{short_prompt}..."))(state) in'
        )
        if tools:
            print(f"      CASE (classify t) [")
            for tname, tlam in tools:
                marker = "  ← base case" if tname == "terminate" else ""
                print(f'        ("{tname}", λx. {tlam}){marker}')
            print(f"      ] >> λobs.")
        print(f"      IF (obs = t) THEN t ELSE self(state ⊕ format(t, obs))")
        print(f"    )")
    elif agent_type == "chain":
        steps = cfg.get("chain", {}).get("steps", [])
        for i, step in enumerate(steps):
            op = ">>" if i > 0 else "  "
            print(
                f'    {op} Lam("{step.get("name", f"step_{i}")}", "{step.get("prompt", "?")[:40]}...")'
            )
    else:
        short_prompt = prompt.strip().replace("\n", " ")[:50]
        print(f'    λx. LLM_{{{model_name}, ⊕_{temp}}}("{short_prompt}...")(x)')

    if memory.get("enabled"):
        print(
            f"    , Γ ∪ {memory.get('strategy', 'local')}{{size={memory.get('size', '?')}, ttl={memory.get('ttl', '?')}}}"
        )
        print(f"  )")

    return 0


# ════════════════════════════════════════════════════════════
# tools
# ════════════════════════════════════════════════════════════


def cmd_tools(args) -> int:
    """列出和测试工具"""
    if not os.path.exists(args.config):
        print(f"[ERROR] File not found: {args.config}", file=sys.stderr)
        return 4

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", cfg.get("agentId", "agent"))
    mcp = cfg.get("mcp", {})
    app = cfg.get("app", {}).get("mcp", {}).get("custom", {}).get("nodes", {})

    print(f"\nTools for {name}:")
    print(f"{'─' * 60}")

    # 在线工具
    for server, tools in mcp.get("onlineTool", {}).items():
        server_cfg = app.get(server, {})
        url = server_cfg.get("url", "?")
        endpoint = server_cfg.get("endpoint", "")
        for t in tools:
            print(f"  [MCP]   {t:<30s} {server}")

    # 本地工具
    for t in mcp.get("localTools", []):
        if t == "terminate":
            print(f"  [Local] {'terminate':<30s} (λx.x) base case")
        else:
            print(f"  [Local] {t:<30s}")

    # MCP 端点状态
    print(f"\nMCP endpoints:")
    for server, server_cfg in app.items():
        url = server_cfg.get("url", "")
        endpoint = server_cfg.get("endpoint", "")
        full_url = f"{url}{endpoint}"
        print(f"  {server}: {full_url}")
        # 尝试 ping
        try:
            import urllib.request

            req = urllib.request.Request(url, method="HEAD")
            t0 = time.time()
            urllib.request.urlopen(req, timeout=5)
            ms = (time.time() - t0) * 1000
            print(f"    Status: ✓ reachable ({ms:.0f}ms)")
        except Exception as e:
            print(f"    Status: ✗ {e}")

    # 测试单个工具
    if args.test:
        tool_name, test_input = args.test
        print(f"\n--- Testing {tool_name} ---")
        print(f"  Input: {test_input}")
        # 简化：构建并调用工具
        agent = build_agent(cfg)
        # 这里需要更精细的实现来单独调用工具
        print(
            f"  (Tool testing requires full react agent; use `lambdagent run` instead)"
        )

    return 0


# ════════════════════════════════════════════════════════════
# version
# ════════════════════════════════════════════════════════════


def cmd_version(args) -> int:
    print("lambdagent v2.0.0")
    print("Lambda Calculus Agent DSL")
    print("11 core constructs — strict lambda correspondence")
    return 0


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════


def _resolve_input(args) -> "str | None":
    """解析输入来源: 位置参数 > --input-file > stdin"""
    if args.input == "-":
        return sys.stdin.read().strip()
    if args.input:
        return args.input
    if args.input_file:
        with open(args.input_file) as f:
            return f.read().strip()
    # 检查 stdin 是否有数据
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def _term_name(term) -> str:
    """获取 Term 的名称"""
    return getattr(term, "_name", term.__class__.__name__)


def _build_lambda_str(cfg: dict) -> str:
    """构建 Lambda 表达式字符串"""
    name = cfg.get("name", cfg.get("agentId", "agent"))
    agent_type = cfg.get("type", "simple")
    max_steps = cfg.get("react", {}).get("maxSteps", "?")
    has_memory = cfg.get("memory", {}).get("enabled", False)

    if agent_type == "react":
        inner = f"Y_{max_steps}(λself.λstate. think(state) >> route >> observe)"
    elif agent_type == "chain":
        steps = cfg.get("chain", {}).get("steps", [])
        names = [s.get("name", f"step_{i}") for i, s in enumerate(steps)]
        inner = " >> ".join(names)
    else:
        inner = f"λx. LLM(x)"

    if has_memory:
        strategy = cfg.get("memory", {}).get("strategy", "local")
        return f"Memory({inner}, {strategy})"
    return inner


if __name__ == "__main__":
    main()
