"""
示例 8: MCP Client + Checkpoint — P0 功能演示

展示:
  1. MCP Client — 模拟 MCP Server 工具发现和调用
  2. Checkpoint — 长时任务的断点保存/恢复/回退
  3. 组合案例 — MCP 工具 + Checkpoint 在研究管道中的协作

运行:
    python -m lambdagent.examples.ex08_mcp_checkpoint
"""

import sys
import os
import time
import json
import tempfile
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lambdagent import (
    Tool, Context, Lam,
    MCPServer, MCPTool,
    Checkpoint, CheckpointManager, save_context, load_context,
    SharedMemory,
)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  案例: {title}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# 辅助: 模拟 MCP Server（本地 HTTP）
# ══════════════════════════════════════════════════════════════

class MockMCPHandler(BaseHTTPRequestHandler):
    """模拟 MCP Server，提供 3 个工具"""

    TOOLS = {
        "web_search": {
            "name": "web_search",
            "description": "Search the web for information",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        "calculator": {
            "name": "calculator",
            "description": "Evaluate math expressions",
            "inputSchema": {"type": "object", "properties": {"expression": {"type": "string"}}},
        },
        "translator": {
            "name": "translator",
            "description": "Translate text between languages",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}},
        },
    }

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode())
        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id", "1")

        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {}, "serverInfo": {"name": "mock-mcp"}}
        elif method == "tools/list":
            result = {"tools": list(self.TOOLS.values())}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            result = self._call_tool(tool_name, args)
        else:
            result = {}

        resp = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def _call_tool(self, name, args):
        if name == "web_search":
            query = args.get("query", args.get("input", ""))
            return {"content": [{"type": "text", "text":
                f"Search results for '{query}':\n"
                f"1. Wikipedia: {query} is a widely studied topic\n"
                f"2. ArXiv: Recent advances in {query}\n"
                f"3. GitHub: Open source projects related to {query}"}]}
        elif name == "calculator":
            expr = args.get("expression", args.get("input", ""))
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return {"content": [{"type": "text", "text": f"{expr} = {result}"}]}
            except:
                return {"content": [{"type": "text", "text": f"Cannot evaluate: {expr}"}]}
        elif name == "translator":
            text = args.get("text", args.get("input", ""))
            lang = args.get("target_lang", "Chinese")
            return {"content": [{"type": "text", "text": f"[{lang}] Translated: {text}"}]}
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

    def log_message(self, format, *args):
        pass


def start_mock_mcp(port=19876):
    """启动模拟 MCP Server"""
    server = HTTPServer(("127.0.0.1", port), MockMCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    return server


# ══════════════════════════════════════════════════════════════
# 案例 1: MCP Client — 工具发现与调用
# ══════════════════════════════════════════════════════════════

def demo_mcp_client():
    """
    MCP Client: 连接 MCP Server，自动发现工具，封装为 lambdagent Term。

    Lambda 语义:
        MCPServer(url) = Γ_mcp : tool_name → Tool(name, λx. mcp_call(server, name, x))
        MCPTool = tool[f] where f(x) = JSON-RPC tools/call
    """
    separator("1. MCP Client — 工具发现与调用")

    # 启动模拟 MCP Server
    mock_server = start_mock_mcp(19876)
    print("  [Mock MCP Server] 已启动 (localhost:19876)")

    try:
        # 连接
        server = MCPServer.http("http://127.0.0.1:19876", name="mock-mcp")
        info = server.initialize()
        print(f"  [连接成功] {server}")
        print(f"  Protocol: {info.get('protocolVersion', '?')}")

        # 发现工具
        tools = server.list_tools()
        print(f"\n  [工具发现] {len(tools)} 个工具:")
        for t in tools:
            print(f"    • {t.name}: {t.description}")

        # 转为 lambdagent Term
        search_tool = server.to_tool("web_search")
        calc_tool = server.to_tool("calculator")
        trans_tool = server.to_tool("translator")

        print(f"\n  [转为 Term]")
        print(f"    search: {search_tool}")
        print(f"    calc:   {calc_tool}")
        print(f"    trans:  {trans_tool}")

        # 调用 = β-规约
        ctx = Context()

        print(f"\n  [β-规约] MCP 工具调用:")
        r1 = search_tool("Lambda calculus", ctx)
        print(f"    search('Lambda calculus'):")
        print(f"      {r1[:80]}...")

        r2 = calc_tool("2**10 + 42", ctx)
        print(f"    calc('2**10 + 42'):")
        print(f"      {r2}")

        r3 = trans_tool(json.dumps({"text": "Hello World", "target_lang": "Chinese"}), ctx)
        print(f"    translator('Hello World' → Chinese):")
        print(f"      {r3}")

        # 管道组合: MCP 工具参与 >>
        pipeline = search_tool >> Tool("extract_first", lambda x: x.split("\n")[1] if "\n" in x else x)
        r4 = pipeline("AI agents", ctx)
        print(f"\n  [管道] search >> extract_first:")
        print(f"    {r4}")

        # Route: MCP 工具作为路由目标
        route_dict = server.to_route_dict()
        print(f"\n  [Route dict] {list(route_dict.keys())}")

        print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
        ctx.print_trace()

    finally:
        mock_server.shutdown()

    print("  ✅ MCP Client 完成")


# ══════════════════════════════════════════════════════════════
# 案例 2: Checkpoint — 断点保存/恢复
# ══════════════════════════════════════════════════════════════

def demo_checkpoint_basic():
    """
    Checkpoint: 序列化 Context 到 JSON，恢复后继续执行。

    Lambda 语义:
        save(Γ, trace) → JSON    序列化环境
        load(JSON) → (Γ, trace)  反序列化环境
        resume = agent(input) [Γ_restored]
    """
    separator("2. Checkpoint — 断点保存/恢复")

    tmpdir = tempfile.mkdtemp()
    try:
        # ── Phase 1: 执行前半段，保存 checkpoint ──
        print("  [Phase 1] 执行前半段...")
        ctx = Context()
        ctx.memory["session_id"] = "demo_001"

        step1 = Tool("research", lambda x: f"研究结果: {x} 有 3 个关键发现")
        step2 = Tool("analyze", lambda x: f"分析: {x} → 趋势向上, 置信度 85%")

        r1 = step1("AI Agent 市场", ctx)
        r2 = step2(r1, ctx)

        print(f"    Step 1: {r1[:50]}...")
        print(f"    Step 2: {r2[:50]}...")
        print(f"    追踪: {len(ctx.trace)} 步")

        # 保存
        cp_path = save_context(ctx, os.path.join(tmpdir, "mid_checkpoint.json"),
                               last_input=r2, description="研究分析完成，待综合")
        print(f"\n  [Save] → {os.path.basename(cp_path)}")

        # 查看 checkpoint 内容
        cp = Checkpoint.load(cp_path)
        print(f"  {cp.summary()}")

        # ── Phase 2: 模拟"重启"——加载 checkpoint 继续 ──
        print(f"\n  [Phase 2] 模拟重启，加载 checkpoint...")
        ctx2 = load_context(cp_path)
        print(f"    恢复追踪: {len(ctx2.trace)} 步")
        print(f"    恢复记忆: session_id={ctx2.memory.get('session_id', '?')}")

        # 在恢复的 context 上继续
        step3 = Tool("synthesize", lambda x: f"综合报告: 基于分析结果，建议投资。ROI 预估 3.2x")
        r3 = step3(cp.last_input, ctx2)
        print(f"    Step 3 (续): {r3}")
        print(f"    总追踪: {len(ctx2.trace)} 步 (2 恢复 + 1 新增)")

        assert len(ctx2.trace) == 3
        assert ctx2.trace[0].term_name == "research"
        assert ctx2.trace[2].term_name == "synthesize"

    finally:
        shutil.rmtree(tmpdir)

    print("  ✅ Checkpoint 基础功能完成")


# ══════════════════════════════════════════════════════════════
# 案例 3: CheckpointManager — 多版本管理
# ══════════════════════════════════════════════════════════════

def demo_checkpoint_manager():
    """
    CheckpointManager: 自动保存多个版本，支持回退。

    用法:
        mgr = CheckpointManager("./checkpoints/my_agent")
        mgr.save(ctx, "after research")
        mgr.save(ctx, "after analysis")
        mgr.rollback()  # 回退到 "after research"
    """
    separator("3. CheckpointManager — 多版本管理与回退")

    tmpdir = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(os.path.join(tmpdir, "research_agent"), max_checkpoints=5)
        ctx = Context()

        # 模拟多步执行，每步保存
        steps = [
            ("search", "搜索 AI Agent 论文"),
            ("filter", "筛选: 保留 top 10 相关论文"),
            ("summarize", "摘要: 10 篇论文的核心贡献"),
            ("critique", "批判: 发现 3 个方法论问题"),
            ("revise", "修订: 根据批评意见重新分析"),
        ]

        for name, output in steps:
            tool = Tool(name, lambda x, o=output: o)
            tool(f"input for {name}", ctx)
            path = mgr.save(ctx, description=f"After {name}", last_input=f"input for {name}")
            print(f"  [{name}] saved → {os.path.basename(path)}")

        # 列出所有 checkpoint
        print(f"\n  所有 checkpoint:")
        for item in mgr.list():
            print(f"    {item['file']}: {item['description']} ({item['steps']} steps)")

        # 获取最新
        latest = mgr.latest()
        print(f"\n  最新: {latest.description} ({latest.step_count} steps)")

        # 回退
        print(f"\n  [回退] 撤销最后一步 (revise)...")
        rolled_ctx = mgr.rollback()
        print(f"    回退后: {len(rolled_ctx.trace)} steps")
        print(f"    剩余 checkpoint: {len(mgr.list())}")

        # 再回退
        print(f"  [再次回退] 撤销 critique...")
        rolled_ctx2 = mgr.rollback()
        print(f"    回退后: {len(rolled_ctx2.trace)} steps")

        assert len(rolled_ctx.trace) == 4   # search + filter + summarize + critique
        assert len(rolled_ctx2.trace) == 3  # search + filter + summarize

    finally:
        shutil.rmtree(tmpdir)

    print("  ✅ CheckpointManager 完成")


# ══════════════════════════════════════════════════════════════
# 案例 4: Checkpoint + SharedMemory
# ══════════════════════════════════════════════════════════════

def demo_checkpoint_shared():
    """
    Checkpoint 与 SharedMemory 联合保存/恢复。

    场景: 多 Agent 共享记忆的研究系统，中途断点，恢复后共享数据还在。
    """
    separator("4. Checkpoint + SharedMemory 联合保存")

    tmpdir = tempfile.mkdtemp()
    try:
        # Phase 1: 执行并保存
        shared = SharedMemory({"project": "LLM Agent 研究"})
        ctx = Context()

        # Agent A: 数据采集
        collector = Tool("collect", lambda x: "market_data_collected")
        collector("采集数据", ctx)
        shared.write("data_points", 156)
        shared.write("last_update", "2026-03-26")

        # Agent B: 初步分析
        analyst = Tool("analyze", lambda x: "preliminary_analysis_done")
        analyst("分析数据", ctx)
        shared.write("trend", "上升")

        print("  [Phase 1] 执行完成:")
        print(f"    追踪: {len(ctx.trace)} 步")
        print(f"    共享记忆: {shared.read_all()}")

        # 保存（含 SharedMemory）
        from lambdagent.checkpoint import save_context_with_shared
        cp_path = save_context_with_shared(
            ctx,
            shared_memories={"research": shared.read_all()},
            path=os.path.join(tmpdir, "with_shared.json"),
            last_input="分析数据",
            description="采集+分析完成",
        )
        print(f"  [Save] → {os.path.basename(cp_path)}")

        # Phase 2: 模拟重启
        print(f"\n  [Phase 2] 模拟重启...")
        cp = Checkpoint.load(cp_path)
        ctx2 = cp.context
        restored_shared = SharedMemory(cp.shared_data.get("research", {}))

        print(f"    恢复追踪: {len(ctx2.trace)} 步")
        print(f"    恢复共享记忆: {restored_shared.read_all()}")

        assert restored_shared.read("data_points") == 156
        assert restored_shared.read("trend") == "上升"

        # 继续执行
        reporter = Tool("report", lambda x: f"报告: 基于 {restored_shared.read('data_points')} 个数据点，趋势{restored_shared.read('trend')}")
        r = reporter("生成报告", ctx2)
        print(f"    Step 3 (续): {r}")
        print(f"    总追踪: {len(ctx2.trace)} 步")

    finally:
        shutil.rmtree(tmpdir)

    print("  ✅ Checkpoint + SharedMemory 完成")


# ══════════════════════════════════════════════════════════════
# 案例 5: MCP + Checkpoint 综合管道
# ══════════════════════════════════════════════════════════════

def demo_mcp_checkpoint_pipeline():
    """
    综合案例: MCP 工具 + Checkpoint 在研究管道中协作。

    管道:
        1. MCP search → 搜索资料
        2. Checkpoint 保存中间状态
        3. MCP calculator → 数据计算
        4. Checkpoint 保存
        5. 模拟中断 → 从 checkpoint 恢复
        6. 继续执行 → 完成
    """
    separator("5. MCP + Checkpoint 综合管道")

    mock_server = start_mock_mcp(19877)
    tmpdir = tempfile.mkdtemp()

    try:
        server = MCPServer.http("http://127.0.0.1:19877", name="research-mcp")
        server.initialize()
        search = server.to_tool("web_search")
        calc = server.to_tool("calculator")

        mgr = CheckpointManager(os.path.join(tmpdir, "pipeline"))
        ctx = Context()

        # Step 1: 搜索
        print("  [Step 1] MCP 搜索...")
        r1 = search("LLM Agent frameworks 2026", ctx)
        print(f"    结果: {r1[:60]}...")
        mgr.save(ctx, "after search", last_input="LLM Agent frameworks 2026")

        # Step 2: 计算
        print("  [Step 2] MCP 计算...")
        r2 = calc("15.6 * 1.34", ctx)
        print(f"    计算: {r2}")
        mgr.save(ctx, "after calculation")

        # Step 3: 模拟中断
        print("\n  [模拟中断] 保存了 2 个 checkpoint...")
        for item in mgr.list():
            print(f"    {item['file']}: {item['description']}")

        # Step 4: 恢复
        print("\n  [恢复] 从最新 checkpoint 继续...")
        cp = mgr.latest()
        ctx_restored = cp.context
        print(f"    恢复: {len(ctx_restored.trace)} 步追踪")

        # Step 5: 继续执行
        final = Tool("report", lambda x: f"最终报告: LLM Agent 市场预计 {r2}。研究完成。")
        r_final = final("生成报告", ctx_restored)
        print(f"  [Step 5] {r_final}")

        print(f"\n  完整追踪 ({len(ctx_restored.trace)} 步):")
        ctx_restored.print_trace()

    finally:
        mock_server.shutdown()
        shutil.rmtree(tmpdir)

    print("  ✅ MCP + Checkpoint 综合管道完成")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  lambdagent P0: MCP Client + Checkpoint 演示                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demos = [
        ("MCP Client 工具发现与调用", demo_mcp_client),
        ("Checkpoint 断点保存/恢复", demo_checkpoint_basic),
        ("CheckpointManager 多版本管理", demo_checkpoint_manager),
        ("Checkpoint + SharedMemory", demo_checkpoint_shared),
        ("MCP + Checkpoint 综合管道", demo_mcp_checkpoint_pipeline),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  全部 5 个 P0 案例完成")
    print("=" * 70)
