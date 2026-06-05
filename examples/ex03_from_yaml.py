"""
示例 3: 从 agent-config.yml 构建 Agent

演示核心论点：YAML 配置文件本质上是一个 Lambda 表达式的声明式序列化。

    agent-config.yml
         ↓  解析
    Lambda 项（lambdagent Term）
         ↓  调用
    β-规约链（执行）

用法：
    python examples/ex03_from_yaml.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambdagent import from_config, describe_config, Context


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent-cofig.yml"
    )

    # ════════════════════════════════════════════
    # Part 1: 将 YAML 解读为 Lambda 表达式
    # ════════════════════════════════════════════

    print("=" * 60)
    print("agent-config.yml → Lambda 演算解读")
    print("=" * 60)
    print()
    print(describe_config(config_path))

    # ════════════════════════════════════════════
    # Part 2: 展示 YAML 字段与 Lambda 的对应
    # ════════════════════════════════════════════

    print()
    print("=" * 60)
    print("YAML 字段 → Lambda 演算 对应表")
    print("=" * 60)

    mapping = [
        ("model.name", "qwen3-max", "LLM 计算单元 (θ, V, c)", "Lam(model=...)"),
        ("systemPrompt", '"你是 SeeCoder..."', "λ 抽象的 body", "Lam(prompt=...)"),
        ("temperature: 0.7", "0.7", "概率 Lambda 参数 ⊕_p", "Lam(temperature=0.7)"),
        ("maxTokens: 4096", "4096", "输出长度界", "Lam(max_tokens=4096)"),
        ("type: react", "react", "Loop(think>>act>>observe)", "Y 组合子"),
        ("react.maxSteps: 20", "20", "Y 组合子最大展开次数", "Loop(max_steps=20)"),
        ("mcp.onlineTool", "example-mcp-server", "外部 Oracle", 'Tool("mcp", fn)'),
        ("mcp.localTools", "[terminate]", "base case (λx.x)", 'Tool("terminate", id)'),
        ("memory.strategy", "redis", "环境扩展 Γ' = Γ ∪ store", "Memory(agent, store)"),
        ("memory.size: 20", "20", "环境容量", "Memory.store 大小"),
        ("memory.ttl: 7200", "7200s", "绑定存活时间", "变量生命周期"),
        ("rag.enabled", "false", "外部存储（无界纸带）", "未启用"),
    ]

    print(f"{'YAML 字段':<25s} {'值':<20s} {'Lambda 演算':<25s} {'DSL 构造':<25s}")
    print("─" * 95)
    for field, val, lambda_meaning, dsl in mapping:
        print(f"{field:<25s} {val:<20s} {lambda_meaning:<25s} {dsl:<25s}")

    # ════════════════════════════════════════════
    # Part 3: 构建可执行 Agent
    # ════════════════════════════════════════════

    print()
    print("=" * 60)
    print("构建可执行 Agent")
    print("=" * 60)

    agent = from_config(config_path)
    print(f"Agent 类型: {agent.__class__.__name__}")
    print(f"Agent 名称: {agent._name}")

    # ════════════════════════════════════════════
    # Part 4: 用本地 Tool 模拟执行（不需要 API key）
    # ════════════════════════════════════════════

    print()
    print("=" * 60)
    print("模拟执行（使用本地 Tool，不调用外部 API）")
    print("=" * 60)

    # 构建一个纯本地版本来演示 β-规约链
    from lambdagent import Lam, Tool, Loop, Memory, Compose

    # 模拟 think：直接返回一个"思考"
    think = Tool(
        "think",
        lambda x: f"[思考] 分析问题: {x[:50]}... → 需要调用 everything_get_sum 工具",
    )
    # 模拟 act：工具调用
    act = Tool(
        "act",
        lambda x: (
            f"[行动] 调用 example-mcp-server/everything_get_sum → 获得结果: sum=42"
        ),
    )
    # 模拟 observe：观察结果
    observe = Tool(
        "observe", lambda x: f"[观察] 结果: 42。问题已解决，调用 terminate。"
    )

    # ReAct = Loop(think >> act >> observe)
    react_agent = Loop(
        think >> act >> observe,
        condition=lambda r, step: "terminate" in r.lower() or step >= 2,
        max_steps=3,
    )

    # 包装 Memory
    agent_with_memory = Memory(
        react_agent,
        store={
            "strategy": "redis",
            "size": 20,
            "ttl": 7200,
        },
    )

    ctx = Context()
    result = agent_with_memory("帮我计算 1+2+3+...+100 的和", ctx)

    print(f"\n输入: 帮我计算 1+2+3+...+100 的和")
    print(f"输出: {result}")

    print(f"\nβ-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()

    # ════════════════════════════════════════════
    # Part 5: 形式化对应总结
    # ════════════════════════════════════════════

    print()
    print("=" * 60)
    print("形式化对应")
    print("=" * 60)
    print("""
agent-config.yml 是一个 Lambda 项的声明式序列化：

  SeeCoderManus = Memory(                     ← memory: {strategy: redis}
      Loop(                                    ← type: react
          think >> act >> observe,              ← ReAct 三步循环
          max_steps = 20                        ← react.maxSteps: 20
      ),
      store = redis(size=20, ttl=7200)         ← memory: {size: 20, ttl: 7200}
  )

  where:
      think   = Lam("SeeCoderManus",           ← agentId + systemPrompt
                    prompt = "你是 SeeCoderManus...",
                    model = dashscope/qwen3-max,  ← model: {provider, name}
                    temperature = 0.7)             ← model.temperature

      act     = Route(think_output, {           ← mcp 工具列表
                    "everything_get_sum": Tool(MCP),
                    "chat_improve_prompt": Tool(MCP),
                    "terminate": Tool(λx.x)      ← base case
                })

  等价的 lambdagent DSL 代码：

      from lambdagent import Lam, Tool, Loop, Memory, Route

      think = Lam("SeeCoderManus", system_prompt, model="qwen3-max", temperature=0.7)
      tools = {
          "everything_get_sum": Tool("sum", mcp_call),
          "chat_improve_prompt": Tool("improve", mcp_call),
          "terminate": Tool("terminate", lambda x: x),
      }
      react = Loop(think >> Route(think, tools), max_steps=20)
      agent = Memory(react, store={"strategy": "redis", "size": 20})
""")


if __name__ == "__main__":
    main()
