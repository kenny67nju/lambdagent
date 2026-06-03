"""
示例 9: A2A Protocol + RAG — P1 功能演示

展示:
  1. A2A Agent Card — Skill → Agent Card 发布
  2. A2A Server/Client — Agent 间 HTTP 通信
  3. RAG 知识库 — 零依赖向量检索
  4. AgenticRAG — Agent 自主决定是否检索
  5. 综合案例 — A2A + RAG + Skills 协作

运行:
    python -m lambdagent.examples.ex09_a2a_rag
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lambdagent import (
    Tool, Context,
    Skill, SkillRegistry, skill, SkillAgent,
    AgentCard, A2AServer, A2AClient,
    skill_to_agent_card, registry_to_agent_card,
    RAGTool, AgenticRAG, SimpleVectorStore, create_rag,
)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  案例: {title}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# 案例 1: A2A Agent Card 发布
# ══════════════════════════════════════════════════════════════

def demo_agent_card():
    """
    将 lambdagent Skill 发布为 A2A Agent Card。

    A2A Agent Card 是 JSON 格式的能力描述文档，
    让其他 A2A 兼容的 Agent 能发现和调用你的 Agent。
    """
    separator("1. A2A Agent Card 发布")

    SkillRegistry().clear()

    # 创建技能
    @skill("code_review", "Review code for bugs and improvements",
           tags=["code", "review"], version="2.0.0", author="lambdagent-team")
    def code_review(x):
        return f"Review of code:\n- No critical bugs found\n- Suggest: add type hints\n- Complexity: moderate"

    @skill("code_explain", "Explain code in plain language",
           tags=["code", "education"])
    def code_explain(x):
        return f"This code does: {x[:30]}... It's a function that processes input data."

    # Skill → AgentCard
    card = skill_to_agent_card(code_review, url="https://my-agent.example.com")
    print("  [单技能 Agent Card]")
    print(f"    Name: {card.name}")
    print(f"    Description: {card.description}")
    print(f"    URL: {card.url}")
    print(f"    Skills: {len(card.skills)}")
    print(f"    Tags: {card.tags}")
    print(f"    Lambda type: {card.lambda_type}")

    # AgentCard JSON
    card_json = card.to_dict()
    print(f"\n  [Agent Card JSON]")
    print(f"    {json.dumps(card_json, indent=2, ensure_ascii=False)[:300]}...")

    # 验证 x-lambdagent 扩展
    ext = card_json.get("x-lambdagent", {})
    assert ext["framework"] == "lambdagent"
    print(f"\n  [x-lambdagent 扩展] framework={ext['framework']}")

    # Registry → AgentCard（多技能）
    reg = SkillRegistry()
    multi_card = registry_to_agent_card(reg, url="https://api.example.com/agents/code")
    print(f"\n  [多技能 Agent Card]")
    print(f"    Name: {multi_card.name}")
    print(f"    Skills: {len(multi_card.skills)}")
    for s in multi_card.skills:
        print(f"      • {s['name']}: {s['description'][:40]}...")

    print("  ✅ Agent Card 发布完成")


# ══════════════════════════════════════════════════════════════
# 案例 2: A2A Server/Client 通信
# ══════════════════════════════════════════════════════════════

def demo_a2a_communication():
    """
    A2A 端到端通信: Server 发布 Agent → Client 发现并调用。

    Lambda 语义:
        A2AServer(agent) = HTTP 服务器，每个 request = 一次远程 β-规约
        A2AClient(url)   = Tool(name, λx. http_call(url, x))  远程 Term
    """
    separator("2. A2A Server/Client 通信")

    # 创建 Agent
    agent = Tool("math_tutor", lambda x:
        f"Math Tutor says: The answer to '{x}' involves basic arithmetic. "
        f"Let me explain step by step...")

    # 发布为 A2A Server
    card = AgentCard(
        name="math-tutor",
        description="A friendly math tutor that explains concepts step by step",
        skills=[{
            "id": "math_101",
            "name": "basic_math",
            "description": "Explain and solve basic math problems",
            "tags": ["math", "education"],
        }],
    )

    server = A2AServer(agent, card=card, port=19878)
    server.start(background=True)
    time.sleep(0.3)

    try:
        # Client 发现
        client = A2AClient("http://127.0.0.1:19878")
        remote_card = client.card
        print(f"  [发现] Agent Card:")
        print(f"    Name: {remote_card.name}")
        print(f"    Skills: {[s['name'] for s in remote_card.skills]}")

        # Client 调用（远程 β-规约）
        ctx = Context()
        r1 = client("What is 2+2?", ctx)
        r2 = client("Explain the Pythagorean theorem", ctx)

        print(f"\n  [远程调用]")
        print(f"    Q: What is 2+2?")
        print(f"    A: {r1[:60]}...")
        print(f"    Q: Pythagorean theorem?")
        print(f"    A: {r2[:60]}...")

        # 与本地 Agent 组合（本地 >> 远程 >> 本地）
        local_pre = Tool("format_question", lambda x: f"Please explain: {x}")
        local_post = Tool("add_emoji", lambda x: f"📚 {x}")

        pipeline = local_pre >> client >> local_post
        r3 = pipeline("What is calculus?", ctx)
        print(f"\n  [管道] local >> remote >> local:")
        print(f"    {r3[:70]}...")

        print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
        ctx.print_trace()

    finally:
        server.stop()

    print("  ✅ A2A 通信完成")


# ══════════════════════════════════════════════════════════════
# 案例 3: RAG 知识库检索
# ══════════════════════════════════════════════════════════════

def demo_rag():
    """
    RAG: 从知识库检索相关文档，增强 Agent 的回答。

    Lambda 语义:
        RAGTool(store, k) = Tool("rag", λx. top_k(store, x))
        = 检索是一个 Tool，可以参与 >> 管道
    """
    separator("3. RAG 知识库检索")

    # 构建知识库
    store = SimpleVectorStore()
    documents = [
        ("Lambda calculus was invented by Alonzo Church in the 1930s as a formal system for expressing computation.",
         {"source": "Wikipedia", "topic": "lambda calculus"}),
        ("The Y combinator Y = λf.(λx.f(x x))(λx.f(x x)) enables recursion in lambda calculus without named functions.",
         {"source": "SICP", "topic": "Y combinator"}),
        ("Chain-of-Thought (CoT) prompting was introduced by Wei et al. 2022, showing that intermediate reasoning steps improve LLM performance.",
         {"source": "NeurIPS 2022", "topic": "CoT"}),
        ("The Model Context Protocol (MCP) is an open standard for connecting AI models to external tools and data sources.",
         {"source": "Anthropic", "topic": "MCP"}),
        ("ReAct agents combine reasoning and acting by interleaving thought and action steps in a loop.",
         {"source": "ICLR 2023", "topic": "ReAct"}),
        ("Type safety in programming languages ensures that well-typed programs don't go wrong at runtime.",
         {"source": "Milner 1978", "topic": "type theory"}),
        ("The terminate tool in a ReAct agent is the identity function λx.x, serving as the base case of the Y combinator.",
         {"source": "lambdagent paper", "topic": "lambdagent"}),
        ("Agent-to-Agent (A2A) protocol enables interoperability between AI agents from different frameworks.",
         {"source": "Google 2025", "topic": "A2A"}),
    ]

    for text, meta in documents:
        store.add(text, meta)
    print(f"  知识库: {len(store)} 篇文档")

    # RAG 检索
    rag = RAGTool(store, top_k=3)
    ctx = Context()

    queries = [
        "What is the Y combinator?",
        "How does Chain-of-Thought work?",
        "What is MCP protocol?",
    ]

    for q in queries:
        results = rag(q, ctx)
        print(f"\n  Q: {q}")
        # 只显示前两个结果的第一行
        for line in results.split("\n\n")[:2]:
            first_line = line.split("\n")[0]
            print(f"    {first_line[:70]}...")

    # JSON 格式
    rag_json = RAGTool(store, top_k=2, format="json")
    r = rag_json("lambda calculus history")
    parsed = json.loads(r)
    print(f"\n  [JSON 格式] {len(parsed)} 结果:")
    for item in parsed:
        print(f"    rank={item['rank']}, score={item['score']:.3f}: {item['content'][:50]}...")

    # create_rag 一行创建
    quick_rag = create_rag(["Python is great", "Java is verbose", "Rust is fast"])
    r = quick_rag("Which language is fast?")
    print(f"\n  [create_rag] Q: Which language is fast?")
    print(f"    A: {r.split(chr(10))[0][:60]}...")

    print("  ✅ RAG 检索完成")


# ══════════════════════════════════════════════════════════════
# 案例 4: AgenticRAG — 智能检索
# ══════════════════════════════════════════════════════════════

def demo_agentic_rag():
    """
    AgenticRAG: Agent 自主决定何时检索。

    Lambda 语义:
        AgenticRAG(agent, rag, decider) =
            λx. IF decider(x) THEN agent(x + rag(x)) ELSE agent(x)

    场景: 问答系统 — 有知识性问题才检索，闲聊不检索
    """
    separator("4. AgenticRAG — 智能检索")

    # 知识库
    rag = create_rag([
        "lambdagent is a Python DSL that models AI agents as Lambda calculus terms.",
        "The Y combinator enables recursion. terminate = λx.x is its base case.",
        "MCP connects AI models to tools. A2A connects agents to agents.",
        "Type Safety means well-typed programs don't go wrong.",
    ])

    # QA Agent
    qa = Tool("qa", lambda x:
        f"Based on the context, here's my answer: {x[:50]}... "
        f"This is a well-studied topic with clear theoretical foundations.")

    # Agentic RAG: 有问号 → 检索，否则直接回答
    agentic = AgenticRAG(
        agent=qa,
        rag=rag,
        decider=lambda x: "?" in x or any(w in x.lower() for w in ["what", "how", "why", "explain"]),
    )

    ctx = Context()

    # 需要检索的查询
    print("  [需要检索的查询]")
    r1 = agentic("What is the Y combinator?", ctx)
    print(f"    Q: What is the Y combinator?")
    print(f"    A: {r1[:70]}...")

    r2 = agentic("How does MCP work?", ctx)
    print(f"    Q: How does MCP work?")
    print(f"    A: {r2[:70]}...")

    # 不需要检索的查询
    print(f"\n  [不需要检索的查询]")
    r3 = agentic("Hello, nice to meet you!", ctx)
    print(f"    Q: Hello, nice to meet you!")
    print(f"    A: {r3[:70]}...")

    # 查看追踪：哪些触发了 RAG
    print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
    for i, e in enumerate(ctx.trace):
        rag_indicator = "📚" if "with_rag" in e.term_name else "💬"
        if "AgenticRAG" in e.term_name:
            print(f"    β[{i}] {rag_indicator} {e.term_name}")

    print("  ✅ AgenticRAG 完成")


# ══════════════════════════════════════════════════════════════
# 案例 5: 综合案例 — A2A + RAG + Skills
# ══════════════════════════════════════════════════════════════

def demo_full_p1():
    """
    综合案例: 知识增强的技能型 Agent 系统。

    架构:
        1. RAG 知识库提供领域知识
        2. Skills 提供可组合的能力
        3. SkillAgent 自动选择技能
        4. A2A 发布为可被外部发现的服务
    """
    separator("5. 综合: RAG + Skills + A2A")

    SkillRegistry().clear()

    # ── Step 1: 构建知识库 ──
    rag = create_rag([
        "Python: dynamically typed, great for data science. Created 1991.",
        "Rust: memory safe without GC. Created 2010. Growing fast in systems programming.",
        "Go: simple concurrency model. Created 2009. Popular for microservices.",
        "TypeScript: typed superset of JavaScript. Created 2012. Dominates web frontend.",
        "Lambda calculus: formal computation model. Created 1930s by Alonzo Church.",
    ])
    print("  [1] 知识库: 5 篇文档")

    # ── Step 2: 创建 RAG 增强的技能 ──
    @skill("lang_compare", "Compare programming languages", tags=["code", "analysis"])
    def lang_compare(x):
        ctx = Context()
        context = rag(x, ctx)
        return f"Comparison based on knowledge:\n{context}\n\nConclusion: Choose based on your use case."

    @skill("lang_recommend", "Recommend a language for a task", tags=["code", "recommendation"])
    def lang_recommend(x):
        ctx = Context()
        context = rag(x, ctx)
        if "web" in x.lower():
            rec = "TypeScript"
        elif "system" in x.lower() or "performance" in x.lower():
            rec = "Rust"
        elif "data" in x.lower() or "ml" in x.lower():
            rec = "Python"
        elif "microservice" in x.lower() or "server" in x.lower():
            rec = "Go"
        else:
            rec = "Python (general purpose)"
        return f"Recommendation: {rec}\nContext: {context[:100]}..."

    @skill("lang_quiz", "Quiz about programming languages", tags=["code", "education"])
    def lang_quiz(x):
        return "Quiz: Which language was created by Alonzo Church?\n(a) Python (b) Lambda calculus (c) Rust\nAnswer: (b)"

    print(f"  [2] 技能: {len(SkillRegistry())} 个已注册")

    # ── Step 3: SkillAgent 自动选择 ──
    selector = Tool("selector", lambda x:
        "lang_compare" if any(w in x.lower() for w in ["compare", "vs", "versus", "difference"]) else
        "lang_recommend" if any(w in x.lower() for w in ["recommend", "which", "best", "should"]) else
        "lang_quiz"
    )
    agent = SkillAgent(classifier=selector, registry=SkillRegistry())

    ctx = Context()
    print(f"\n  [3] SkillAgent 自动选择技能:")

    queries = [
        "Compare Python vs Rust for data processing",
        "Which language should I use for web development?",
        "Give me a quiz about programming",
    ]
    for q in queries:
        r = agent(q, ctx)
        print(f"    Q: {q}")
        print(f"    A: {r[:70]}...")
        print()

    # ── Step 4: 发布为 A2A 服务 ──
    card = registry_to_agent_card(
        SkillRegistry(),
        url="https://lang-advisor.example.com",
    )
    print(f"  [4] A2A Agent Card:")
    print(f"    Name: {card.name}")
    print(f"    Skills: {[s['name'] for s in card.skills]}")
    print(f"    URL: {card.url}")

    # 本地 A2A 测试
    server = A2AServer(agent, card=card, port=19879)
    server.start(background=True)
    time.sleep(0.3)

    try:
        client = A2AClient("http://127.0.0.1:19879")
        r = client("Which language for microservices?")
        print(f"\n  [5] A2A 远程调用:")
        print(f"    Q: Which language for microservices?")
        print(f"    A: {r[:70]}...")
    finally:
        server.stop()

    print(f"\n  总 β-规约: {len(ctx.trace)} 步")
    print("  ✅ 综合案例完成")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  lambdagent P1: A2A Protocol + RAG 演示                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demos = [
        ("A2A Agent Card", demo_agent_card),
        ("A2A Server/Client", demo_a2a_communication),
        ("RAG 知识库", demo_rag),
        ("AgenticRAG 智能检索", demo_agentic_rag),
        ("综合: RAG + Skills + A2A", demo_full_p1),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  全部 5 个 P1 案例完成")
    print("=" * 70)
