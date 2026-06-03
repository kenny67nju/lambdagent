"""
示例 6: 多智能体系统 — 5 个新构造的完整演示

展示 lambdagent 的 5 个多智能体构造：
  1. Channel + Send/Receive  — π-calculus 通道通信
  2. SharedMemory            — 多 Agent 共享记忆
  3. GroupChat               — 群组对话（Y_n + Route 组合）
  4. Handoff                 — 动态委派（运行时 Route）
  5. AsyncPar                — 真并行执行

每个案例都用 Tool 模拟 LLM（无需 API key），展示纯粹的 Lambda/π 语义。
替换为 Lam 即可接入真实 LLM。

运行:
    python -m lambdagent.examples.ex06_multiagent
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lambdagent import (
    Tool, Context, Lam,
    Channel, Send, Receive,
    SharedMemory,
    GroupChat,
    Handoff,
    AsyncPar,
    Par,
)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  案例: {title}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# 案例 1: Channel 通信 — 生产者/消费者模式
# ══════════════════════════════════════════════════════════════

def demo_channel():
    """
    π-calculus 通道通信。

    Lambda + π 语义:
        ν(ch). (Send(producer, ch) | Receive(ch, consumer))
        = 创建私有通道 ch，producer 和 consumer 通过 ch 通信

    场景: 研究员 Agent 搜索资料，通过通道发送给写作 Agent
    """
    separator("1. Channel 通信 (π-calculus)")

    # 创建通道（无缓冲 = 同步通信）
    research_channel = Channel("research", capacity=5)
    print(f"  创建通道: {research_channel}")

    # 研究员: 搜索并发送到通道
    researcher = Tool("researcher", lambda x:
        f"[研究结果] 关于'{x}'的3个要点: 1.定义 2.应用 3.趋势")
    sender = Send(researcher, research_channel)

    # 写作者: 从通道接收并加工
    writer = Tool("writer", lambda x:
        f"[文章] 基于研究结果撰写:\n{x}\n→ 综上所述，这是一个重要话题。")
    receiver = Receive(research_channel, handler=writer, timeout=5)

    # 执行: 研究员发送
    ctx = Context()
    print("  [研究员] 开始搜索...")
    research_result = sender("人工智能安全", ctx)
    print(f"  [研究员] 已发送: {research_result[:60]}...")
    print(f"  [通道] 待接收消息: {research_channel.pending}")

    # 执行: 写作者接收
    print("  [写作者] 等待研究结果...")
    article = receiver("", ctx)
    print(f"  [写作者] 完成: {article[:80]}...")

    # 追踪
    print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()

    # ── 高级: 多线程通道 ──
    print("\n  --- 多线程通道演示 ---")
    async_ch = Channel("async_ch", capacity=0)  # 无缓冲 = 必须同步

    results = []

    def producer_thread():
        time.sleep(0.1)  # 模拟异步延迟
        async_ch.send("来自另一个线程的消息")

    def consumer_thread():
        msg = async_ch.receive(timeout=2)
        results.append(msg)

    t1 = threading.Thread(target=producer_thread)
    t2 = threading.Thread(target=consumer_thread)
    t2.start(); t1.start()
    t1.join(); t2.join()

    print(f"  跨线程通信: {results[0]}")
    print("  ✅ Channel 通信完成")


# ══════════════════════════════════════════════════════════════
# 案例 2: SharedMemory — 多 Agent 共享知识库
# ══════════════════════════════════════════════════════════════

def demo_shared_memory():
    """
    共享记忆: 多个 Agent 读写同一个 Γ_shared。

    Lambda 语义:
        SharedMem(store) 创建 Γ_shared
        sm.wrap(agent_a) = λx. agent_a(x) [Γ ∪ Γ_shared]
        sm.wrap(agent_b) = λx. agent_b(x) [Γ ∪ Γ_shared]
        agent_a 和 agent_b 看到同一个 Γ_shared

    场景: 数据采集 Agent 写入，分析 Agent 读取，报告 Agent 汇总
    """
    separator("2. SharedMemory (共享环境 Γ_shared)")

    # 创建共享记忆（append-only 保证类型安全）
    shared = SharedMemory(
        store={"project": "AI Agent 市场分析", "data_points": 0},
        append_only=True,
    )
    print(f"  创建: {shared}")

    # Agent A: 数据采集
    collector = Tool("collector", lambda x: (
        shared.write("market_size", "$15.6B"),
        shared.write("growth_rate", "34.2%"),
        shared.write("data_points", 2),  # int → int, 类型一致 ✓
        f"已采集 2 个数据点"
    )[-1])
    wrapped_collector = shared.wrap(collector)

    # Agent B: 分析
    analyst = Tool("analyst", lambda x:
        f"分析结果: 市场规模={shared.read('market_size', 'N/A')}, "
        f"增长率={shared.read('growth_rate', 'N/A')}, "
        f"共{shared.read('data_points', 0)}个数据点")
    wrapped_analyst = shared.wrap(analyst)

    # Agent C: 报告
    reporter = Tool("reporter", lambda x:
        f"[报告] 项目: {shared.read('project')}\n"
        f"  {shared.read('market_size', '')} 市场, {shared.read('growth_rate', '')} 增速\n"
        f"  结论: 市场增长强劲")
    wrapped_reporter = shared.wrap(reporter)

    # 执行流水线
    ctx = Context()
    print("  [采集] ", wrapped_collector("开始采集", ctx))
    print("  [分析] ", wrapped_analyst("请分析", ctx))
    print("  [报告] ", wrapped_reporter("生成报告", ctx))

    # 验证类型安全
    print("\n  --- 类型安全测试 ---")
    try:
        shared.write("data_points", "not a number")  # int → str, 违反 Σ'⊇Σ
        print("  ❌ 应该抛出 TypeError")
    except TypeError as e:
        print(f"  ✅ 类型安全: {e}")

    print(f"\n  共享记忆最终状态: {shared.read_all()}")
    print("  ✅ SharedMemory 完成")


# ══════════════════════════════════════════════════════════════
# 案例 3: GroupChat — 多 Agent 辩论
# ══════════════════════════════════════════════════════════════

def demo_groupchat():
    """
    群组对话: 多个 Agent 轮流发言直到达成共识。

    Lambda 语义:
        GroupChat([a,b,c], n) = Y_n(λself.λstate.
            let speaker = scheduler(state) in
            let msg = speaker(state) in
            IF "CONSENSUS" ∈ state' THEN state' ELSE self(state')
        )

    场景: 三位专家讨论技术选型
    """
    separator("3. GroupChat (多 Agent 辩论)")

    round_counter = [0]

    # 三位专家 Agent
    alice = Tool("Alice_CTO", lambda x: (
        round_counter.__setitem__(0, round_counter[0] + 1),
        f"[Alice/CTO] 从架构角度看，我建议用 Rust。性能和安全性是第一位的。"
        if round_counter[0] <= 1 else
        f"[Alice/CTO] 好的，综合考虑大家意见，我同意 Go 是更务实的选择。CONSENSUS"
    )[-1])

    bob = Tool("Bob_Lead", lambda x:
        f"[Bob/Lead] 我认为 Go 更适合我们团队。学习曲线低，生态成熟，部署简单。")

    carol = Tool("Carol_SRE", lambda x:
        f"[Carol/SRE] 从运维角度，Go 的编译速度和部署体验确实更好。支持 Bob 的建议。")

    # 创建 GroupChat
    chat = GroupChat(
        agents=[alice, bob, carol],
        max_rounds=6,
        scheduler="round_robin",
        # 默认终止条件: 包含 "CONSENSUS"
    )

    # 执行
    ctx = Context()
    print("  开始技术选型讨论...")
    result = chat("我们的新微服务应该用什么语言？", ctx)
    print(f"\n  最终结论: {result[:80]}...")

    print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()
    print("  ✅ GroupChat 完成")


# ══════════════════════════════════════════════════════════════
# 案例 4: Handoff — 智能客服路由
# ══════════════════════════════════════════════════════════════

def demo_handoff():
    """
    动态委派: 运行时确定路由目标。

    Lambda 语义:
        Handoff(selector, registry) =
            λx. let target = selector(x) in registry[target](x)

    与 Route 的区别:
        Route  = 编译时确定路由表（静态 CASE）
        Handoff = 运行时确定（动态 CASE）+ 可动态注册新 Agent

    场景: 智能客服系统，根据用户问题动态路由到专家
    """
    separator("4. Handoff (智能客服动态路由)")

    # 专家 Agent 团队
    billing = Tool("billing_expert", lambda x:
        f"[账单专家] 您的账户余额为 ¥1,234.56。如需充值请访问 pay.example.com")
    tech = Tool("tech_support", lambda x:
        f"[技术支持] 请尝试: 1)重启应用 2)清除缓存 3)检查网络连接")
    sales = Tool("sales_rep", lambda x:
        f"[销售顾问] 我们目前有企业版 8 折优惠，包含: API 无限调用 + 专属客服")

    # 路由选择器
    def route_selector(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["账单", "余额", "付款", "充值", "billing"]):
            return "billing_expert"
        elif any(w in q for w in ["bug", "错误", "崩溃", "无法", "tech"]):
            return "tech_support"
        elif any(w in q for w in ["价格", "优惠", "购买", "企业版", "sales"]):
            return "sales_rep"
        return "general"

    # 创建 Handoff（带 fallback）
    general = Tool("general", lambda x: f"[通用客服] 我来帮您转接专家，请稍候...")
    handoff = Handoff(
        selector=route_selector,
        registry={
            "billing_expert": billing,
            "tech_support": tech,
            "sales_rep": sales,
        },
        fallback=general,
    )

    # 测试多个场景
    ctx = Context()
    queries = [
        "我想查一下账单余额",
        "应用启动后崩溃了",
        "企业版现在什么价格？",
        "你们公司在哪里？",  # fallback
    ]

    for q in queries:
        result = handoff(q, ctx)
        print(f"  用户: {q}")
        print(f"  回复: {result}")
        print()

    # ── 动态注册新专家 ──
    print("  --- 动态注册 VIP 专家 ---")
    vip = Tool("vip_support", lambda x:
        f"[VIP专属] 尊敬的客户，我是您的专属顾问，正在为您优先处理。")
    handoff.register("vip_support", vip)
    print(f"  已注册 VIP 专家，当前 {len(handoff.registry)} 个专家")

    print(f"\n  β-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()
    print("  ✅ Handoff 完成")


# ══════════════════════════════════════════════════════════════
# 案例 5: AsyncPar — 多维度并行分析
# ══════════════════════════════════════════════════════════════

def demo_async_par():
    """
    真并行执行: 多个 Agent 同时运行，互不阻塞。

    Lambda 语义:
        AsyncPar(f, g, h) = λx. concurrent(f(x), g(x), h(x))

    与 Par 的区别:
        Par      = 顺序执行（假并行）: 总时间 = Σ(各 Agent 时间)
        AsyncPar = 线程池并发（真并行）: 总时间 ≈ max(各 Agent 时间)

    场景: 对同一份数据做多维度分析（技术/市场/风险）
    """
    separator("5. AsyncPar (多维度并行分析)")

    # 三个分析 Agent（各需 0.3 秒）
    def slow_tech(x):
        time.sleep(0.3)
        return f"[技术分析] {x[:20]}... → 技术可行性: 高。核心技术成熟度 8/10。"

    def slow_market(x):
        time.sleep(0.3)
        return f"[市场分析] {x[:20]}... → 市场规模: $15B，增速 34%。竞争中等。"

    def slow_risk(x):
        time.sleep(0.3)
        return f"[风险分析] {x[:20]}... → 主要风险: 监管不确定性。建议: 合规先行。"

    tech = Tool("tech_analyst", slow_tech)
    market = Tool("market_analyst", slow_market)
    risk = Tool("risk_analyst", slow_risk)

    input_data = "AI Agent 平台项目投资评估"

    # ── 对比: 串行 vs 并行 ──
    print("  [串行执行 Par]")
    t0 = time.time()
    ctx1 = Context()
    r_seq = Par(tech, market, risk).apply(input_data, ctx1)
    seq_time = time.time() - t0
    print(f"    耗时: {seq_time:.3f}s")
    for r in r_seq:
        print(f"    {r[:60]}...")

    print(f"\n  [并行执行 AsyncPar]")
    t0 = time.time()
    ctx2 = Context()
    r_par = AsyncPar(tech, market, risk).apply(input_data, ctx2)
    par_time = time.time() - t0
    print(f"    耗时: {par_time:.3f}s")
    for r in r_par:
        print(f"    {r[:60]}...")

    speedup = seq_time / par_time
    print(f"\n  加速比: {speedup:.1f}x ({seq_time:.3f}s → {par_time:.3f}s)")
    print(f"  节省: {seq_time - par_time:.3f}s ({(1 - par_time/seq_time)*100:.0f}%)")

    # 合并结果
    merge = Tool("merge", lambda x: f"[综合评估]\n" + "\n".join(x) + "\n→ 建议: 投资可行")
    merged = merge(list(r_par))
    print(f"\n  合并结果: {merged[:80]}...")
    print("  ✅ AsyncPar 完成")


# ══════════════════════════════════════════════════════════════
# 案例 6: 综合案例 — 多 Agent 研究系统
# ══════════════════════════════════════════════════════════════

def demo_full_system():
    """
    综合案例: 组合使用所有多智能体构造。

    架构:
        1. Handoff 路由用户请求到正确的工作流
        2. AsyncPar 并行执行研究和批评
        3. SharedMemory 在 Agent 间共享发现
        4. Channel 传递中间结果
        5. GroupChat 讨论最终结论

    Lambda 语义:
        Handoff(selector, {
            "research": AsyncPar(search, analyze) >> Channel >> GroupChat([reviewer, editor])
        })
    """
    separator("6. 综合案例 — 多构造协作")

    # 共享记忆
    shared = SharedMemory(store={"task": "", "findings": []})

    # 研究管道: 并行搜索 + 分析
    search = Tool("search", lambda x: (
        shared.write("task", x),
        f"搜索结果: {x} 相关的 5 篇论文"
    )[-1])

    analyze = Tool("analyze", lambda x:
        f"分析结果: {x} 领域有 3 个关键趋势")

    # 并行执行搜索和分析
    research_par = AsyncPar(search, analyze)

    # 通道: 传递研究结果给审阅者
    review_channel = Channel("review", capacity=5)

    # 审阅讨论
    reviewer = Tool("reviewer", lambda x:
        f"[审阅] 研究质量良好，建议补充定量数据。CONSENSUS")
    editor = Tool("editor", lambda x:
        f"[编辑] 同意审阅意见，文章可以发表。CONSENSUS")

    discussion = GroupChat(
        agents=[reviewer, editor],
        max_rounds=3,
        scheduler="round_robin",
    )

    # 执行完整流水线
    ctx = Context()
    input_task = "Lambda 演算在 AI Agent 中的应用"

    print(f"  任务: {input_task}")
    print()

    # Step 1: 并行研究
    print("  [Step 1] 并行研究 (AsyncPar)...")
    t0 = time.time()
    search_result, analysis_result = research_par(input_task, ctx)
    print(f"    搜索: {search_result}")
    print(f"    分析: {analysis_result}")

    # Step 2: 通过通道传递
    print("\n  [Step 2] 通道传递 (Channel)...")
    combined = f"{search_result}\n{analysis_result}"
    review_channel.send(combined)
    print(f"    已发送到审阅通道 (pending={review_channel.pending})")

    # Step 3: 接收并讨论
    print("\n  [Step 3] 群组讨论 (GroupChat)...")
    received = review_channel.receive(timeout=1)
    conclusion = discussion(received, ctx)
    print(f"    结论: {conclusion[:60]}...")

    # Step 4: 写入共享记忆
    shared.write("conclusion", conclusion)
    shared.write("status", "completed")
    print(f"\n  [Step 4] 共享记忆最终状态:")
    for k, v in shared.read_all().items():
        print(f"    {k}: {str(v)[:50]}")

    total_time = time.time() - t0
    print(f"\n  总耗时: {total_time:.3f}s")
    print(f"  β-规约总步数: {len(ctx.trace)}")
    print("\n  完整追踪:")
    ctx.print_trace()
    print("  ✅ 综合案例完成")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  lambdagent 多智能体系统演示 (5 构造 + 综合案例)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demos = [
        ("Channel 通信", demo_channel),
        ("SharedMemory 共享记忆", demo_shared_memory),
        ("GroupChat 群组对话", demo_groupchat),
        ("Handoff 动态委派", demo_handoff),
        ("AsyncPar 真并行", demo_async_par),
        ("综合案例", demo_full_system),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  全部 6 个多智能体案例完成")
    print("=" * 70)
