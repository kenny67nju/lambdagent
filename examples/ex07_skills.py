"""
示例 7: Skill 系统 — 可复用、可发现、可组合的技能

展示 lambdagent 的 Skill 系统：
  1. @skill 装饰器 — 一行创建 + 自动注册
  2. Skill 组合 (>>) — 带类型检查的管道
  3. Skill 柯里化 (.bind) — 偏应用
  4. SkillPack — 技能包分发
  5. SkillRegistry — 全局注册表搜索
  6. SkillAgent — LLM 驱动的技能自动发现

运行:
    python -m lambdagent.examples.ex07_skills
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lambdagent import (
    Tool, Context,
    Skill, SkillSignature, SkillPack,
    SkillRegistry, SkillAgent, skill,
)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  案例: {title}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# 案例 1: @skill 装饰器 — 零摩擦创建技能
# ══════════════════════════════════════════════════════════════

def demo_skill_decorator():
    """
    @skill 装饰器: 一行代码创建技能 + 自动注册到全局 Registry。

    Lambda 语义:
        @skill("name", desc) def f(x): ...
        = let name = Tool("name", f) in Γ_skills[name ↦ Skill(name, f)]
    """
    separator("1. @skill 装饰器")

    SkillRegistry().clear()

    # ── 文本处理技能 ──

    @skill("word_count", "Count the number of words in text",
           tags=["text", "analysis"],
           input_type="Str", output_type="Int",
           examples=[("hello world", "2"), ("one two three", "3")])
    def word_count(x):
        return str(len(x.split()))

    @skill("char_count", "Count characters in text",
           tags=["text", "analysis"])
    def char_count(x):
        return str(len(x))

    @skill("to_upper", "Convert text to UPPERCASE",
           tags=["text", "transform"])
    def to_upper(x):
        return x.upper()

    @skill("to_lower", "Convert text to lowercase",
           tags=["text", "transform"])
    def to_lower(x):
        return x.lower()

    @skill("reverse", "Reverse the text",
           tags=["text", "transform"])
    def reverse_text(x):
        return x[::-1]

    @skill("extract_numbers", "Extract all numbers from text",
           tags=["text", "extraction"],
           input_type="Str", output_type="List[Int]")
    def extract_numbers(x):
        import re
        nums = re.findall(r'\d+', x)
        return ", ".join(nums) if nums else "no numbers found"

    @skill("summarize_stats", "Generate text statistics",
           tags=["text", "analysis"],
           input_type="Str", output_type="Str")
    def summarize_stats(x):
        words = len(x.split())
        chars = len(x)
        sentences = x.count('.') + x.count('!') + x.count('?')
        return f"Words: {words}, Chars: {chars}, Sentences: {sentences}"

    reg = SkillRegistry()
    print(f"  已注册 {len(reg)} 个技能:")
    for name in reg.list_all():
        s = reg.get(name)
        print(f"    • {name}: {s.description} [{', '.join(s.tags)}]")

    # 测试
    text = "Hello World! This is a test. AI has 3 models and 100 users."
    print(f"\n  输入: \"{text}\"")
    print(f"  word_count → {word_count(text)}")
    print(f"  char_count → {char_count(text)}")
    print(f"  extract_numbers → {extract_numbers(text)}")
    print(f"  summarize_stats → {summarize_stats(text)}")
    print("  ✅ 技能创建完成")


# ══════════════════════════════════════════════════════════════
# 案例 2: Skill 组合 — 带类型检查的管道
# ══════════════════════════════════════════════════════════════

def demo_skill_composition():
    """
    技能组合: skill_a >> skill_b = λx. skill_b(skill_a(x))

    Lambda 语义:
        组合时检查类型: a.τ_out 兼容 b.τ_in
        组合后的类型: a.τ_in → b.τ_out
    """
    separator("2. Skill 组合 (>> 管道)")

    reg = SkillRegistry()

    to_upper = reg.get("to_upper")
    reverse = reg.get("reverse")
    word_count = reg.get("word_count")

    # 简单组合
    pipeline1 = to_upper >> reverse
    result = pipeline1("hello world")
    print(f"  to_upper >> reverse: \"hello world\" → \"{result}\"")
    assert result == "DLROW OLLEH"

    # 三段管道
    pipeline2 = to_upper >> reverse >> word_count
    result = pipeline2("hello world")
    print(f"  to_upper >> reverse >> word_count: → {result}")

    # 查看组合后的技能元数据
    print(f"\n  组合技能:")
    print(f"    名称: {pipeline1._name}")
    print(f"    类型: {pipeline1.signature.input_type} → {pipeline1.signature.output_type}")
    print(f"    标签: {pipeline1.tags}")
    print("  ✅ 技能组合完成")


# ══════════════════════════════════════════════════════════════
# 案例 3: Skill 柯里化 — 偏应用
# ══════════════════════════════════════════════════════════════

def demo_skill_currying():
    """
    柯里化: skill.bind(param=value) 固定部分参数。

    Lambda 语义:
        skill.bind(lang="zh") = λx. skill("[lang=zh]\n" + x)
        = 偏应用，生成新技能
    """
    separator("3. Skill 柯里化 (.bind)")

    @skill("formatter", "Format text with style",
           tags=["text", "format"])
    def formatter(x):
        # 检测参数
        if "[style=markdown]" in x:
            text = x.split("\n")[-1]
            return f"# {text}\n\n> Formatted as markdown"
        elif "[style=json]" in x:
            text = x.split("\n")[-1]
            return f'{{"text": "{text}", "format": "json"}}'
        else:
            return x

    # 柯里化: 生成特化技能
    md_formatter = formatter.bind(style="markdown")
    json_formatter = formatter.bind(style="json")

    print(f"  原始技能: {formatter._name}")
    print(f"  MD 版本:  {md_formatter._name}")
    print(f"  JSON 版本: {json_formatter._name}")

    r1 = md_formatter("Hello World")
    r2 = json_formatter("Hello World")
    print(f"\n  md_formatter(\"Hello World\"):")
    print(f"    {r1}")
    print(f"  json_formatter(\"Hello World\"):")
    print(f"    {r2}")
    print("  ✅ 柯里化完成")


# ══════════════════════════════════════════════════════════════
# 案例 4: SkillPack — 技能包
# ══════════════════════════════════════════════════════════════

def demo_skill_pack():
    """
    SkillPack: 一组相关技能的集合。

    Lambda 语义:
        SkillPack = {name₁: skill₁, name₂: skill₂, ...}
        = Python package 的 Lambda 演算版本

    用途: 按领域组织、版本化分发、一键注册
    """
    separator("4. SkillPack (技能包)")

    reg = SkillRegistry()

    # ── 构建文本处理技能包 ──
    text_pack = SkillPack(
        name="text-utils",
        description="A collection of text processing skills",
        version="1.2.0",
        author="lambdagent",
    )

    # 从 registry 中选取技能加入 pack
    for name in ["word_count", "char_count", "to_upper", "to_lower", "reverse"]:
        s = reg.get(name)
        if s:
            text_pack.add(s)

    print(f"  {text_pack}")
    print(f"  技能列表: {text_pack.list_skills()}")
    print(f"  版本: {text_pack.version}")

    # ── 构建分析技能包 ──
    analysis_pack = SkillPack("analytics", "Data analysis skills", "1.0.0")

    @skill("avg_word_len", "Calculate average word length", tags=["analysis"])
    def avg_word_len(x):
        words = x.split()
        if not words:
            return "0"
        avg = sum(len(w) for w in words) / len(words)
        return f"{avg:.1f}"

    @skill("top_words", "Find most common words", tags=["analysis"])
    def top_words(x):
        from collections import Counter
        words = x.lower().split()
        common = Counter(words).most_common(3)
        return ", ".join(f"{w}({c})" for w, c in common)

    analysis_pack.add(avg_word_len).add(top_words)

    # 批量注册
    reg.register_pack(analysis_pack)
    print(f"\n  已注册 {analysis_pack}")
    print(f"  Registry 总计: {len(reg)} 个技能")
    print("  ✅ SkillPack 完成")


# ══════════════════════════════════════════════════════════════
# 案例 5: SkillRegistry 搜索
# ══════════════════════════════════════════════════════════════

def demo_registry_search():
    """
    技能注册表搜索: 按名称、描述、标签查找技能。

    Lambda 语义:
        search(query) = {s ∈ Γ_skills | query matches s.name ∨ s.description}
        search(tags)  = {s ∈ Γ_skills | tags ⊆ s.tags}
    """
    separator("5. Registry 搜索")

    reg = SkillRegistry()

    # 文本搜索
    print("  文本搜索:")
    for query in ["count", "reverse", "format", "upper"]:
        results = reg.search(query=query)
        names = [s._name for s in results]
        print(f"    search(\"{query}\") → {names}")

    # 标签搜索
    print("\n  标签搜索:")
    for tags in [["text"], ["analysis"], ["transform"], ["text", "analysis"]]:
        results = reg.search(tags=tags)
        names = [s._name for s in results]
        print(f"    tags={tags} → {names}")

    # 统计
    stats = reg.stats()
    print(f"\n  注册表统计:")
    print(f"    总技能数: {stats['total_skills']}")
    print(f"    标签分布: {stats['skills_by_tag']}")
    print(f"    最常使用: {stats['most_used'][:5]}")
    print("  ✅ Registry 搜索完成")


# ══════════════════════════════════════════════════════════════
# 案例 6: SkillAgent — 自动技能发现
# ══════════════════════════════════════════════════════════════

def demo_skill_agent():
    """
    SkillAgent: LLM 驱动的技能自动发现和执行。

    Lambda 语义:
        SkillAgent(classifier, Γ_skills) =
            λx. let name = classifier(x) in Γ_skills[name](x)

    = Handoff 的技能化版本，classifier 选择的是 Skill 而非普通 Agent
    """
    separator("6. SkillAgent (自动技能发现)")

    reg = SkillRegistry()

    # 分类器 (用 Tool 模拟 LLM)
    classifier = Tool("skill_selector", lambda x: (
        "word_count" if any(w in x.lower() for w in ["count", "how many words"]) else
        "to_upper" if any(w in x.lower() for w in ["upper", "capitalize", "大写"]) else
        "reverse" if any(w in x.lower() for w in ["reverse", "反转", "backwards"]) else
        "extract_numbers" if any(w in x.lower() for w in ["number", "数字", "extract"]) else
        "summarize_stats" if any(w in x.lower() for w in ["stats", "统计", "analyze"]) else
        "top_words"  # fallback
    ))

    agent = SkillAgent(classifier=classifier, registry=reg)

    # 测试
    queries = [
        ("How many words: the quick brown fox", "word_count"),
        ("Make UPPER: hello world", "to_upper"),
        ("Reverse this: abcdefg", "reverse"),
        ("Extract numbers from: I have 3 cats and 12 dogs", "extract_numbers"),
        ("Show stats: Hello World. This is a test.", "summarize_stats"),
    ]

    ctx = Context()
    print("  SkillAgent 自动选择技能:")
    for query, expected_skill in queries:
        result = agent(query, ctx)
        print(f"    输入: \"{query[:50]}\"")
        print(f"    选择: → {expected_skill}")
        print(f"    输出: {result}")
        print()

    print(f"  β-规约追踪 ({len(ctx.trace)} 步):")
    ctx.print_trace()
    print("  ✅ SkillAgent 完成")


# ══════════════════════════════════════════════════════════════
# 案例 7: 综合案例 — Skill 驱动的文档处理管道
# ══════════════════════════════════════════════════════════════

def demo_full_pipeline():
    """
    综合案例: 用 Skill 构建一个完整的文档处理管道。

    管道: extract_numbers >> to_upper >> summarize_stats
    + SkillAgent 根据用户意图自动选择
    """
    separator("7. 综合案例 — Skill 驱动的文档处理")

    reg = SkillRegistry()

    # 创建专门的文档处理技能
    @skill("doc_clean", "Clean and normalize document text",
           tags=["document", "preprocessing"])
    def doc_clean(x):
        # 去除多余空白，统一换行
        import re
        cleaned = re.sub(r'\s+', ' ', x).strip()
        return cleaned

    @skill("doc_extract_entities", "Extract named entities from text",
           tags=["document", "extraction", "NLP"])
    def doc_extract_entities(x):
        # 简单的实体提取（模拟 NER）
        import re
        # 找大写开头的词作为"实体"
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', x)
        unique = list(set(entities))[:10]
        return f"Entities: {', '.join(unique)}" if unique else "No entities found"

    @skill("doc_sentiment", "Analyze sentiment of text",
           tags=["document", "analysis", "NLP"])
    def doc_sentiment(x):
        positive = sum(1 for w in ["good", "great", "excellent", "amazing", "love"]
                      if w in x.lower())
        negative = sum(1 for w in ["bad", "terrible", "awful", "hate", "poor"]
                      if w in x.lower())
        if positive > negative:
            return f"Sentiment: POSITIVE (score: +{positive - negative})"
        elif negative > positive:
            return f"Sentiment: NEGATIVE (score: {positive - negative})"
        return f"Sentiment: NEUTRAL (score: 0)"

    # 构建管道: clean → entities → sentiment
    pipeline = doc_clean >> doc_extract_entities

    doc = """
    Apple Inc. announced a new AI product called Apple Intelligence.
    Microsoft and Google are also competing in this great market.
    The technology is amazing and the growth is excellent.
    However, some critics say the hype is bad for the industry.
    """

    ctx = Context()
    print(f"  输入文档: ({len(doc)} 字符)")
    print(f"    \"{doc.strip()[:80]}...\"")

    # 管道执行
    print(f"\n  [Pipeline] doc_clean >> doc_extract_entities")
    result = pipeline(doc, ctx)
    print(f"    结果: {result}")

    # 单独执行情感分析
    print(f"\n  [Skill] doc_sentiment")
    sentiment = doc_sentiment(doc)
    print(f"    结果: {sentiment}")

    # 统计
    print(f"\n  [Skill] summarize_stats")
    stats = reg.get("summarize_stats")
    if stats:
        print(f"    结果: {stats(doc)}")

    print(f"\n  注册表最终状态: {len(reg)} 个技能")
    print(f"  β-规约: {len(ctx.trace)} 步")
    ctx.print_trace()
    print("  ✅ 综合案例完成")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  lambdagent Skill 系统演示 (7 个案例)                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demos = [
        ("@skill 装饰器", demo_skill_decorator),
        ("Skill 组合", demo_skill_composition),
        ("Skill 柯里化", demo_skill_currying),
        ("SkillPack", demo_skill_pack),
        ("Registry 搜索", demo_registry_search),
        ("SkillAgent", demo_skill_agent),
        ("综合案例", demo_full_pipeline),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  全部 7 个 Skill 案例完成")
    print("=" * 70)
