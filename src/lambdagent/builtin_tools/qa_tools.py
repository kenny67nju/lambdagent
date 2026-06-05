"""
lambdagent.builtin_tools.qa_tools — 文档问答智能体专用工具

4 个工具:
  IngestFiles     — 批量喂文件 (内部: map_reduce pattern)
  QueryKnowledge  — 问答 (内部: pipeline + review pattern)
  ListKnowledge   — 查看已索引文件
  RemoveKnowledge — 删除文件索引

Lambda 语义:
  IngestFiles    = map_reduce(file_lister, reader >> splitter >> indexer, summarizer)
  QueryKnowledge = review(pipeline(retriever, answerer, citer), fact_checker)
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

from .knowledge_tools import ChunkSplitter, kb_manage, _parse


# ════════════════════════════════════════════════════════════
# 常量
# ════════════════════════════════════════════════════════════

DEFAULT_KB = "qaagent_default"


# ════════════════════════════════════════════════════════════
# Skill 实例 (Lam) — 延迟初始化
# ════════════════════════════════════════════════════════════

_skills_cache: Dict[str, Any] = {}


def _get_skill(name: str):
    """获取已编译的 Skill (Lam 实例)，延迟初始化避免启动开销。"""
    if name in _skills_cache:
        return _skills_cache[name]

    from lambdagent.primitives import Lam

    if name == "answer-generator":
        skill = Lam(
            "answer-generator",
            "你是一个严谨的问答助手。根据提供的参考文档回答用户问题。\n\n"
            "规则:\n"
            "1. 只基于提供的文档内容回答，不要编造信息\n"
            "2. 如果文档中没有相关信息，明确说明'文档中未找到相关信息'\n"
            "3. 回答要具体，引用文档中的原文作为依据\n"
            "4. 对于数字、日期等事实性信息，必须与文档原文一致\n"
            "5. 用分点列举组织回答结构",
        )
    elif name == "source-citer":
        skill = Lam(
            "source-citer",
            "你是一个引用标注专家。给定一个回答和原始文档块，\n"
            "为回答中的每个关键论点添加来源引用。\n\n"
            "格式: 在每个论点后添加 [来源: 文件名 位置]\n"
            "例如: '系统使用 Redis 作为缓存 [来源: architecture.md 4.2]'\n\n"
            "规则: 每个事实性陈述必须标注来源。不要改变原始回答内容。",
        )
    elif name == "fact-checker":
        skill = Lam(
            "fact-checker",
            "你是一个严格的事实核查员。检查回答中的每个论点是否有原文支撑。\n\n"
            "检查标准:\n"
            "1. 每个事实性陈述必须在原文中有对应内容\n"
            "2. 数字、日期必须与原文完全一致\n"
            "3. 不能有原文未提及的推测性内容\n\n"
            "如果所有论点都有据可查，回复: VERIFIED\n"
            "如果发现问题，回复: REJECTED: [具体指出哪个论点缺乏依据]",
        )
    elif name == "deep-analyzer":
        skill = Lam(
            "deep-analyzer",
            "你是一个深度分析专家。基于提供的多路检索结果进行综合分析。\n\n"
            "分析类型:\n"
            "- 对比分析: 列出异同点，给出对比表格\n"
            "- 多角度评估: 从指定维度逐一分析\n"
            "- 综合总结: 提炼核心观点\n\n"
            "要求: 每个论点引用原文 [来源: 文件名]，区分明确提到和推断。",
        )
    elif name == "synthesizer":
        skill = Lam(
            "synthesizer",
            "你是一个综合分析专家。你会收到多份独立的分析报告。\n\n"
            "你的工作:\n"
            "1. 阅读所有子分析报告\n"
            "2. 去除重复内容，合并互补信息\n"
            "3. 输出一份完整、连贯、结构化的最终报告\n\n"
            "要求: 保留来源引用，标注矛盾点，给出明确结论和建议。",
        )
    else:
        raise ValueError(f"Unknown skill: {name}")

    _skills_cache[name] = skill
    return skill
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".css", ".html", ".xml", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".sh", ".bat", ".sql", ".r", ".rb", ".php", ".swift", ".kt",
    ".pdf", ".csv", ".log",
}


# ════════════════════════════════════════════════════════════
# IngestFiles — 批量喂文件 (map_reduce pattern)
# ════════════════════════════════════════════════════════════

def ingest_files(input_val: Any) -> str:
    """
    批量加入文件到知识库。

    Pattern: map_reduce
      splitter = 列出文件
      mapper   = read → chunk → index (每个文件)
      reducer  = 汇总统计

    输入 (JSON 或文本):
      {"path": "/docs", "collection": "my_kb"}
      {"path": "*.py", "collection": "code_kb"}
      "/path/to/file.md"

    输出: 索引汇总报告
    """
    params = _parse(input_val)
    path = params.get("path", str(input_val).strip())
    collection = params.get("collection", DEFAULT_KB)
    chunk_size = params.get("chunk_size", 512)
    chunk_overlap = params.get("chunk_overlap", 50)

    # 1. 收集文件列表 (splitter)
    files = _collect_files(path)
    if not files:
        return f"[INFO] 未找到可索引的文件: {path}"

    # 2. 确保知识库存在
    _ensure_kb(collection)

    # 3. 对每个文件: read → chunk → index (mapper)
    results = []
    total_chunks = 0
    errors = []

    for filepath in files:
        try:
            result = _ingest_single_file(filepath, collection, chunk_size, chunk_overlap)
            results.append(result)
            total_chunks += result["chunks"]
        except Exception as e:
            errors.append({"file": filepath, "error": str(e)})

    # 4. 汇总 (reducer)
    report = (
        f"✅ 索引完成\n"
        f"  文件数: {len(results)}\n"
        f"  总块数: {total_chunks}\n"
        f"  知识库: {collection}\n"
    )

    if results:
        report += "\n  文件清单:\n"
        for r in results:
            report += f"    {r['file']} ({r['type']}) → {r['chunks']} 个块\n"

    if errors:
        report += f"\n  ⚠️ {len(errors)} 个文件失败:\n"
        for e in errors:
            report += f"    {e['file']}: {e['error']}\n"

    return report


def _collect_files(path: str) -> List[str]:
    """收集文件列表，支持单文件/目录/glob。"""
    path = os.path.expanduser(path.strip())

    # Glob 模式
    if "*" in path or "?" in path:
        return [f for f in glob.glob(path, recursive=True)
                if os.path.isfile(f) and _is_supported(f)]

    # 单文件
    if os.path.isfile(path):
        return [path] if _is_supported(path) else []

    # 目录
    if os.path.isdir(path):
        files = []
        for root, dirs, filenames in os.walk(path):
            # 跳过隐藏目录和常见忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d not in ('node_modules', '__pycache__', '.git', 'venv')]
            for name in sorted(filenames):
                full = os.path.join(root, name)
                if _is_supported(full):
                    files.append(full)
        return files

    return []


def _is_supported(filepath: str) -> bool:
    """检查文件扩展名是否支持。"""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def _detect_chunk_strategy(filepath: str) -> str:
    """根据文件类型选择分块策略。"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp',
               '.rb', '.php', '.swift', '.kt'):
        return "paragraph"  # 代码文件按空行分段（近似函数边界）
    elif ext in ('.md', '.html', '.xml'):
        return "heading"    # Markdown/HTML 按标题分块
    else:
        return "paragraph"  # 默认按段落


def _ingest_single_file(filepath: str, collection: str,
                         chunk_size: int, chunk_overlap: int) -> Dict:
    """处理单个文件: read → chunk → index。"""
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    # 读取
    if ext == ".pdf":
        content = _read_pdf(filepath)
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    if not content.strip():
        return {"file": filename, "type": ext, "chunks": 0, "skipped": True}

    # 分块
    strategy = _detect_chunk_strategy(filepath)
    chunks = ChunkSplitter.split(content, strategy=strategy,
                                  chunk_size=chunk_size, overlap=chunk_overlap)

    if not chunks:
        return {"file": filename, "type": ext, "chunks": 0}

    # 索引 — 每个块带元数据
    for i, chunk in enumerate(chunks):
        kb_manage(json.dumps({
            "action": "add",
            "name": collection,
            "content": chunk,
            "metadata": {
                "source": filename,
                "file_path": filepath,
                "file_type": ext,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "strategy": strategy,
            },
        }))

    return {"file": filename, "type": ext, "chunks": len(chunks)}


def _read_pdf(filepath: str) -> str:
    """读取 PDF 文件。"""
    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(pages)
    except ImportError:
        return f"[ERROR] 需要安装 PyPDF2 才能读取 PDF: pip install PyPDF2"
    except Exception as e:
        return f"[ERROR] PDF 读取失败: {e}"


def _ensure_kb(collection: str):
    """确保知识库存在，不存在则创建。"""
    try:
        info = kb_manage(json.dumps({"action": "info", "name": collection}))
        if "not found" in info.lower() or "不存在" in info:
            kb_manage(json.dumps({
                "action": "create",
                "name": collection,
                "description": f"QA Agent 知识库: {collection}",
            }))
    except Exception:
        try:
            kb_manage(json.dumps({
                "action": "create",
                "name": collection,
                "description": f"QA Agent 知识库: {collection}",
            }))
        except Exception:
            pass  # 可能已存在


# ════════════════════════════════════════════════════════════
# QueryKnowledge — 问答 (pipeline + review pattern)
# ════════════════════════════════════════════════════════════

def query_knowledge(input_val: Any) -> str:
    """
    基于知识库回答问题。

    Pattern: pipeline(retriever >> answerer >> citer) + review(fact_checker)
      1. 检索: 从知识库找 top-k 相关块
      2. 回答: 基于上下文生成回答 (LLM)
      3. 引用: 标注来源
      4. 核查: 验证回答有原文支撑

    输入 (JSON 或文本):
      {"query": "有哪些安全风险？", "collection": "my_kb", "top_k": 5}
      "有哪些安全风险？"

    输出: 带引用来源的回答
    """
    params = _parse(input_val)
    query = params.get("query", str(input_val).strip())
    collection = params.get("collection", DEFAULT_KB)
    top_k = params.get("top_k", 5)

    # Step 1: 检索 (context-retriever skill)
    search_result = kb_manage(json.dumps({
        "action": "search",
        "name": collection,
        "query": query,
        "top_k": top_k,
    }))

    # 解析检索结果
    try:
        search_data = json.loads(search_result)
        if isinstance(search_data, dict) and "results" in search_data:
            results = search_data["results"]
        else:
            results = []
    except (json.JSONDecodeError, TypeError):
        # 纯文本结果
        if "未找到" in search_result or "no results" in search_result.lower():
            return (
                "📭 文档中未找到与该问题直接相关的内容。\n\n"
                "建议:\n"
                "  - 换一种方式描述问题\n"
                "  - 检查是否已加入相关文件 (调用 ListKnowledge)\n"
                "  - 加入更多相关文件 (调用 IngestFiles)"
            )
        results = [{"content": search_result, "source": "unknown"}]

    if not results:
        return "📭 知识库中暂无相关内容。请先使用 IngestFiles 加入文件。"

    # Step 2: 构建上下文
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        content = r.get("content", r.get("text", str(r)))
        source = r.get("metadata", {}).get("source", r.get("source", f"块{i+1}"))
        chunk_idx = r.get("metadata", {}).get("chunk_index", "")
        score = r.get("score", r.get("relevance", 0))

        source_label = source
        if chunk_idx:
            source_label = f"{source} #块{chunk_idx}"

        context_parts.append(f"[参考 {i+1} | 来源: {source_label}]\n{content}")
        sources.append({"source": source, "chunk": chunk_idx, "score": score})

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: 生成回答 (answer-generator skill — LLM 调用)
    from lambdagent.core import Context
    ctx = Context()

    answer_prompt = f"参考文档:\n{context}\n\n问题: {query}"
    try:
        answer_skill = _get_skill("answer-generator")
        raw_answer = answer_skill.apply(answer_prompt, ctx)
    except Exception as e:
        # LLM 不可用时降级为返回检索原文
        return (
            f"📚 检索到 {len(results)} 条相关内容 (LLM 不可用，返回原文):\n\n"
            f"{context}\n\n"
            f"来源文件: {', '.join(set(s['source'] for s in sources))}\n"
            f"⚠️ LLM 回答生成失败: {e}"
        )

    # Step 4: 标注来源 (source-citer skill — LLM 调用)
    cite_prompt = f"回答:\n{raw_answer}\n\n原始文档块:\n{context}"
    try:
        citer_skill = _get_skill("source-citer")
        cited_answer = citer_skill.apply(cite_prompt, ctx)
    except Exception:
        cited_answer = raw_answer  # 标注失败则返回无引用的回答

    # Step 5: 事实核查 (fact-checker skill — review pattern, 最多 2 轮)
    check_prompt = f"回答:\n{cited_answer}\n\n原文:\n{context}"
    verified = False
    final_answer = cited_answer

    try:
        checker_skill = _get_skill("fact-checker")
        for round_i in range(2):  # review pattern: max 2 rounds
            verdict = checker_skill.apply(check_prompt, ctx)
            if "VERIFIED" in str(verdict):
                verified = True
                break
            else:
                # REJECTED — 重新生成 (review pattern 的重试逻辑)
                retry_prompt = (
                    f"你的上一次回答被核查员拒绝。\n"
                    f"拒绝原因: {verdict}\n\n"
                    f"参考文档:\n{context}\n\n"
                    f"问题: {query}\n\n"
                    f"请修正回答，确保每个论点有原文支撑。"
                )
                raw_answer = answer_skill.apply(retry_prompt, ctx)
                cited_answer = citer_skill.apply(
                    f"回答:\n{raw_answer}\n\n原始文档块:\n{context}", ctx
                )
                final_answer = cited_answer
                check_prompt = f"回答:\n{cited_answer}\n\n原文:\n{context}"
    except Exception:
        pass  # 核查失败不阻塞回答

    # 组装最终输出
    source_list = ', '.join(set(s['source'] for s in sources))
    status = "✅ 已通过事实核查" if verified else "⚠️ 未经事实核查验证"

    return (
        f"📚 回答 (基于 {len(results)} 条检索结果):\n\n"
        f"{final_answer}\n\n"
        f"---\n"
        f"来源文件: {source_list}\n"
        f"{status}"
    )


# ════════════════════════════════════════════════════════════
# ListKnowledge — 查看已索引文件
# ════════════════════════════════════════════════════════════

def list_knowledge(input_val: Any) -> str:
    """
    查看知识库索引状态。

    输入 (JSON 或文本):
      {"collection": "my_kb"}
      "my_kb"
      "" (使用默认知识库)

    输出: 已索引文件清单和统计信息
    """
    params = _parse(input_val)
    collection = params.get("collection", params.get("name", DEFAULT_KB))

    # 获取知识库信息
    info = kb_manage(json.dumps({"action": "info", "name": collection}))
    return f"📋 知识库状态 ({collection}):\n\n{info}"


# ════════════════════════════════════════════════════════════
# RemoveKnowledge — 删除文件索引
# ════════════════════════════════════════════════════════════

def remove_knowledge(input_val: Any) -> str:
    """
    从知识库删除指定文件的索引。

    输入 (JSON 或文本):
      {"collection": "my_kb", "source": "old-file.md"}
      {"name": "my_kb"}  (删除整个知识库)

    输出: 删除确认
    """
    params = _parse(input_val)
    collection = params.get("collection", params.get("name", DEFAULT_KB))
    source = params.get("source", params.get("file", ""))

    if source:
        # 删除特定文件的所有块 (需要知识库支持按 metadata 过滤删除)
        # 当前简化实现: 提示用户
        return (
            f"⚠️ 按文件删除功能需要重建知识库。\n"
            f"建议: 删除整个知识库后重新索引（排除不需要的文件）:\n"
            f'  1. RemoveKnowledge {{"name": "{collection}"}}\n'
            f'  2. IngestFiles {{"path": "/your/docs", "collection": "{collection}"}}'
        )
    else:
        # 删除整个知识库
        result = kb_manage(json.dumps({"action": "delete", "name": collection}))
        return f"🗑️ {result}"


# ════════════════════════════════════════════════════════════
# DeepAnalysis — 复杂推理 (SubAgent 编排, fan_out_merge pattern)
# ════════════════════════════════════════════════════════════

def deep_analysis(input_val: Any) -> str:
    """
    复杂问题的深度分析 — 启动子智能体并行处理。

    Pattern: fan_out_merge(retriever_agents, synthesizer_agent)
      1. 解析问题 → 拆分为多个检索子任务
      2. 多个 retriever 子智能体并行检索 (fan_out)
      3. analyzer 子智能体对各路结果深度分析
      4. synthesizer 子智能体合并所有分析 (merge)

    适用场景 (主 Agent 判断后调用):
      - "对比 A 和 B 的差异"
      - "从安全、性能两个角度评估"
      - "总结所有文件的核心观点"

    输入 (JSON 或文本):
      {"query": "对比模块 A 和 B 的架构差异", "collection": "my_kb", "perspectives": ["架构", "性能"]}
      "对比文档 A 和文档 B"

    输出: 综合分析报告（带引用来源）
    """
    params = _parse(input_val)
    query = params.get("query", str(input_val).strip())
    collection = params.get("collection", DEFAULT_KB)
    perspectives = params.get("perspectives", None)

    # Step 1: 任务拆分 — 将复杂问题拆为多个检索子任务
    sub_queries = _decompose_query(query, perspectives)

    # Step 2: 并行检索 (fan_out — 多路 retriever)
    from concurrent.futures import ThreadPoolExecutor

    retrieval_results = {}

    def _retrieve(sub_query: str) -> Dict:
        result = kb_manage(json.dumps({
            "action": "search",
            "name": collection,
            "query": sub_query,
            "top_k": 5,
        }))
        return {"query": sub_query, "result": result}

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
        futures = {pool.submit(_retrieve, sq): sq for sq in sub_queries}
        for future in futures:
            try:
                res = future.result(timeout=30)
                retrieval_results[res["query"]] = res["result"]
            except Exception as e:
                retrieval_results[futures[future]] = f"[检索失败: {e}]"

    # Step 3: 组装分析上下文
    analysis_context = []
    for i, (sq, result) in enumerate(retrieval_results.items()):
        analysis_context.append(
            f"## 检索维度 {i+1}: {sq}\n\n"
            f"检索结果:\n{result}\n"
        )

    full_context = "\n---\n\n".join(analysis_context)

    # Step 4: 深度分析 (deep-analyzer skill — LLM 调用)
    from lambdagent.core import Context

    analyze_prompt = (
        f"原始问题: {query}\n\n"
        f"以下是 {len(sub_queries)} 路并行检索的结果:\n\n"
        f"{full_context}\n\n"
        f"请进行深度分析，引用原文，标注来源，给出结论。"
    )

    try:
        analyzer = _get_skill("deep-analyzer")
        ctx = Context()
        analysis = analyzer.apply(analyze_prompt, ctx)
    except Exception as e:
        # LLM 不可用时降级为返回原始检索素材
        analysis = (
            f"⚠️ 深度分析 LLM 不可用 ({e})，返回原始检索素材:\n\n"
            f"{full_context}"
        )

    # Step 5: 综合 (synthesizer skill — LLM 调用)
    try:
        synthesizer = _get_skill("synthesizer")
        synth_prompt = (
            f"原始问题: {query}\n\n"
            f"深度分析结果:\n{analysis}\n\n"
            f"请综合以上内容，输出一份完整、结构化的最终报告。"
            f"保留来源引用，给出明确结论。"
        )
        final_report = synthesizer.apply(synth_prompt, ctx)
    except Exception:
        final_report = analysis  # 综合失败则返回分析结果

    report = (
        f"🔬 深度分析完成 ({len(sub_queries)} 路并行检索 + 分析 + 综合)\n\n"
        f"{final_report}"
    )

    return report


def _decompose_query(query: str, perspectives: Optional[List[str]] = None) -> List[str]:
    """
    将复杂问题拆分为多个子检索任务。

    策略:
      1. 如果用户指定了 perspectives → 每个角度一个子任务
      2. 如果问题含"对比/比较" → 拆为两个目标各自检索
      3. 如果问题含"所有/全部/总结" → 按知识库文件分组检索
      4. 默认 → 原问题 + 2 个补充角度
    """
    if perspectives:
        return [f"{query} — 聚焦: {p}" for p in perspectives]

    query_lower = query.lower()

    # 对比型: "对比 A 和 B" → 拆成两路
    compare_keywords = ["对比", "比较", "区别", "差异", "异同", "vs", "versus"]
    if any(k in query_lower for k in compare_keywords):
        return [
            f"{query} — 重点关注第一个对象的特点和优势",
            f"{query} — 重点关注第二个对象的特点和优势",
            f"{query} — 关注两者的共同点和相似之处",
        ]

    # 总结型: "总结/概述所有" → 多角度
    summary_keywords = ["总结", "概述", "概括", "归纳", "综述", "全面"]
    if any(k in query_lower for k in summary_keywords):
        return [
            f"{query} — 核心概念和定义",
            f"{query} — 具体实现和方法",
            f"{query} — 优缺点和适用场景",
        ]

    # 评估型: "评估/分析" → 多维度
    eval_keywords = ["评估", "分析", "评价", "审查", "检查", "风险"]
    if any(k in query_lower for k in eval_keywords):
        return [
            f"{query} — 优势和亮点",
            f"{query} — 问题和风险",
            f"{query} — 改进建议",
        ]

    # 默认: 原始问题 + 补充检索
    return [
        query,
        f"{query} — 相关背景和上下文",
    ]
