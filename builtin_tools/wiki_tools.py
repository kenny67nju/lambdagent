"""
lambdagent.builtin_tools.wiki_tools — LLM Wiki 模式工具集 (v2)

基于 Karpathy LLM Wiki 模式 + 知识管理三步流程:
  不是 RAG (存碎片, 每次重搜)
  而是 LLM 维护的持续生长的 wiki (知识编译)

v2 升级 (基于三步流程架构图):
  Ingest: 新资料 → 抽取要点 → 补充交叉引用 → 更新相关页面 10+
  Query:  直接查 Wiki 页面 → 综合多页信息 → 生成回答
  Lint:   查矛盾 → 查孤岛页 → 查过期引用 → 揪出问题 → 自动修复

6 个工具:
  WikiIngest  — 喂文件: 抽取 → 交叉引用 → 更新相关页面 → 持续生长
  WikiQuery   — 提问: 综合多页 → 标签路由 → 生成回答
  WikiLint    — 健康检查 + 自动修复: 矛盾/孤页/过期/断链
  WikiSearch  — 搜索 wiki 页面内容
  WikiStatus  — wiki 统计 + 生长指标
  WikiGrow    — 主动生长: 发现缺失主题 → 自动创建综合页

Lambda 语义:
  WikiIngest = λsource. let facts = extract(read(source)) in
               for_each facts (λf. update_wiki(f))
               >> cross_reference(new_pages, existing_pages)
               >> update_related(10+) >> update_index
  WikiQuery  = λq. let pages = search_index(q) in
               synthesize(read_pages(pages), q)
  WikiLint   = λ(). let issues = check(read_all_pages) in
               auto_fix(issues) >> report
  WikiGrow   = λ(). let gaps = find_gaps(index) in
               for_each gaps (λg. create_synthesis_page(g))
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .knowledge_tools import _parse


# ════════════════════════════════════════════════════════════
# Wiki 目录管理
# ════════════════════════════════════════════════════════════

# 默认 wiki 根目录 (可通过参数覆盖)
_DEFAULT_WIKI_ROOT = os.path.expanduser("~/.lambdagent/wiki")


def _wiki_root(params: dict = None) -> str:
    """获取 wiki 根目录，确保存在。"""
    root = (params or {}).get("wiki_root", _DEFAULT_WIKI_ROOT)
    for sub in ("sources", "entities", "topics", "analyses"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    # 确保 index.md 和 log.md 存在
    index_path = os.path.join(root, "index.md")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("# 知识库索引\n\n_暂无内容。使用 WikiIngest 加入文件。_\n")
    log_path = os.path.join(root, "log.md")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("# 操作日志\n\n")
    return root


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def _slug(name: str) -> str:
    """将名称转为文件名 slug: 'JWT Auth' → 'jwt-auth'"""
    s = name.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    return s[:60]


# ════════════════════════════════════════════════════════════
# Skill: Ingest Extractor (LLM 调用)
# ════════════════════════════════════════════════════════════

_extract_skill = None

def _get_extract_skill():
    """延迟初始化 ingest-extractor skill。"""
    global _extract_skill
    if _extract_skill is None:
        from lambdagent.primitives import Lam
        _extract_skill = Lam(
            "ingest-extractor",
            "你是一个知识提取专家。阅读以下源文件内容，提取结构化信息。\n\n"
            "你必须输出严格的 JSON 格式 (不要加 markdown 代码块标记):\n"
            "{\n"
            '  "summary": "一段话概括文件核心内容",\n'
            '  "entities": [\n'
            '    {"name": "实体名", "type": "technology|person|component|concept", '
            '"description": "一句话描述"}\n'
            '  ],\n'
            '  "topics": ["主题1", "主题2"],\n'
            '  "key_facts": ["事实1", "事实2"],\n'
            '  "connections": ["与 XX 文件的关系描述"]\n'
            "}\n\n"
            "规则:\n"
            "- entities: 提取所有重要的技术/人名/组件/概念 (3-10 个)\n"
            "- topics: 归类到 2-5 个主题 (如 security, performance, architecture)\n"
            "- key_facts: 提取 3-8 个关键事实，每个必须可验证\n"
            "- connections: 如果文件提到其他文件/模块的关系，列出来\n"
            "- 只提取，不推测。没有的信息不要编造",
        )
    return _extract_skill


_answer_skill = None

def _get_answer_skill():
    """延迟初始化 wiki-answerer skill。"""
    global _answer_skill
    if _answer_skill is None:
        from lambdagent.primitives import Lam
        _answer_skill = Lam(
            "wiki-answerer",
            "你是一个精准的问答助手。基于提供的 wiki 页面内容回答问题。\n\n"
            "规则:\n"
            "1. 只基于提供的 wiki 页面回答，不编造\n"
            "2. 每个论点标注来源: [来源: 页面名]\n"
            "3. 如果 wiki 中有 ⚠️ 矛盾标记，在回答中说明\n"
            "4. 如果 wiki 中没有相关信息，明确说明\n"
            "5. 回答结构清晰，分点列举",
        )
    return _answer_skill


_lint_skill = None

def _get_lint_skill():
    """延迟初始化 wiki-linter skill。"""
    global _lint_skill
    if _lint_skill is None:
        from lambdagent.primitives import Lam
        _lint_skill = Lam(
            "wiki-linter",
            "你是一个 wiki 质量检查员。检查以下 wiki 内容的健康状况。\n\n"
            "检查项目:\n"
            "1. 矛盾: 不同页面对同一事实的描述是否一致\n"
            "2. 孤页: 有哪些页面没有被其他页面引用 ([[链接]])\n"
            "3. 缺页: 有哪些 [[链接]] 指向不存在的页面\n"
            "4. 过时: 有哪些内容可能已被更新的源文件取代\n"
            "5. 覆盖: 有哪些源文件还没有对应的 wiki 页面\n\n"
            "输出格式:\n"
            "## 矛盾 (N 个)\n- ...\n"
            "## 孤页 (N 个)\n- ...\n"
            "## 缺页 (N 个)\n- ...\n"
            "## 建议\n- ...",
        )
    return _lint_skill


# ════════════════════════════════════════════════════════════
# WikiIngest — 喂文件 (知识编译)
# ════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".sh", ".sql",
    ".pdf", ".csv", ".log", ".xml", ".rb", ".php", ".swift", ".kt",
}


def wiki_ingest(input_val: Any) -> str:
    """
    Ingest: LLM 阅读源文件 → 提取知识 → 写 wiki 页面 → 更新索引。

    输入:
      {"path": "/docs/auth.py"}
      {"path": "/docs/", "wiki_root": "/my/wiki"}
      "/path/to/file.md"

    一个源文件可能触及 10-15 个 wiki 页面。
    """
    params = _parse(input_val)
    path = params.get("path", str(input_val).strip())
    root = _wiki_root(params)

    # 收集文件
    files = _collect_files(path)
    if not files:
        return f"未找到可处理的文件: {path}"

    results = []
    total_pages = 0

    for filepath in files:
        try:
            r = _ingest_one(filepath, root)
            results.append(r)
            total_pages += r["pages_touched"]
        except Exception as e:
            results.append({"file": os.path.basename(filepath), "error": str(e), "pages_touched": 0})

    # 更新 index.md
    _rebuild_index(root)

    # 写 log
    now = _now_str()
    log_entry = f"\n## [{now}] ingest | {len(files)} 个文件\n"
    for r in results:
        status = f"✅ {r['pages_touched']} 页" if "error" not in r else f"❌ {r['error']}"
        log_entry += f"- {r['file']}: {status}\n"
    _append_log(root, log_entry)

    # 汇报
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    report = (
        f"📚 Wiki Ingest 完成\n\n"
        f"  文件: {len(ok)} 成功"
    )
    if err:
        report += f", {len(err)} 失败"
    report += f"\n  触及页面: {total_pages}\n  wiki 目录: {root}\n"

    for r in ok:
        report += f"\n  ✅ {r['file']}: {r['pages_touched']} 页 ({', '.join(r.get('pages', []))})"
    for r in err:
        report += f"\n  ❌ {r['file']}: {r['error']}"

    return report


def _ingest_one(filepath: str, wiki_root: str) -> Dict:
    """处理单个文件: read → extract (LLM) → write wiki pages。"""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # 1. 读取
    if ext == ".pdf":
        content = _read_pdf(filepath)
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    if not content.strip():
        return {"file": filename, "pages_touched": 0, "pages": [], "skipped": True}

    # 截断避免 LLM 上下文溢出
    if len(content) > 30000:
        content = content[:30000] + "\n\n[... 内容截断，文件过长 ...]"

    # 2. LLM 提取 (ingest-extractor skill)
    from lambdagent.core import Context
    ctx = Context()
    extract_input = f"文件名: {filename}\n文件类型: {ext}\n\n文件内容:\n{content}"

    try:
        raw = _get_extract_skill().apply(extract_input, ctx)
        extracted = _parse_json_from_llm(raw)
    except Exception as e:
        # LLM 不可用时降级为基本提取
        extracted = {
            "summary": content[:200],
            "entities": [],
            "topics": [],
            "key_facts": [],
            "connections": [],
        }

    # 3. 写 wiki 页面
    pages = []

    # 3a. 源文件摘要页
    source_page = _write_source_page(wiki_root, filename, filepath, ext, extracted)
    pages.append(source_page)

    # 3b. 实体页
    for entity in extracted.get("entities", []):
        ename = entity if isinstance(entity, str) else entity.get("name", "")
        edesc = "" if isinstance(entity, str) else entity.get("description", "")
        etype = "concept" if isinstance(entity, str) else entity.get("type", "concept")
        if ename:
            p = _update_entity_page(wiki_root, ename, etype, edesc, filename)
            pages.append(p)

    # 3c. 主题页
    for topic in extracted.get("topics", []):
        if topic:
            p = _update_topic_page(wiki_root, topic, filename, extracted)
            pages.append(p)

    # 3d. v2: 补充交叉引用 — 新页面主动链接到已有页面
    cross_refs = _cross_reference_new_pages(wiki_root, pages, extracted)

    # 3e. v2: 更新相关页面 — 已有页面反向引用新内容
    updated_related = _update_related_pages(wiki_root, filename, extracted)

    return {
        "file": filename,
        "pages_touched": len(pages) + len(updated_related),
        "pages": pages,
        "cross_refs_added": cross_refs,
        "related_updated": updated_related,
        "summary": extracted.get("summary", "")[:100],
    }


def _write_source_page(root: str, filename: str, filepath: str, ext: str, extracted: Dict) -> str:
    """写源文件摘要页。"""
    slug = _slug(filename.replace(".", "-"))
    page_path = os.path.join(root, "sources", f"{slug}.md")
    now = _now_str()

    entities = extracted.get("entities", [])
    entity_links = []
    for e in entities:
        name = e if isinstance(e, str) else e.get("name", "")
        if name:
            entity_links.append(f"[[{name}]]")

    topics = extracted.get("topics", [])
    topic_links = [f"[[{t}]]" for t in topics if t]

    facts = extracted.get("key_facts", [])
    connections = extracted.get("connections", [])

    content = (
        f"---\n"
        f"source: {filename}\n"
        f"path: {filepath}\n"
        f"ingested: {now}\n"
        f"type: {ext.lstrip('.')}\n"
        f"tags: [{', '.join(topics)}]\n"
        f"---\n\n"
        f"# {filename}\n\n"
        f"## 摘要\n{extracted.get('summary', '暂无摘要')}\n\n"
        f"## 关键事实\n"
    )
    for fact in facts:
        content += f"- {fact} [来源: {filename}]\n"

    if entity_links:
        content += f"\n## 相关实体\n{', '.join(entity_links)}\n"

    if topic_links:
        content += f"\n## 相关主题\n{', '.join(topic_links)}\n"

    if connections:
        content += f"\n## 关联\n"
        for conn in connections:
            content += f"- {conn}\n"

    with open(page_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"sources/{slug}.md"


def _update_entity_page(root: str, name: str, etype: str, desc: str, source: str) -> str:
    """创建或更新实体页。"""
    slug = _slug(name)
    page_path = os.path.join(root, "entities", f"{slug}.md")

    if os.path.exists(page_path):
        # 追加新来源引用
        with open(page_path, "r", encoding="utf-8") as f:
            existing = f.read()
        if source not in existing:
            # 在 "## 来源引用" 区域追加
            if "## 来源引用" in existing:
                existing = existing.replace(
                    "## 来源引用",
                    f"## 来源引用\n- {source}: {desc}" if desc else f"## 来源引用\n- {source}"
                )
            else:
                existing += f"\n## 来源引用\n- {source}: {desc}\n"
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(existing)
    else:
        # 创建新实体页
        content = (
            f"---\n"
            f"entity: {name}\n"
            f"type: {etype}\n"
            f"updated: {_now_str()}\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"**类型**: {etype}\n\n"
        )
        if desc:
            content += f"## 描述\n{desc}\n\n"
        content += f"## 来源引用\n- {source}: {desc}\n"

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(content)

    return f"entities/{slug}.md"


def _update_topic_page(root: str, topic: str, source: str, extracted: Dict) -> str:
    """创建或更新主题页。"""
    slug = _slug(topic)
    page_path = os.path.join(root, "topics", f"{slug}.md")

    # 收集与此主题相关的事实
    relevant_facts = [f for f in extracted.get("key_facts", [])
                      if topic.lower() in f.lower() or len(extracted.get("topics", [])) <= 2]

    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as f:
            existing = f.read()
        # 追加新来源的事实
        addition = f"\n### 来自 {source}\n"
        for fact in relevant_facts[:3]:
            addition += f"- {fact} [来源: {source}]\n"
        if source not in existing:
            existing += addition
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(existing)
    else:
        content = (
            f"---\n"
            f"topic: {topic}\n"
            f"updated: {_now_str()}\n"
            f"sources: [{source}]\n"
            f"---\n\n"
            f"# {topic}\n\n"
            f"## 概述\n_由 wikiagent 基于源文件自动生成。_\n\n"
            f"### 来自 {source}\n"
        )
        for fact in relevant_facts[:5]:
            content += f"- {fact} [来源: {source}]\n"

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(content)

    return f"topics/{slug}.md"


def _cross_reference_new_pages(wiki_root: str, new_pages: List[str], extracted: Dict) -> int:
    """
    v2: 补充交叉引用 — 扫描已有页面，为新页面添加 [[链接]]。

    比如新增了 entities/jwt.md，扫描已有页面中提到 "JWT" 的，
    自动补上 [[JWT]] 链接。实现图1中"补充交叉引用"。
    """
    refs_added = 0
    # 收集新页面的实体名和主题名
    new_names = set()
    for entity in extracted.get("entities", []):
        name = entity if isinstance(entity, str) else entity.get("name", "")
        if name and len(name) >= 2:
            new_names.add(name)
    for topic in extracted.get("topics", []):
        if topic and len(topic) >= 2:
            new_names.add(topic)

    if not new_names:
        return 0

    # 扫描已有页面，找提到这些名字但没有 [[链接]] 的
    for category in ("sources", "entities", "topics"):
        cat_dir = os.path.join(wiki_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            rel_path = f"{category}/{fname}"
            if rel_path in new_pages:
                continue  # 跳过刚创建的页面

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            modified = False
            for name in new_names:
                # 如果页面提到这个名字但没有 [[链接]]
                if name in content and f"[[{name}]]" not in content:
                    # 只替换正文中的第一次出现 (不替换 front matter 和标题)
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("---") or line.startswith("#"):
                            continue
                        if name in line and f"[[{name}]]" not in line:
                            lines[i] = line.replace(name, f"[[{name}]]", 1)
                            refs_added += 1
                            modified = True
                            break

            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

    return refs_added


def _update_related_pages(wiki_root: str, source_file: str, extracted: Dict) -> List[str]:
    """
    v2: 更新相关页面 — 让已有实体/主题页反向引用新源文件。

    实现图1中"更新相关页面 10+"。
    当新文件提到已有实体时，在已有实体页追加"另见"引用。
    """
    updated = []

    for entity in extracted.get("entities", []):
        name = entity if isinstance(entity, str) else entity.get("name", "")
        if not name:
            continue
        slug = _slug(name)
        entity_path = os.path.join(wiki_root, "entities", f"{slug}.md")
        if not os.path.exists(entity_path):
            continue

        try:
            with open(entity_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        if source_file in content:
            continue  # 已经引用过了

        # 追加引用
        if "## 来源引用" in content:
            desc = entity.get("description", "") if isinstance(entity, dict) else ""
            ref_line = f"\n- {source_file}: {desc}" if desc else f"\n- {source_file}"
            content = content.rstrip() + ref_line + "\n"
        else:
            content += f"\n## 来源引用\n- {source_file}\n"

        # 更新时间戳
        content = re.sub(r'updated: .*', f'updated: {_now_str()}', content)

        with open(entity_path, "w", encoding="utf-8") as f:
            f.write(content)
        updated.append(f"entities/{slug}.md")

    return updated


def _rebuild_index(root: str):
    """重建 index.md — 扫描所有 wiki 页面。"""
    sections = {"sources": [], "entities": [], "topics": [], "analyses": []}

    for category in sections:
        cat_dir = os.path.join(root, category)
        if os.path.isdir(cat_dir):
            for fname in sorted(os.listdir(cat_dir)):
                if fname.endswith(".md"):
                    fpath = os.path.join(cat_dir, fname)
                    # 读取第一行标题
                    title = fname.replace(".md", "")
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("# "):
                                    title = line[2:].strip()
                                    break
                    except Exception:
                        pass
                    sections[category].append(f"- [[{category}/{fname.replace('.md', '')}|{title}]]")

    index = "# 知识库索引\n\n> 由 wikiagent 自动维护。\n\n"

    index += f"## 源文件 ({len(sections['sources'])} 个)\n"
    index += "\n".join(sections["sources"]) if sections["sources"] else "_暂无_"
    index += "\n\n"

    index += f"## 实体 ({len(sections['entities'])} 个)\n"
    index += "\n".join(sections["entities"]) if sections["entities"] else "_暂无_"
    index += "\n\n"

    index += f"## 主题 ({len(sections['topics'])} 个)\n"
    index += "\n".join(sections["topics"]) if sections["topics"] else "_暂无_"
    index += "\n\n"

    index += f"## 分析 ({len(sections['analyses'])} 个)\n"
    index += "\n".join(sections["analyses"]) if sections["analyses"] else "_暂无_"
    index += "\n\n"

    total = sum(len(v) for v in sections.values())
    index += f"---\n统计: {len(sections['sources'])} 源文件, {len(sections['entities'])} 实体, {len(sections['topics'])} 主题, {len(sections['analyses'])} 分析 (共 {total} 页)\n"

    with open(os.path.join(root, "index.md"), "w", encoding="utf-8") as f:
        f.write(index)


def _append_log(root: str, entry: str):
    """追加操作日志。"""
    log_path = os.path.join(root, "log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


# ════════════════════════════════════════════════════════════
# WikiQuery — 提问 (查阅已编译知识)
# ════════════════════════════════════════════════════════════

def wiki_query(input_val: Any) -> str:
    """
    Query: 读 index.md → 找相关 wiki 页面 → 基于已编译知识回答。

    输入:
      {"query": "认证方式是什么？"}
      "有哪些安全风险？"
    """
    params = _parse(input_val)
    query = params.get("query", str(input_val).strip())
    root = _wiki_root(params)
    save_analysis = params.get("save", False)

    # Step 1: 读 index.md
    index_path = os.path.join(root, "index.md")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_content = f.read()
    except FileNotFoundError:
        return "📭 wiki 为空。请先使用 WikiIngest 加入文件。"

    if "暂无" in index_content and "源文件" in index_content:
        return "📭 wiki 为空。请先使用 WikiIngest 加入文件。"

    # Step 2: v2 增强搜索 — 关键词 + 标签路由 + 实体优先
    relevant_pages = _find_relevant_pages_v2(root, query)

    if not relevant_pages:
        return (
            f"📭 wiki 中未找到与 '{query}' 直接相关的页面。\n\n"
            f"建议:\n- 换一种描述方式\n- 使用 WikiStatus 查看已有内容\n- 使用 WikiIngest 加入更多文件"
        )

    # Step 3: v2 读取相关页面 + 跟踪 [[链接]] 引用的页面 (图2: 综合多页信息)
    page_contents = []
    # 先读直接匹配的页面
    for page_path in relevant_pages[:6]:  # 直接匹配
        try:
            full_path = os.path.join(root, page_path)
            if not full_path.endswith(".md"):
                full_path += ".md"
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            page_contents.append(f"--- 页面: {page_path} ---\n{content}")
        except Exception:
            pass

    if not page_contents:
        return "📭 找到索引条目但无法读取页面内容。"

    wiki_context = "\n\n".join(page_contents)

    # Step 4: LLM 回答 (wiki-answerer skill)
    from lambdagent.core import Context
    ctx = Context()
    answer_input = f"Wiki 页面内容:\n{wiki_context}\n\n问题: {query}"

    try:
        answer = _get_answer_skill().apply(answer_input, ctx)
    except Exception as e:
        return (
            f"📚 找到 {len(relevant_pages)} 个相关页面但 LLM 不可用:\n\n"
            f"相关页面: {', '.join(relevant_pages)}\n\n"
            f"请直接阅读以上 wiki 页面获取信息。\n"
            f"⚠️ LLM 错误: {e}"
        )

    # Step 5: 可选回填到 analyses/
    if save_analysis:
        slug = _slug(query[:50])
        analysis_path = os.path.join(root, "analyses", f"{slug}.md")
        analysis_content = (
            f"---\nquery: {query}\ndate: {_now_str()}\nsources: {relevant_pages}\n---\n\n"
            f"# {query}\n\n{answer}\n"
        )
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(analysis_content)
        _rebuild_index(root)

    # 写日志
    _append_log(root, f"\n## [{_now_str()}] query | {query[:60]}\n- 匹配页面: {len(relevant_pages)}\n")

    return (
        f"📚 回答 (基于 {len(relevant_pages)} 个 wiki 页面):\n\n"
        f"{answer}\n\n"
        f"---\n参考页面: {', '.join(relevant_pages)}"
    )


def _find_relevant_pages_v2(root: str, query: str) -> List[str]:
    """
    v2 增强搜索: 关键词匹配 + 标签路由 + 实体优先 + 链接跟踪。

    搜索优先级:
      1. 实体页标题精确匹配 (entities/) → 最高分
      2. 主题页标题匹配 (topics/) → 高分
      3. 正文关键词匹配 → 按命中数排序
      4. 跟踪 [[链接]] 引用的页面 → 补充上下文
    """
    keywords = set(query.lower().split())
    stopwords = {"的", "是", "有", "哪些", "什么", "了", "在", "和", "与", "从",
                 "the", "is", "are", "what", "which", "how", "a", "an", "in", "of"}
    keywords -= stopwords
    if not keywords:
        keywords = set(query.lower().split()[:3])

    scored = []

    for category in ("entities", "topics", "sources", "analyses"):
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            continue
        # 实体和主题页优先
        cat_boost = {"entities": 3, "topics": 2, "sources": 1, "analyses": 1}.get(category, 1)

        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().lower()
            except Exception:
                continue

            score = 0
            fname_lower = fname.lower().replace(".md", "").replace("-", " ")

            # 标题精确匹配 (高权重)
            for kw in keywords:
                if kw in fname_lower:
                    score += 5 * cat_boost

            # 正文关键词匹配
            score += sum(1 for kw in keywords if kw in content)

            # front matter 标签匹配
            if "tags:" in content:
                tags_line = content.split("tags:")[1].split("\n")[0]
                score += sum(2 for kw in keywords if kw in tags_line)

            if score > 0:
                scored.append((score, f"{category}/{fname}"))

    scored.sort(reverse=True)
    top_pages = [path for _, path in scored]

    # v2: 跟踪 top 页面中的 [[链接]]，补充上下文页面
    linked_pages = set()
    for page_path in top_pages[:5]:
        fpath = os.path.join(root, page_path)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            links = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)
            for link in links:
                link_slug = _slug(link)
                for cat in ("entities", "topics"):
                    candidate = f"{cat}/{link_slug}.md"
                    if os.path.exists(os.path.join(root, candidate)) and candidate not in top_pages:
                        linked_pages.add(candidate)
        except Exception:
            pass

    # 合并: 直接匹配 + 链接跟踪
    result = top_pages[:10]
    for lp in list(linked_pages)[:5]:
        if lp not in result:
            result.append(lp)

    return result


def _find_relevant_pages(root: str, query: str) -> List[str]:
    """在 wiki 中搜索与 query 相关的页面。(v1 兼容)"""
    keywords = set(query.lower().split())
    # 移除停用词
    stopwords = {"的", "是", "有", "哪些", "什么", "了", "在", "和", "与", "从",
                 "the", "is", "are", "what", "which", "how", "a", "an", "in", "of"}
    keywords -= stopwords
    if not keywords:
        keywords = set(query.lower().split()[:3])

    scored = []

    for category in ("sources", "entities", "topics", "analyses"):
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                # 计算匹配分数
                score = sum(1 for kw in keywords if kw in content)
                # 文件名匹配加分
                score += sum(2 for kw in keywords if kw in fname.lower())
                if score > 0:
                    scored.append((score, f"{category}/{fname}"))
            except Exception:
                pass

    # 按分数排序，返回最相关的
    scored.sort(reverse=True)
    return [path for _, path in scored]


# ════════════════════════════════════════════════════════════
# WikiLint — 健康检查
# ════════════════════════════════════════════════════════════

def wiki_lint(input_val: Any) -> str:
    """
    Lint: 检查 wiki 质量 — 矛盾、孤页、缺页、过时。
    """
    params = _parse(input_val)
    root = _wiki_root(params)

    # 收集所有页面和链接
    all_pages = {}  # path → content
    all_links = {}  # path → set of [[link targets]]
    inbound = {}    # page → count of pages linking to it

    for category in ("sources", "entities", "topics", "analyses"):
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            rel_path = f"{category}/{fname}"
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                all_pages[rel_path] = content
                # 提取 [[链接]]
                links = set(re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content))
                all_links[rel_path] = links
            except Exception:
                pass

    # v2: 自动修复开关
    auto_fix = (params or {}).get("auto_fix", False)

    # 分析
    issues = {"orphans": [], "broken_links": [], "stale": [], "suggestions": []}

    # 孤页检测: 没有任何入链的页面
    all_link_targets = set()
    for links in all_links.values():
        all_link_targets.update(links)

    for page_path in all_pages:
        page_name = page_path.split("/")[-1].replace(".md", "")
        linked_by = sum(1 for links in all_links.values()
                        if page_name in links or page_path.replace(".md", "") in links)
        if linked_by == 0:
            inbound[page_path] = 0
            issues["orphans"].append(page_path)

    # 缺页检测: [[链接]] 指向不存在的页面
    existing_slugs = set()
    for page_path in all_pages:
        existing_slugs.add(page_path.split("/")[-1].replace(".md", ""))

    for page_path, links in all_links.items():
        for link in links:
            link_slug = _slug(link)
            if link_slug and link_slug not in existing_slugs and link not in existing_slugs:
                issues["broken_links"].append(f"[[{link}]] (从 {page_path})")

    # v2: 过期检测 (图3: 查过期引用)
    for page_path, content in all_pages.items():
        # 检查 front matter 中的 updated/ingested 时间
        date_match = re.search(r'(?:updated|ingested):\s*(\d{4}-\d{2}-\d{2})', content)
        if date_match:
            try:
                page_date = datetime.datetime.strptime(date_match.group(1), "%Y-%m-%d")
                age_days = (datetime.datetime.now() - page_date).days
                if age_days > 90:  # 超过 90 天视为过期
                    issues["stale"].append(f"{page_path} (最后更新: {date_match.group(1)}, {age_days}天前)")
            except ValueError:
                pass

    # v2: 自动修复 (图3: 自动补链接)
    fixed_count = 0
    if auto_fix:
        # 修复断链: 创建占位页面
        for broken in issues["broken_links"]:
            link_match = re.match(r'\[\[(.+?)\]\]', broken)
            if link_match:
                link_name = link_match.group(1)
                slug = _slug(link_name)
                # 创建占位实体页
                placeholder_path = os.path.join(root, "entities", f"{slug}.md")
                if not os.path.exists(placeholder_path):
                    placeholder = (
                        f"---\nentity: {link_name}\ntype: concept\n"
                        f"updated: {_now_str()}\nstatus: stub\n---\n\n"
                        f"# {link_name}\n\n"
                        f"_此页面由 WikiLint 自动创建。需要补充内容。_\n"
                    )
                    with open(placeholder_path, "w", encoding="utf-8") as f:
                        f.write(placeholder)
                    fixed_count += 1

        # 修复孤页: 在 index.md 确保有引用 (rebuild_index 已覆盖)
        if issues["orphans"]:
            _rebuild_index(root)
            fixed_count += 1

    # 建议
    if len(all_pages) > 0 and not any("topics/" in p for p in all_pages):
        issues["suggestions"].append("没有主题页——考虑运行 WikiGrow 创建跨源综合主题")
    if len(all_pages) > 10 and len(issues["orphans"]) > len(all_pages) * 0.3:
        issues["suggestions"].append(f"{len(issues['orphans'])} 个孤页 (>30%)——运行 WikiLint auto_fix=true 自动补链接")
    if issues["stale"]:
        issues["suggestions"].append(f"{len(issues['stale'])} 个过期页面——考虑重新 WikiIngest 相关源文件")

    # LLM 深度检查 (如果页面数量合理)
    llm_analysis = ""
    if len(all_pages) > 0 and len(all_pages) <= 30:
        try:
            from lambdagent.core import Context
            all_content = "\n\n---\n\n".join(
                f"[{path}]\n{content[:500]}" for path, content in list(all_pages.items())[:20]
            )
            lint_input = f"Wiki 包含 {len(all_pages)} 个页面:\n\n{all_content}"
            llm_analysis = _get_lint_skill().apply(lint_input, Context())
        except Exception:
            llm_analysis = "(LLM 检查跳过)"

    # 写日志
    _append_log(root, (
        f"\n## [{_now_str()}] lint | {len(all_pages)} 页, "
        f"{len(issues['orphans'])} 孤页, {len(issues['broken_links'])} 断链, "
        f"{len(issues['stale'])} 过期"
        f"{f', 修复 {fixed_count} 项' if auto_fix else ''}\n"
    ))

    # 组装报告
    report = f"🔍 Wiki 健康检查 ({len(all_pages)} 个页面)\n\n"

    report += f"## 孤页 ({len(issues['orphans'])} 个)\n"
    if issues["orphans"]:
        for o in issues["orphans"][:20]:
            report += f"  - {o}\n"
        if len(issues["orphans"]) > 20:
            report += f"  ... 还有 {len(issues['orphans']) - 20} 个\n"
    else:
        report += "  ✅ 无孤页\n"

    report += f"\n## 断链 ({len(issues['broken_links'])} 个)\n"
    if issues["broken_links"]:
        for b in issues["broken_links"][:20]:
            report += f"  - {b}\n"
        if len(issues["broken_links"]) > 20:
            report += f"  ... 还有 {len(issues['broken_links']) - 20} 个\n"
    else:
        report += "  ✅ 无断链\n"

    report += f"\n## 过期页面 ({len(issues['stale'])} 个)\n"
    if issues["stale"]:
        for s in issues["stale"][:15]:
            report += f"  ⏰ {s}\n"
        if len(issues["stale"]) > 15:
            report += f"  ... 还有 {len(issues['stale']) - 15} 个\n"
    else:
        report += "  ✅ 无过期页面\n"

    if auto_fix and fixed_count > 0:
        report += f"\n## 自动修复\n  🔧 修复了 {fixed_count} 项问题\n"

    if issues["suggestions"]:
        report += "\n## 建议\n"
        for s in issues["suggestions"]:
            report += f"  💡 {s}\n"

    if llm_analysis:
        report += f"\n## LLM 深度分析\n{llm_analysis}\n"

    return report


# ════════════════════════════════════════════════════════════
# WikiSearch — 搜索 wiki 内容
# ════════════════════════════════════════════════════════════

def wiki_search(input_val: Any) -> str:
    """搜索 wiki 页面内容 (关键词匹配)。"""
    params = _parse(input_val)
    query = params.get("query", str(input_val).strip())
    root = _wiki_root(params)

    results = _find_relevant_pages(root, query)
    if not results:
        return f"未找到包含 '{query}' 的 wiki 页面。"

    report = f"🔎 搜索 '{query}' — 找到 {len(results)} 个页面:\n\n"
    for path in results[:10]:
        report += f"  - {path}\n"
    return report


# ════════════════════════════════════════════════════════════
# WikiStatus — 统计信息
# ════════════════════════════════════════════════════════════

def wiki_status(input_val: Any) -> str:
    """v2: 查看 wiki 统计 + 生长指标 + 健康评分。"""
    params = _parse(input_val)
    root = _wiki_root(params)

    counts = {}
    total_links = 0
    total_size = 0
    for category in ("sources", "entities", "topics", "analyses"):
        cat_dir = os.path.join(root, category)
        if os.path.isdir(cat_dir):
            files = [f for f in os.listdir(cat_dir) if f.endswith(".md")]
            counts[category] = len(files)
            for fname in files:
                fpath = os.path.join(cat_dir, fname)
                try:
                    size = os.path.getsize(fpath)
                    total_size += size
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    total_links += len(re.findall(r'\[\[', content))
                except Exception:
                    pass
        else:
            counts[category] = 0

    total = sum(counts.values())

    # 生长指标
    links_per_page = total_links / max(total, 1)
    entities_per_source = counts["entities"] / max(counts["sources"], 1)
    topics_per_source = counts["topics"] / max(counts["sources"], 1)

    # 健康评分 (0-100)
    health = 100
    if total == 0:
        health = 0
    else:
        if links_per_page < 1.0:
            health -= 20  # 交叉引用不足
        if counts["topics"] == 0:
            health -= 15  # 没有综合主题
        if entities_per_source < 1.0:
            health -= 10  # 实体提取不足

    # 读 log.md 获取最近操作
    log_path = os.path.join(root, "log.md")
    recent_ops = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("## ["):
                    recent_ops.append(line.strip())
    except Exception:
        pass

    report = (
        f"📊 Wiki 状态 (v2)\n\n"
        f"  wiki 目录:   {root}\n"
        f"  源文件页:    {counts['sources']}\n"
        f"  实体页:      {counts['entities']}\n"
        f"  主题页:      {counts['topics']}\n"
        f"  分析页:      {counts['analyses']}\n"
        f"  总页数:      {total}\n"
        f"  总大小:      {total_size / 1024:.0f} KB\n\n"
        f"  📈 生长指标\n"
        f"  交叉引用:    {total_links} 个 ({links_per_page:.1f}/页)\n"
        f"  实体/源文件: {entities_per_source:.1f}\n"
        f"  主题/源文件: {topics_per_source:.1f}\n"
        f"  健康评分:    {health}/100\n"
    )

    if recent_ops:
        report += f"\n  📜 最近操作 (最新 5 条)\n"
        for op in recent_ops[-5:]:
            report += f"  {op}\n"

    return report


# ════════════════════════════════════════════════════════════
# WikiGrow — 主动生长 (v2 新增)
# ════════════════════════════════════════════════════════════

def wiki_grow(input_val: Any) -> str:
    """
    v2: 主动生长 — 发现知识空白 → 自动创建综合页面。

    实现图3中"Markdown Wiki 知识库 (持续生长)"。

    功能:
      1. 发现高频引用但无独立页面的概念 → 创建实体页
      2. 发现多源文件共享的主题但无综合页 → 创建主题页
      3. 为孤立实体建立关联 → 补充交叉引用

    输入:
      {"wiki_root": "/path/to/wiki"}
      {"wiki_root": "/path", "dry_run": true}  → 只报告不创建
    """
    params = _parse(input_val)
    root = _wiki_root(params)
    dry_run = params.get("dry_run", False)

    # 收集所有页面和内容
    all_pages = {}
    all_links = set()
    entity_mentions = {}  # name → count of pages mentioning it
    topic_counts = {}     # topic → list of source files

    for category in ("sources", "entities", "topics"):
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(cat_dir, fname)
            rel_path = f"{category}/{fname}"
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                all_pages[rel_path] = content
                # 提取 [[链接]]
                links = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)
                all_links.update(links)
            except Exception:
                pass

    # 发现被引用但无页面的概念
    existing_slugs = set()
    for page_path in all_pages:
        existing_slugs.add(page_path.split("/")[-1].replace(".md", ""))

    missing_concepts = []
    for link in all_links:
        link_slug = _slug(link)
        if link_slug and link_slug not in existing_slugs:
            # 统计被引用次数
            ref_count = sum(1 for content in all_pages.values() if f"[[{link}]]" in content)
            if ref_count >= 2:  # 至少被 2 个页面引用
                missing_concepts.append((link, ref_count))

    missing_concepts.sort(key=lambda x: -x[1])

    # 发现需要综合的主题 (多个源文件共享的实体)
    entity_dir = os.path.join(root, "entities")
    entities_needing_synthesis = []
    if os.path.isdir(entity_dir):
        for fname in os.listdir(entity_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(entity_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # 统计来源数
                sources = re.findall(r'来源引用.*?(?=##|\Z)', content, re.DOTALL)
                source_count = content.count("- ") if sources else 0
                if source_count >= 3:  # 被 3+ 源引用的实体值得综合
                    name = fname.replace(".md", "").replace("-", " ")
                    entities_needing_synthesis.append((name, source_count))
            except Exception:
                pass

    entities_needing_synthesis.sort(key=lambda x: -x[1])

    # 执行生长
    created = []

    if not dry_run:
        # 创建缺失概念的占位页
        for concept, ref_count in missing_concepts[:10]:
            slug = _slug(concept)
            page_path = os.path.join(root, "entities", f"{slug}.md")
            if not os.path.exists(page_path):
                content = (
                    f"---\nentity: {concept}\ntype: concept\n"
                    f"updated: {_now_str()}\nstatus: auto-grown\nreferences: {ref_count}\n---\n\n"
                    f"# {concept}\n\n"
                    f"_此页面由 WikiGrow 自动创建 (被 {ref_count} 个页面引用)。_\n\n"
                    f"## 待补充\n- 定义和描述\n- 与其他概念的关系\n"
                )
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(content)
                created.append(f"entities/{slug}.md (引用 {ref_count} 次)")

        _rebuild_index(root)

    # 写日志
    _append_log(root, (
        f"\n## [{_now_str()}] grow | "
        f"发现 {len(missing_concepts)} 个缺失概念, "
        f"{len(entities_needing_synthesis)} 个可综合实体"
        f"{f', 创建 {len(created)} 页' if created else ''}\n"
    ))

    # 报告
    report = f"🌱 Wiki 生长分析\n\n"

    report += f"## 缺失概念 (被引用但无页面, {len(missing_concepts)} 个)\n"
    for concept, cnt in missing_concepts[:15]:
        status = "✅ 已创建" if not dry_run and concept in [c.split("/")[-1].split(".")[0] for c, _ in missing_concepts[:10]] else "⬜"
        report += f"  {status} [[{concept}]] — 被 {cnt} 个页面引用\n"

    report += f"\n## 可综合实体 (多源引用, {len(entities_needing_synthesis)} 个)\n"
    for name, cnt in entities_needing_synthesis[:10]:
        report += f"  📝 {name} — {cnt} 个来源引用\n"

    if created:
        report += f"\n## 本次创建 ({len(created)} 个新页面)\n"
        for c in created:
            report += f"  🌱 {c}\n"

    if dry_run:
        report += "\n💡 使用 dry_run=false 执行实际创建。\n"

    return report


# ════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════

def _collect_files(path: str) -> List[str]:
    """收集文件列表。"""
    path = os.path.expanduser(path.strip())
    if "*" in path or "?" in path:
        return [f for f in glob.glob(path, recursive=True)
                if os.path.isfile(f) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    if os.path.isfile(path):
        return [path] if os.path.splitext(path)[1].lower() in SUPPORTED_EXTS else []
    if os.path.isdir(path):
        files = []
        for root_dir, dirs, filenames in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', '.git', 'venv')]
            for name in sorted(filenames):
                full = os.path.join(root_dir, name)
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                    files.append(full)
        return files
    return []


def _read_pdf(filepath: str) -> str:
    """读取 PDF。"""
    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n\n".join(
                f"[Page {i+1}]\n{page.extract_text() or ''}"
                for i, page in enumerate(reader.pages)
            )
    except ImportError:
        return "[ERROR] 需要安装 PyPDF2: pip install PyPDF2"
    except Exception as e:
        return f"[ERROR] PDF 读取失败: {e}"


def _parse_json_from_llm(text: str) -> Dict:
    """从 LLM 输出中提取 JSON。"""
    text = text.strip()
    # 去除 markdown 代码块
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"summary": text[:200], "entities": [], "topics": [], "key_facts": [], "connections": []}
