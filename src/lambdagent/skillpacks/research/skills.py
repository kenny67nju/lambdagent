"""
research.skills — 科研流程四阶段 Skill 定义

Lambda 语义:
  research-pipeline = paper-reader >> code-reproducer >> experimenter >> lab-notebook
  = λpaper. notebook(experiment(reproduce(read(paper))))

每个 Skill 都是独立的、可单独调用的 Lambda Term:
  paper-reader:     λquery. {title, abstract, method, key_findings, code_url}
  code-reproducer:  λpaper_info. {repo_path, reproduced_results, match_score}
  experimenter:     λreproduction. {experiment_design, results, analysis}
  lab-notebook:     λexperiment. {notebook_path, summary, knowledge_base_entry}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# 确保 lambdagent 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lambdagent.core import Term, Context
from lambdagent.primitives import Lam, Compose, Tool
from lambdagent.skills import Skill, SkillSignature, SkillPack, SkillRegistry


SKILL_PACK_NAME = "research"


# ════════════════════════════════════════════════════════════
# Skill 1: paper-reader (读论文)
# ════════════════════════════════════════════════════════════

PAPER_READER_PROMPT = """\
你是 paper-reader，一个论文阅读专家。

## 任务
给定论文标题、URL 或关键词，执行以下步骤：
1. **搜索**: 用 WebSearch 找到论文 (arXiv, Semantic Scholar, Google Scholar)
2. **获取**: 用 WebFetch 获取论文页面，提取关键信息
3. **分析**: 提取以下结构化信息:
   - title: 论文标题
   - authors: 作者列表
   - venue: 发表会议/期刊
   - year: 年份
   - abstract: 摘要
   - problem: 解决什么问题
   - method: 核心方法/算法
   - key_findings: 主要发现/结果
   - datasets: 使用的数据集
   - baselines: 对比的基线方法
   - code_url: 代码仓库 (如果有)
   - limitations: 局限性

## 工具
```json
{"action": "WebSearch", "input": {"query": "论文关键词"}}
{"action": "WebFetch", "input": {"url": "论文URL"}}
{"action": "terminate", "input": {"summary": "JSON格式的论文分析结果"}}
```

## 输出格式
最终用 terminate 输出 JSON 格式的论文分析结果。
"""

PAPER_READER_CONFIG = {
    "type": "react",
    "name": "paper-reader",
    "model": {
        "provider": "claude-code",
        "name": "sonnet",
        "temperature": 0.2,
        "maxTokens": 4096,
    },
    "systemPrompt": PAPER_READER_PROMPT,
    "react": {
        "maxSteps": 10,
        "observationEnabled": True,
        "toolTimeout": 45,
        "thinkTimeout": 60,
    },
    "mcp": {
        "localTools": ["WebSearch", "WebFetch", "terminate"],
        "policy": {"mode": "auto"},
    },
    "guard": {"maxOutputLength": 5000, "retry": 1, "fallback": "last"},
}


# ════════════════════════════════════════════════════════════
# Skill 2: code-reproducer (复现)
# ════════════════════════════════════════════════════════════

CODE_REPRODUCER_PROMPT = """\
你是 code-reproducer，一个代码复现专家。

## 任务
给定论文分析结果 (来自 paper-reader)，执行复现：
1. **获取代码**: 如果 code_url 存在，克隆仓库；否则根据 method 描述从零实现
2. **理解代码**: 阅读关键文件，理解模型架构和训练流程
3. **环境搭建**: 检查依赖，创建运行环境
4. **运行复现**: 执行训练/推理脚本
5. **对比结果**: 将复现结果与论文报告的结果对比

## 工具
```json
{"action": "Bash", "input": {"command": "git clone ..."}}
{"action": "ReadFile", "input": {"path": "..."}}
{"action": "WriteFile", "input": {"path": "...", "content": "..."}}
{"action": "ListFiles", "input": {"path": "..."}}
{"action": "terminate", "input": {"summary": "复现结果JSON"}}
```

## 输出格式
用 terminate 输出 JSON:
{
  "repo_path": "本地仓库路径",
  "key_files": ["模型文件", "训练脚本", ...],
  "reproduced_results": {"metric1": value1, ...},
  "paper_results": {"metric1": value1, ...},
  "match_score": 0.85,
  "notes": "复现笔记"
}
"""

CODE_REPRODUCER_CONFIG = {
    "type": "react",
    "name": "code-reproducer",
    "model": {
        "provider": "claude-code",
        "name": "sonnet",
        "temperature": 0.2,
        "maxTokens": 4096,
    },
    "systemPrompt": CODE_REPRODUCER_PROMPT,
    "react": {
        "maxSteps": 20,
        "observationEnabled": True,
        "toolTimeout": 120,
        "thinkTimeout": 60,
    },
    "mcp": {
        "localTools": [
            "Bash",
            "ReadFile",
            "WriteFile",
            "EditFile",
            "ListFiles",
            "SearchContent",
            "CodeSearch",
            "RunTests",
            "terminate",
        ],
        "policy": {"mode": "auto"},
    },
    "guard": {
        "dangerousCommandBlock": True,
        "maxOutputLength": 5000,
        "retry": 1,
        "fallback": "last",
    },
}


# ════════════════════════════════════════════════════════════
# Skill 3: experimenter (实验)
# ════════════════════════════════════════════════════════════

EXPERIMENTER_PROMPT = """\
你是 experimenter，一个实验设计与执行专家。

## 任务
给定复现结果 (来自 code-reproducer)，设计并执行扩展实验：
1. **设计实验**: 基于论文方法，设计消融实验 / 对比实验 / 参数敏感性分析
2. **编写脚本**: 写实验运行脚本 (确保可复现)
3. **执行实验**: 运行实验，收集结果
4. **分析结果**: 统计指标、对比分析、可视化

## 工具
```json
{"action": "Bash", "input": {"command": "python run_experiment.py"}}
{"action": "WriteFile", "input": {"path": "experiments/exp1.py", "content": "..."}}
{"action": "ReadFile", "input": {"path": "results/exp1.json"}}
{"action": "terminate", "input": {"summary": "实验结果JSON"}}
```

## 输出格式
用 terminate 输出 JSON:
{
  "experiment_design": "实验设计描述",
  "experiments": [
    {"name": "exp1", "description": "...", "results": {...}},
    ...
  ],
  "analysis": "分析总结",
  "key_insights": ["发现1", "发现2"],
  "figures": ["path/to/fig1.png", ...]
}
"""

EXPERIMENTER_CONFIG = {
    "type": "react",
    "name": "experimenter",
    "model": {
        "provider": "claude-code",
        "name": "sonnet",
        "temperature": 0.3,
        "maxTokens": 4096,
    },
    "systemPrompt": EXPERIMENTER_PROMPT,
    "react": {
        "maxSteps": 25,
        "observationEnabled": True,
        "toolTimeout": 180,
        "thinkTimeout": 60,
    },
    "mcp": {
        "localTools": [
            "Bash",
            "ReadFile",
            "WriteFile",
            "EditFile",
            "ListFiles",
            "SearchContent",
            "RunTests",
            "terminate",
        ],
        "policy": {"mode": "auto"},
    },
    "guard": {
        "dangerousCommandBlock": True,
        "maxOutputLength": 8000,
        "retry": 1,
        "fallback": "last",
    },
}


# ════════════════════════════════════════════════════════════
# Skill 4: lab-notebook (记录)
# ════════════════════════════════════════════════════════════

LAB_NOTEBOOK_PROMPT = """\
你是 lab-notebook，一个科研记录专家。

## 任务
给定实验结果 (来自 experimenter)，整理结构化研究笔记：
1. **整理笔记**: 将整个研究过程 (论文→复现→实验) 整理为 Markdown 笔记
2. **生成报告**: 包含背景、方法、复现、实验、分析、结论
3. **保存到知识库**: 将笔记添加到知识库，便于未来检索
4. **记忆关键信息**: 保存关键发现到记忆系统

## 笔记格式
```markdown
# [论文标题] 研究笔记

## 基本信息
- 作者: ...
- 会议: ...
- 代码: ...

## 问题与动机
[论文解决什么问题]

## 核心方法
[方法描述]

## 复现结果
| 指标 | 论文 | 复现 | 差异 |
|------|------|------|------|
| ... | ... | ... | ... |

## 扩展实验
[实验设计和结果]

## 关键发现
- ...

## 个人思考
[对这篇工作的理解和未来方向]

## 标签
#paper #[领域] #[方法]
```

## 工具
```json
{"action": "WriteFile", "input": {"path": "notes/paper_xxx.md", "content": "..."}}
{"action": "KBAdd", "input": {"kb_name": "research", "content": "...", "metadata": {...}}}
{"action": "MemoryStore", "input": {"key": "paper:xxx", "value": "...", "tags": [...]}}
{"action": "terminate", "input": {"summary": "笔记路径和摘要"}}
```
"""

LAB_NOTEBOOK_CONFIG = {
    "type": "react",
    "name": "lab-notebook",
    "model": {
        "provider": "claude-code",
        "name": "sonnet",
        "temperature": 0.3,
        "maxTokens": 4096,
    },
    "systemPrompt": LAB_NOTEBOOK_PROMPT,
    "react": {
        "maxSteps": 10,
        "observationEnabled": True,
        "toolTimeout": 30,
        "thinkTimeout": 60,
    },
    "mcp": {
        "localTools": [
            "WriteFile",
            "ReadFile",
            "DocGen",
            "KBCreate",
            "KBAdd",
            "KBSearch",
            "MemoryStore",
            "MemoryRecall",
            "terminate",
        ],
        "policy": {"mode": "auto"},
    },
    "guard": {"maxOutputLength": 8000, "retry": 1, "fallback": "last"},
}


# ════════════════════════════════════════════════════════════
# Skill 构建与注册
# ════════════════════════════════════════════════════════════


class _InlineConfigTerm(Term):
    """从内联 config dict 懒编译的 Term"""

    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self._config = config
        self._compiled = None

    def apply(self, input_val, ctx=None):
        if self._compiled is None:
            from lambdagent.fromconfig.compiler import build_agent

            self._compiled = build_agent(self._config, {})
        ctx = ctx or Context()
        return self._compiled.apply(str(input_val), ctx)


def _build_skill(name: str, config: dict, description: str, tags: list) -> Skill:
    """从 config dict 构建 Skill"""
    term = _InlineConfigTerm(name, config)
    return Skill(
        name=name,
        term=term,
        description=description,
        signature=SkillSignature(input_type="Str", output_type="Str"),
        tags=tags,
        version="1.0.0",
        author="research-skills",
    )


def _build_pipeline_skill(skills: dict) -> Skill:
    """
    构建完整 pipeline: reader >> reproducer >> experimenter >> notebook

    Lambda: λpaper. notebook(experimenter(reproducer(reader(paper))))
    """
    reader = skills["paper-reader"]
    reproducer = skills["code-reproducer"]
    experimenter = skills["experimenter"]
    notebook = skills["lab-notebook"]

    # Compose: reader >> reproducer >> experimenter >> notebook
    pipeline = reader >> reproducer >> experimenter >> notebook

    return pipeline


def build_pack() -> SkillPack:
    """构建 research skill pack (不注册)"""
    pack = SkillPack(
        name=SKILL_PACK_NAME,
        description="科研流程技能包: 读论文→复现→实验→记录",
        version="1.0.0",
        author="lambdagent",
    )

    # 4 个独立 Skill
    skills = {}

    skills["paper-reader"] = _build_skill(
        "paper-reader",
        PAPER_READER_CONFIG,
        "论文阅读: 搜索→阅读→提取关键信息",
        ["paper", "read", "search", "论文", "阅读", "文献"],
    )
    pack.add(skills["paper-reader"])

    skills["code-reproducer"] = _build_skill(
        "code-reproducer",
        CODE_REPRODUCER_CONFIG,
        "代码复现: 克隆仓库→理解代码→运行复现→对比结果",
        ["code", "reproduce", "复现", "代码", "实现"],
    )
    pack.add(skills["code-reproducer"])

    skills["experimenter"] = _build_skill(
        "experimenter",
        EXPERIMENTER_CONFIG,
        "实验设计与执行: 消融实验→对比实验→结果分析",
        ["experiment", "run", "analysis", "实验", "分析"],
    )
    pack.add(skills["experimenter"])

    skills["lab-notebook"] = _build_skill(
        "lab-notebook",
        LAB_NOTEBOOK_CONFIG,
        "研究记录: 整理笔记→结构化报告→保存知识库",
        ["notebook", "record", "write", "笔记", "记录", "知识库"],
    )
    pack.add(skills["lab-notebook"])

    # 1 个组合 pipeline Skill
    pipeline = _build_pipeline_skill(skills)
    pack.add(pipeline)

    return pack


def register_all() -> SkillPack:
    """构建并注册 research skill pack 到全局 SkillRegistry"""
    pack = build_pack()
    registry = SkillRegistry()
    registry.register_pack(pack)
    return pack


def get_skill(name: str) -> Optional[Skill]:
    """获取单个研究技能"""
    registry = SkillRegistry()
    skill = registry.get(name)
    if skill is None:
        register_all()
        skill = registry.get(name)
    return skill
