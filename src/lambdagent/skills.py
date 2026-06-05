"""
lambdagent.skills — Skill 系统

Skill = 命名的、可复用的、可发现的、可组合的 Lambda 项。

Lambda 演算语义:
    Skill     = let name = λx.body in ...    (命名的 Lambda 项 + 元数据)
    Registry  = Γ_skills : Name → Skill      (技能注册表 = 特殊环境)
    Discovery = Route(LLM, Γ_skills)         (LLM 从注册表中选择技能)
    Compose   = skill_a >> skill_b           (技能组合 = 函数组合)
    Learn     = λexamples. Dataset(examples).to_lam()  (从经验构造新技能)
    Curry     = skill.bind(param=value)      (偏应用 = 柯里化)

核心概念:
    Skill       一个有名字、描述、类型签名的 Lambda 项
    SkillPack   一组相关技能的集合（类似 Python package）
    Registry    全局技能注册表（可发现、可搜索）
    SkillAgent  能自动发现和使用技能的 Agent
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .core import Term, Context, LambdagentError


# ════════════════════════════════════════════════════════════
# Skill: 命名的 Lambda 项
# ════════════════════════════════════════════════════════════


@dataclass
class SkillSignature:
    """
    技能的类型签名。

    Lambda 类型: τ_in → τ_out
    用于组合检查: skill_a >> skill_b 要求 a.τ_out ⊆ b.τ_in
    """

    input_type: str = "Str"  # 输入类型描述
    output_type: str = "Str"  # 输出类型描述
    input_schema: Optional[dict] = None  # JSON Schema (可选)
    output_schema: Optional[dict] = None  # JSON Schema (可选)

    def compatible_with(self, other: SkillSignature) -> bool:
        """检查 self >> other 是否类型兼容"""
        # 简单检查: 输出类型 = 输入类型
        if self.output_type == other.input_type:
            return True
        # Str 与一切兼容（LLM 的万能胶水）
        if self.output_type == "Str" or other.input_type == "Str":
            return True
        return False


class Skill(Term):
    """
    Skill = 命名的、可复用的、可发现的 Lambda 项。

    Lambda 语义:
        Skill(name, term) = let name = term in ...

    与普通 Term 的区别:
        1. 有 description（自然语言描述，用于 LLM 发现）
        2. 有 signature（类型签名，用于组合检查）
        3. 有 tags（标签，用于搜索过滤）
        4. 有 examples（使用示例，用于 few-shot 学习）
        5. 可序列化（存盘/加载/分享）
    """

    def __init__(
        self,
        name: str,
        term: Term,
        description: str = "",
        signature: Optional[SkillSignature] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        version: str = "1.0.0",
        author: str = "",
    ):
        super().__init__(name)
        self.term = term
        self.description = description
        self.signature = signature or SkillSignature()
        self.tags = tags or []
        self.examples = examples or []
        self.version = version
        self.author = author
        self.skill_id = f"skill_{uuid.uuid4().hex[:8]}"
        self._call_count = 0
        self._total_time_ms = 0.0

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """执行技能 = β-规约"""
        ctx = ctx or Context()
        t0 = time.time()
        result = self.term.apply(input, ctx)
        elapsed = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_time_ms += elapsed
        ctx.log(f"Skill:{self._name}", self._trace_id, input, result, elapsed)
        return result

    def bind(self, **kwargs) -> Skill:
        """
        偏应用（柯里化）: 固定部分参数，返回新技能。

        Lambda 语义:
            skill.bind(lang="zh") = λx. skill(x + "[lang=zh]")
        """
        prefix = "\n".join(f"[{k}={v}]" for k, v in kwargs.items())
        from .primitives import Tool

        binder = Tool(f"bind({','.join(kwargs.keys())})", lambda x: f"{prefix}\n{x}")
        new_term = binder >> self.term if hasattr(binder, "__rshift__") else self.term
        # Use Compose explicitly
        from .primitives import Compose

        new_term = Compose(binder, self.term)
        return Skill(
            name=f"{self._name}[{','.join(f'{k}={v}' for k, v in kwargs.items())}]",
            term=new_term,
            description=f"{self.description} (with {kwargs})",
            signature=self.signature,
            tags=self.tags,
            examples=self.examples,
            version=self.version,
            author=self.author,
        )

    def __rshift__(self, other: Union[Term, Skill]) -> Skill:
        """
        技能组合: skill_a >> skill_b

        Lambda 语义: λx. other(self(x))
        类型检查: self.τ_out 兼容 other.τ_in
        """
        from .primitives import Compose

        other_sig = other.signature if isinstance(other, Skill) else SkillSignature()
        if isinstance(other, Skill) and not self.signature.compatible_with(other_sig):
            import warnings

            warnings.warn(
                f"Skill type mismatch: {self._name}:{self.signature.output_type} "
                f">> {other._name}:{other_sig.input_type}"
            )
        composed_term = Compose(
            self.term, other.term if isinstance(other, Skill) else other
        )
        return Skill(
            name=f"{self._name}>>{other._name}",
            term=composed_term,
            description=f"{self.description} then {other.description if isinstance(other, Skill) else ''}",
            signature=SkillSignature(
                input_type=self.signature.input_type,
                output_type=other_sig.output_type,
            ),
            tags=list(
                set(self.tags + (other.tags if isinstance(other, Skill) else []))
            ),
        )

    @property
    def stats(self) -> dict:
        """使用统计"""
        return {
            "calls": self._call_count,
            "total_ms": self._total_time_ms,
            "avg_ms": self._total_time_ms / max(1, self._call_count),
        }

    def to_dict(self) -> dict:
        """序列化为字典（用于存盘/分享）"""
        return {
            "name": self._name,
            "skill_id": self.skill_id,
            "description": self.description,
            "signature": {
                "input_type": self.signature.input_type,
                "output_type": self.signature.output_type,
            },
            "tags": self.tags,
            "examples": self.examples,
            "version": self.version,
            "author": self.author,
            "stats": self.stats,
        }

    def __repr__(self):
        return (
            f"Skill({self._name!r}, "
            f"{self.signature.input_type}→{self.signature.output_type}, "
            f"tags={self.tags})"
        )


# ════════════════════════════════════════════════════════════
# SkillPack: 技能集合
# ════════════════════════════════════════════════════════════


class SkillPack:
    """
    一组相关技能的集合（类似 Python package / npm 包）。

    Lambda 语义:
        SkillPack = {name₁: skill₁, name₂: skill₂, ...}
        = 一组命名的 Lambda 项

    用途:
        - 按领域组织技能（"writing" pack, "coding" pack, "research" pack）
        - 版本化分发
        - 一键注册到 Registry
    """

    def __init__(
        self, name: str, description: str = "", version: str = "1.0.0", author: str = ""
    ):
        self.name = name
        self.description = description
        self.version = version
        self.author = author
        self.skills: Dict[str, Skill] = {}

    def add(self, skill: Skill) -> SkillPack:
        """添加技能"""
        self.skills[skill._name] = skill
        return self

    def get(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有技能名"""
        return list(self.skills.keys())

    def __len__(self):
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills.values())

    def __repr__(self):
        return f"SkillPack({self.name!r}, {len(self.skills)} skills)"


# ════════════════════════════════════════════════════════════
# Registry: 全局技能注册表
# ════════════════════════════════════════════════════════════


class SkillRegistry:
    """
    全局技能注册表: Γ_skills : Name → Skill

    Lambda 语义:
        Registry = 一个特殊的环境 Γ，存储命名的 Lambda 项
        register(skill) = Γ_skills[name ↦ skill]
        discover(query) = Route(LLM, Γ_skills)(query)
        search(tags)    = {s ∈ Γ_skills | tags ⊆ s.tags}

    特性:
        - 单例模式（全局唯一）
        - 支持按名称、标签、描述搜索
        - 支持 LLM 驱动的自动发现
        - 支持从 SkillPack 批量注册
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: Dict[str, Skill] = {}
            cls._instance._packs: Dict[str, SkillPack] = {}
        return cls._instance

    def register(self, skill: Skill) -> None:
        """注册技能"""
        self._skills[skill._name] = skill

    def register_pack(self, pack: SkillPack) -> None:
        """批量注册技能包"""
        self._packs[pack.name] = pack
        for skill in pack:
            self._skills[skill._name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """按名称获取"""
        return self._skills.get(name)

    def search(self, query: str = "", tags: Optional[List[str]] = None) -> List[Skill]:
        """
        搜索技能。

        Args:
            query: 文本搜索（匹配 name 或 description）
            tags: 标签过滤（AND 逻辑）
        """
        results = []
        for skill in self._skills.values():
            # 文本匹配（任意 query 词在 name/description/tags 中出现即可）
            if query:
                query_words = query.lower().split()
                text = (
                    f"{skill._name} {skill.description} {' '.join(skill.tags)}".lower()
                )
                if not any(w in text for w in query_words):
                    continue
            # 标签过滤
            if tags:
                if not all(t in skill.tags for t in tags):
                    continue
            results.append(skill)
        return results

    def discover(
        self, task: str, classifier: Optional[Term] = None, top_k: int = 3
    ) -> List[Skill]:
        """
        LLM 驱动的技能发现。

        Lambda 语义:
            discover(task) = Route(classifier, Γ_skills)(task)

        如果提供 classifier（LLM Agent），则用 LLM 从描述中选择最相关的技能。
        否则用简单的关键词匹配。
        """
        if classifier is not None:
            # 构造 skill catalog prompt
            catalog = "\n".join(
                f"- {s._name}: {s.description} (tags: {','.join(s.tags)})"
                for s in self._skills.values()
            )
            prompt_input = f"Task: {task}\n\nAvailable skills:\n{catalog}\n\nSelect the best skill name:"
            ctx = Context()
            result = str(classifier.apply(prompt_input, ctx)).strip()
            # 从结果中匹配技能
            matched = []
            for name, skill in self._skills.items():
                if name.lower() in result.lower():
                    matched.append(skill)
            return matched[:top_k] if matched else self.search(task)[:top_k]
        else:
            return self.search(task)[:top_k]

    def build_route(self) -> Term:
        """
        将注册表构建为一个 Route 构造。

        Lambda 语义:
            build_route() = Route(classifier, {name₁: skill₁, ...})
            = CASE (classifier x) [(name₁, skill₁), ...]

        返回的 Term 需要一个 classifier 来驱动。
        这里返回的是 routes dict，调用者负责提供 classifier。
        """
        from .extensions import Route

        routes = {name: skill for name, skill in self._skills.items()}
        return routes

    def stats(self) -> dict:
        """注册表统计"""
        return {
            "total_skills": len(self._skills),
            "total_packs": len(self._packs),
            "skills_by_tag": self._tag_distribution(),
            "most_used": sorted(
                [(s._name, s._call_count) for s in self._skills.values()],
                key=lambda x: -x[1],
            )[:10],
        }

    def _tag_distribution(self) -> Dict[str, int]:
        dist = {}
        for skill in self._skills.values():
            for tag in skill.tags:
                dist[tag] = dist.get(tag, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def clear(self):
        """清空注册表（测试用）"""
        self._skills.clear()
        self._packs.clear()

    def list_all(self) -> List[str]:
        """列出所有已注册技能"""
        return list(self._skills.keys())

    def __len__(self):
        return len(self._skills)

    def __repr__(self):
        return f"SkillRegistry({len(self._skills)} skills, {len(self._packs)} packs)"


# ════════════════════════════════════════════════════════════
# SkillAgent: 能自动发现和使用技能的 Agent
# ════════════════════════════════════════════════════════════


class SkillAgent(Term):
    """
    能自动发现和使用技能的 Agent。

    Lambda 语义:
        SkillAgent(classifier, registry) =
            λx. let skill = discover(classifier, registry, x) in
                 skill(x)

    这是 Handoff 的技能化版本:
        Handoff  = 动态选择 Agent
        SkillAgent = 动态选择 Skill（有描述、有类型、有统计的 Agent）

    工作流:
        1. 接收输入 x
        2. 用 classifier 从 registry 中发现最佳技能
        3. 执行该技能
        4. 返回结果
    """

    def __init__(
        self,
        classifier: Term,
        registry: Optional[SkillRegistry] = None,
        fallback: Optional[Skill] = None,
    ):
        super().__init__("SkillAgent")
        self.classifier = classifier
        self.registry = registry or SkillRegistry()
        self.fallback = fallback

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        t0 = time.time()

        # 用 classifier 选择技能名
        selected_name = str(self.classifier.apply(input, ctx)).strip()

        # 从 registry 查找（精确 → 模糊）
        found_skill = self.registry.get(selected_name)
        if found_skill is None:
            for name, s in self.registry._skills.items():
                if (
                    name.lower() in selected_name.lower()
                    or selected_name.lower() in name.lower()
                ):
                    found_skill = s
                    break

        if found_skill is None:
            # fallback: 用关键词搜索
            candidates = self.registry.search(selected_name)
            if candidates:
                found_skill = candidates[0]

        if found_skill is None:
            if self.fallback:
                found_skill = self.fallback
            else:
                raise LambdagentError(
                    f"SkillAgent: no skill '{selected_name}' found. "
                    f"Available: {self.registry.list_all()}"
                )

        # 执行技能
        result = found_skill.apply(input, ctx)
        elapsed = (time.time() - t0) * 1000
        ctx.log(
            f"SkillAgent→{found_skill._name}",
            self._trace_id,
            str(input)[:100],
            str(result)[:100],
            elapsed,
        )
        return result


# ════════════════════════════════════════════════════════════
# 便利函数: 快速创建 Skill
# ════════════════════════════════════════════════════════════


def skill(
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    input_type: str = "Str",
    output_type: str = "Str",
    examples: Optional[List[Tuple[str, str]]] = None,
    version: str = "1.0.0",
    author: str = "",
):
    """
    装饰器: 将 Term 或函数包装为 Skill。

    用法:
        @skill("summarize", "Summarize text", tags=["writing"])
        def summarize(x):
            return f"Summary: {x[:50]}..."

        # 或包装已有的 Term:
        my_skill = skill("translate", "Translate text")(my_lam_agent)
    """

    def decorator(fn_or_term):
        if isinstance(fn_or_term, Term):
            term = fn_or_term
        elif callable(fn_or_term):
            from .primitives import Tool

            term = Tool(name, fn_or_term)
        else:
            raise TypeError(f"Expected Term or callable, got {type(fn_or_term)}")

        s = Skill(
            name=name,
            term=term,
            description=description,
            signature=SkillSignature(input_type=input_type, output_type=output_type),
            tags=tags or [],
            examples=examples or [],
            version=version,
            author=author,
        )
        # 自动注册到全局 Registry
        SkillRegistry().register(s)
        return s

    return decorator
