"""
lambdagent.types — Paper III 类型与效果系统

实现论文 III 的类型系统:
  - AgentType: Agent 函数类型 τ1 →^ε τ2
  - LamType: 基础类型构造 (Str, Int, Bool, Float, Any, Json(S))
  - Json(S): 复用 JSON Schema 作为结构化类型语言 (Definition 2)
  - 子类型关系 <: (Definition 5): 宽度/深度子类型
  - T-Compose 规则: f >> g 要求 output(f) <: input(g) (Paper III §3.3.3)

核心方程:
    is_subtype(τ1, τ2) = True  ⟺  τ1 <: τ2

依赖图:
    types.py  ←  effects.py (效果标注)
              ←  compiler.py (编译时类型检查)
              ←  core.py (Term.input_type / output_type)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union


# ============================================================
# 基础类型 (Paper III Definition 1)
# ============================================================

class TypeTag(Enum):
    """类型标签枚举"""
    ANY = auto()      # ⊤ — 顶类型，所有类型的超类型
    NONE = auto()     # ⊥ — 底类型，所有类型的子类型
    STR = auto()      # 字符串
    INT = auto()      # 整数
    FLOAT = auto()    # 浮点数
    BOOL = auto()     # 布尔
    JSON = auto()     # Json(S) — JSON Schema 结构化类型
    TUPLE = auto()    # 元组类型 (Pair 的输出)
    UNION = auto()    # 联合类型 (Route/If 的输出)


@dataclass(frozen=True)
class LamType:
    """
    Lambda Agent 的类型。

    Paper III Definition 1:
        τ ::= Str | Int | Bool | Float | Any | Json(S) | τ1 × τ2 | τ1 | τ2

    其中 S 是 JSON Schema（Definition 2），
    复用 JSON Schema 作为结构化类型语言。
    """
    tag: TypeTag
    # Json(S): JSON Schema dict (when tag == JSON)
    schema: Optional[Dict[str, Any]] = field(default=None, hash=False)
    # Tuple: element types (when tag == TUPLE)
    elements: Optional[Tuple[LamType, ...]] = None
    # Union: member types (when tag == UNION)
    members: Optional[FrozenSet[LamType]] = None

    def __repr__(self) -> str:
        if self.tag == TypeTag.ANY:
            return "Any"
        elif self.tag == TypeTag.NONE:
            return "None"
        elif self.tag == TypeTag.STR:
            return "Str"
        elif self.tag == TypeTag.INT:
            return "Int"
        elif self.tag == TypeTag.FLOAT:
            return "Float"
        elif self.tag == TypeTag.BOOL:
            return "Bool"
        elif self.tag == TypeTag.JSON:
            if self.schema:
                t = self.schema.get("type", "object")
                if t == "object":
                    props = self.schema.get("properties", {})
                    if props:
                        fields = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in list(props.items())[:3])
                        if len(props) > 3:
                            fields += ", ..."
                        return f"Json({{{fields}}})"
                elif t == "array":
                    items = self.schema.get("items", {})
                    return f"Json([{items.get('type', '?')}])"
                return f"Json({t})"
            return "Json"
        elif self.tag == TypeTag.TUPLE:
            if self.elements:
                inner = ", ".join(str(e) for e in self.elements)
                return f"({inner})"
            return "()"
        elif self.tag == TypeTag.UNION:
            if self.members:
                inner = " | ".join(str(m) for m in sorted(self.members, key=str))
                return f"({inner})"
            return "Never"
        return f"LamType({self.tag})"


# ============================================================
# 类型常量（快捷方式）
# ============================================================

T_ANY = LamType(TypeTag.ANY)
T_NONE = LamType(TypeTag.NONE)
T_STR = LamType(TypeTag.STR)
T_INT = LamType(TypeTag.INT)
T_FLOAT = LamType(TypeTag.FLOAT)
T_BOOL = LamType(TypeTag.BOOL)


def T_JSON(schema: Dict[str, Any] | None = None) -> LamType:
    """构造 Json(S) 类型"""
    return LamType(TypeTag.JSON, schema=schema)


def T_TUPLE(*elements: LamType) -> LamType:
    """构造元组类型"""
    return LamType(TypeTag.TUPLE, elements=tuple(elements))


def T_UNION(*members: LamType) -> LamType:
    """构造联合类型"""
    return LamType(TypeTag.UNION, members=frozenset(members))


# ============================================================
# AgentType: Agent 函数类型 (Paper III Definition 3)
# ============================================================

@dataclass(frozen=True)
class AgentType:
    """
    Agent 函数类型: τ1 →^ε τ2

    Paper III Definition 3:
        每个 Agent 的类型签名是 input_type →^effect output_type

    effect 在 effects.py 中定义，此处暂用字符串占位。
    """
    input_type: LamType
    output_type: LamType
    effect: str = "pure"  # 暂用字符串；P0-2 后替换为 Effect 类型

    def __repr__(self) -> str:
        eff = f"^{self.effect}" if self.effect != "pure" else ""
        return f"{self.input_type} →{eff} {self.output_type}"


# ============================================================
# 子类型关系 <: (Paper III Definition 5)
# ============================================================

def is_subtype(sub: LamType, sup: LamType) -> bool:
    """
    子类型判断: sub <: sup

    Paper III Definition 5:
        1. ⊥ <: τ (None 是所有类型的子类型)
        2. τ <: ⊤ (所有类型是 Any 的子类型)
        3. τ <: τ (自反性)
        4. Str <: Json(string) (字符串嵌入 JSON)
        5. Int <: Float (数值提升)
        6. Bool <: Int (布尔嵌入整数)
        7. Json(S1) <: Json(S2) when S1 structurally subtypes S2
           (宽度子类型: S1 有更多字段 → S1 <: S2)
           (深度子类型: 对应字段类型 S1.f <: S2.f)
        8. (τ1, τ2) <: (σ1, σ2) when τ1 <: σ1 ∧ τ2 <: σ2 (元组协变)
        9. τ <: (τ | σ) (联合类型引入)
    """
    # ⊥ <: τ
    if sub.tag == TypeTag.NONE:
        return True

    # τ <: ⊤
    if sup.tag == TypeTag.ANY:
        return True

    # 自反性
    if sub == sup:
        return True

    # τ <: (τ | σ) — 联合类型：sub 是 sup 的某个 member 的子类型
    if sup.tag == TypeTag.UNION and sup.members:
        return any(is_subtype(sub, m) for m in sup.members)

    # (τ1 | τ2) <: σ — 联合类型的子类型：所有 member 都是 σ 的子类型
    if sub.tag == TypeTag.UNION and sub.members:
        return all(is_subtype(m, sup) for m in sub.members)

    # Bool <: Int <: Float
    if sub.tag == TypeTag.BOOL and sup.tag == TypeTag.INT:
        return True
    if sub.tag == TypeTag.BOOL and sup.tag == TypeTag.FLOAT:
        return True
    if sub.tag == TypeTag.INT and sup.tag == TypeTag.FLOAT:
        return True

    # Str <: Json(string)
    if sub.tag == TypeTag.STR and sup.tag == TypeTag.JSON:
        if sup.schema and sup.schema.get("type") == "string":
            return True
        # Str <: Json (untyped JSON) — 字符串可以被解析为 JSON
        if sup.schema is None:
            return True

    # 基础类型 <: Json(对应类型)
    _tag_to_json_type = {
        TypeTag.STR: "string",
        TypeTag.INT: "integer",
        TypeTag.FLOAT: "number",
        TypeTag.BOOL: "boolean",
    }
    if sub.tag in _tag_to_json_type and sup.tag == TypeTag.JSON:
        if sup.schema and sup.schema.get("type") == _tag_to_json_type[sub.tag]:
            return True
        # FIX-04: 基础类型 <: Json (无 schema = untyped JSON, 接受一切)
        # Paper III S-StrJson / S-NumJson / S-BoolJson
        if sup.schema is None:
            return True

    # Json(S1) <: Json(S2) — 结构子类型
    if sub.tag == TypeTag.JSON and sup.tag == TypeTag.JSON:
        return _json_schema_subtype(sub.schema, sup.schema)

    # 元组协变: (τ1, τ2) <: (σ1, σ2)
    if sub.tag == TypeTag.TUPLE and sup.tag == TypeTag.TUPLE:
        if sub.elements and sup.elements:
            if len(sub.elements) != len(sup.elements):
                return False
            return all(
                is_subtype(s, t) for s, t in zip(sub.elements, sup.elements)
            )

    return False


def _json_schema_subtype(
    sub_schema: Optional[Dict[str, Any]],
    sup_schema: Optional[Dict[str, Any]],
) -> bool:
    """
    JSON Schema 结构子类型检查。

    Paper III Definition 5 规则 7:
        Json(S1) <: Json(S2) 当且仅当:
          - S2 的所有 required 字段在 S1 中都存在
          - 对应字段类型满足 S1.field <: S2.field (深度子类型)
          - S1 可以有额外字段 (宽度子类型)

    类似 TypeScript 的结构子类型。
    """
    # 无 schema → Any JSON → Json <: Json
    if sup_schema is None:
        return True
    if sub_schema is None:
        # 未指定的 JSON 不是有具体 schema 的子类型
        return sup_schema is None

    sub_type = sub_schema.get("type")
    sup_type = sup_schema.get("type")

    # 类型不同 → 检查 JSON 原始类型的子类型关系
    if sub_type != sup_type:
        # integer <: number
        if sub_type == "integer" and sup_type == "number":
            return True
        return False

    # object 子类型: 宽度 + 深度
    if sup_type == "object":
        sub_props = sub_schema.get("properties", {})
        sup_props = sup_schema.get("properties", {})
        sup_required = set(sup_schema.get("required", []))

        # sup 的所有 required 字段必须在 sub 中存在
        for req_field in sup_required:
            if req_field not in sub_props:
                return False

        # 深度子类型: 公共字段类型兼容
        for field_name, sup_field_schema in sup_props.items():
            if field_name in sub_props:
                if not _json_schema_subtype(sub_props[field_name], sup_field_schema):
                    return False
            elif field_name in sup_required:
                return False
            # sup 有字段但 sub 没有 + 非 required → OK (宽度子类型的逆方向，
            # 这里 sub 少字段不影响，因为 sup 不要求该字段)

        return True

    # array 子类型: items 协变
    if sup_type == "array":
        sub_items = sub_schema.get("items", {})
        sup_items = sup_schema.get("items", {})
        if sub_items and sup_items:
            return _json_schema_subtype(sub_items, sup_items)
        return True

    # 基础类型相同 → 子类型
    return True


# ============================================================
# 类型检查错误
# ============================================================

class AgentTypeError(Exception):
    """Agent 类型检查错误 — T-Compose 规则违反"""

    def __init__(self, message: str, source_type: Optional[LamType] = None,
                 target_type: Optional[LamType] = None, position: int = -1):
        self.source_type = source_type
        self.target_type = target_type
        self.position = position
        detail = ""
        if source_type and target_type:
            detail = f"\n  Output type: {source_type}\n  Input type:  {target_type}"
            if position >= 0:
                detail += f"\n  At composition boundary: step {position} >> step {position + 1}"
        detail += "\n  Rule: Paper III T-Compose: f: A →^ε1 B, g: B' →^ε2 C requires B <: B'"
        super().__init__(f"{message}{detail}")


# ============================================================
# 类型检查：T-Compose 规则 (Paper III §3.3.3)
# ============================================================

def check_compose_types(agent_types: List[AgentType]) -> AgentType:
    """
    T-Compose 类型检查。

    Paper III §3.3.3:
        f: A →^ε1 B,  g: B' →^ε2 C,  B <: B'
        ─────────────────────────────────────────
              f >> g : A →^(ε1 · ε2) C

    检查链式组合中每对相邻 agent 的类型兼容性。
    返回整个组合的类型。

    Raises:
        AgentTypeError: 类型不兼容时抛出
    """
    if not agent_types:
        return AgentType(T_ANY, T_ANY)

    if len(agent_types) == 1:
        return agent_types[0]

    for i in range(len(agent_types) - 1):
        f_type = agent_types[i]
        g_type = agent_types[i + 1]

        # T-Compose: output(f) <: input(g)
        if not is_subtype(f_type.output_type, g_type.input_type):
            raise AgentTypeError(
                f"Type mismatch in pipeline at step {i} >> step {i + 1}: "
                f"{f_type.output_type} is not a subtype of {g_type.input_type}",
                source_type=f_type.output_type,
                target_type=g_type.input_type,
                position=i,
            )

    # 组合结果类型: input(first) → output(last)
    combined_effect = " · ".join(at.effect for at in agent_types if at.effect != "pure")
    return AgentType(
        input_type=agent_types[0].input_type,
        output_type=agent_types[-1].output_type,
        effect=combined_effect or "pure",
    )


# ============================================================
# 类型推断辅助
# ============================================================

def parse_type_annotation(annotation: Any) -> LamType:
    """
    从 YAML 配置中的类型标注解析为 LamType。

    支持的格式:
        "Str"                    → T_STR
        "Int"                    → T_INT
        "Float"                  → T_FLOAT
        "Bool"                   → T_BOOL
        "Any"                    → T_ANY
        "Json"                   → T_JSON()
        {"type": "object", ...}  → T_JSON(schema)
        {"type": "string"}       → T_JSON({"type": "string"})
    """
    if annotation is None:
        return T_ANY

    if isinstance(annotation, str):
        _name_map = {
            "str": T_STR, "string": T_STR,
            "int": T_INT, "integer": T_INT,
            "float": T_FLOAT, "number": T_FLOAT,
            "bool": T_BOOL, "boolean": T_BOOL,
            "any": T_ANY,
            "json": T_JSON(),
        }
        lower = annotation.lower().strip()
        if lower in _name_map:
            return _name_map[lower]
        # 可能是 JSON Schema 字符串
        try:
            parsed = json.loads(annotation)
            if isinstance(parsed, dict):
                return T_JSON(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        return T_ANY

    if isinstance(annotation, dict):
        # JSON Schema dict
        return T_JSON(annotation)

    return T_ANY


def infer_type_from_value(value: Any) -> LamType:
    """从运行时值推断类型（用于调试/trace）"""
    if isinstance(value, str):
        return T_STR
    elif isinstance(value, bool):
        return T_BOOL
    elif isinstance(value, int):
        return T_INT
    elif isinstance(value, float):
        return T_FLOAT
    elif isinstance(value, dict):
        return T_JSON()
    elif isinstance(value, (tuple, list)):
        if isinstance(value, tuple):
            return T_TUPLE(*(infer_type_from_value(v) for v in value))
        return T_JSON({"type": "array"})
    return T_ANY
