"""
Tests for Paper III Type System — T-Compose + Json(S) subtyping.

Tests cover:
  1. Basic subtype relations (Definition 5)
  2. Json(S) structural subtyping (width + depth)
  3. T-Compose type checking (§3.3.3)
  4. Type annotation parsing (YAML integration)
  5. Compose >> operator type checking
"""

import pytest
from lambdagent.lam_types import (
    LamType, TypeTag, AgentType, AgentTypeError,
    T_ANY, T_NONE, T_STR, T_INT, T_FLOAT, T_BOOL, T_JSON, T_TUPLE, T_UNION,
    is_subtype, check_compose_types, parse_type_annotation, infer_type_from_value,
)
from lambdagent.core import Term, Context
from lambdagent.primitives import Lam, Compose, Tool, Pair


# ============================================================
# 1. Basic Subtype Relations (Paper III Definition 5)
# ============================================================

class TestBasicSubtyping:
    """Test basic subtype rules from Definition 5."""

    def test_reflexivity(self):
        """τ <: τ"""
        assert is_subtype(T_STR, T_STR)
        assert is_subtype(T_INT, T_INT)
        assert is_subtype(T_FLOAT, T_FLOAT)
        assert is_subtype(T_BOOL, T_BOOL)

    def test_top_type(self):
        """τ <: ⊤ (Any is top)"""
        assert is_subtype(T_STR, T_ANY)
        assert is_subtype(T_INT, T_ANY)
        assert is_subtype(T_BOOL, T_ANY)
        assert is_subtype(T_JSON(), T_ANY)

    def test_bottom_type(self):
        """⊥ <: τ (None is bottom)"""
        assert is_subtype(T_NONE, T_STR)
        assert is_subtype(T_NONE, T_INT)
        assert is_subtype(T_NONE, T_ANY)
        assert is_subtype(T_NONE, T_JSON())

    def test_numeric_hierarchy(self):
        """Bool <: Int <: Float"""
        assert is_subtype(T_BOOL, T_INT)
        assert is_subtype(T_BOOL, T_FLOAT)
        assert is_subtype(T_INT, T_FLOAT)
        # Not the reverse
        assert not is_subtype(T_FLOAT, T_INT)
        assert not is_subtype(T_INT, T_BOOL)

    def test_str_json_string(self):
        """Str <: Json(string)"""
        json_str = T_JSON({"type": "string"})
        assert is_subtype(T_STR, json_str)
        # Str <: Json (untyped)
        assert is_subtype(T_STR, T_JSON())

    def test_primitive_to_json(self):
        """Int <: Json(integer), Bool <: Json(boolean)"""
        assert is_subtype(T_INT, T_JSON({"type": "integer"}))
        assert is_subtype(T_BOOL, T_JSON({"type": "boolean"}))
        assert is_subtype(T_FLOAT, T_JSON({"type": "number"}))

    def test_incompatible_types(self):
        """Str ≮: Int, Int ≮: Str"""
        assert not is_subtype(T_STR, T_INT)
        assert not is_subtype(T_INT, T_STR)
        assert not is_subtype(T_STR, T_BOOL)


# ============================================================
# 2. Json(S) Structural Subtyping
# ============================================================

class TestJsonSubtyping:
    """Test Json Schema structural subtyping (width + depth)."""

    def test_json_any(self):
        """Json(S) <: Json (untyped)"""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert is_subtype(T_JSON(schema), T_JSON())

    def test_width_subtyping(self):
        """More fields <: fewer fields (width subtyping)"""
        sub = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},  # extra field
            },
            "required": ["name", "age"],
        })
        sup = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        })
        assert is_subtype(sub, sup)

    def test_missing_required_field(self):
        """Missing required field → not subtype"""
        sub = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        })
        sup = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        })
        assert not is_subtype(sub, sup)

    def test_depth_subtyping(self):
        """Nested field types must be compatible (depth subtyping)"""
        sub = T_JSON({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        })
        sup = T_JSON({
            "type": "object",
            "properties": {
                "count": {"type": "number"},  # integer <: number
            },
        })
        assert is_subtype(sub, sup)

    def test_depth_subtype_failure(self):
        """Incompatible nested types → not subtype"""
        sub = T_JSON({
            "type": "object",
            "properties": {
                "count": {"type": "string"},
            },
        })
        sup = T_JSON({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        })
        assert not is_subtype(sub, sup)

    def test_array_subtyping(self):
        """Array items covariance"""
        sub = T_JSON({
            "type": "array",
            "items": {"type": "integer"},
        })
        sup = T_JSON({
            "type": "array",
            "items": {"type": "number"},
        })
        assert is_subtype(sub, sup)

    def test_type_mismatch(self):
        """Different JSON types → not subtype (except integer/number)"""
        assert not is_subtype(
            T_JSON({"type": "string"}),
            T_JSON({"type": "integer"}),
        )
        assert not is_subtype(
            T_JSON({"type": "object"}),
            T_JSON({"type": "array"}),
        )


# ============================================================
# 3. Tuple and Union Types
# ============================================================

class TestCompoundTypes:

    def test_tuple_covariance(self):
        """(Int, Str) <: (Float, Str) when Int <: Float"""
        assert is_subtype(
            T_TUPLE(T_INT, T_STR),
            T_TUPLE(T_FLOAT, T_STR),
        )

    def test_tuple_length_mismatch(self):
        """Different lengths → not subtype"""
        assert not is_subtype(
            T_TUPLE(T_INT),
            T_TUPLE(T_INT, T_STR),
        )

    def test_union_introduction(self):
        """τ <: (τ | σ)"""
        union = T_UNION(T_STR, T_INT)
        assert is_subtype(T_STR, union)
        assert is_subtype(T_INT, union)
        assert not is_subtype(T_FLOAT, union)

    def test_union_subtype(self):
        """(τ1 | τ2) <: σ when all members <: σ"""
        union = T_UNION(T_INT, T_BOOL)
        assert is_subtype(union, T_FLOAT)  # both Int, Bool <: Float


# ============================================================
# 4. T-Compose Type Checking (Paper III §3.3.3)
# ============================================================

class TestTCompose:

    def test_compatible_chain(self):
        """Str → Json(S) >> Json(S) → Str should pass"""
        json_type = T_JSON({"type": "object", "properties": {"result": {"type": "string"}}})
        types = [
            AgentType(T_STR, json_type),
            AgentType(json_type, T_STR),
        ]
        result = check_compose_types(types)
        assert result.input_type == T_STR
        assert result.output_type == T_STR

    def test_incompatible_chain(self):
        """Str → Int >> Str → Str should fail (Int ≮: Str)"""
        types = [
            AgentType(T_STR, T_INT),
            AgentType(T_STR, T_STR),
        ]
        with pytest.raises(AgentTypeError) as exc_info:
            check_compose_types(types)
        assert "not a subtype" in str(exc_info.value)

    def test_any_type_passes(self):
        """Any → Any always compatible"""
        types = [
            AgentType(T_ANY, T_ANY),
            AgentType(T_ANY, T_ANY),
        ]
        result = check_compose_types(types)
        assert result.input_type == T_ANY

    def test_subtype_compatible(self):
        """Int output <: Float input should pass"""
        types = [
            AgentType(T_STR, T_INT),
            AgentType(T_FLOAT, T_STR),
        ]
        result = check_compose_types(types)
        assert result.input_type == T_STR
        assert result.output_type == T_STR

    def test_json_width_subtype_in_chain(self):
        """Agent outputting {name, age, email} >> Agent expecting {name, age}"""
        full = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age"],
        })
        partial = T_JSON({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        })
        types = [
            AgentType(T_STR, full),
            AgentType(partial, T_STR),
        ]
        result = check_compose_types(types)
        assert result.input_type == T_STR
        assert result.output_type == T_STR

    def test_three_step_chain(self):
        """A → B >> B → C >> C → D"""
        types = [
            AgentType(T_STR, T_INT),
            AgentType(T_INT, T_FLOAT),
            AgentType(T_FLOAT, T_STR),
        ]
        result = check_compose_types(types)
        assert result.input_type == T_STR
        assert result.output_type == T_STR

    def test_three_step_chain_fail_middle(self):
        """A → B >> C → D fails when B ≮: C"""
        types = [
            AgentType(T_STR, T_INT),
            AgentType(T_STR, T_FLOAT),  # Int ≮: Str
            AgentType(T_FLOAT, T_STR),
        ]
        with pytest.raises(AgentTypeError) as exc_info:
            check_compose_types(types)
        assert "step 0 >> step 1" in str(exc_info.value)


# ============================================================
# 5. Type Annotation Parsing
# ============================================================

class TestTypeAnnotationParsing:

    def test_parse_str(self):
        assert parse_type_annotation("Str") == T_STR
        assert parse_type_annotation("string") == T_STR

    def test_parse_int(self):
        assert parse_type_annotation("Int") == T_INT
        assert parse_type_annotation("integer") == T_INT

    def test_parse_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = parse_type_annotation(schema)
        assert result.tag == TypeTag.JSON
        assert result.schema == schema

    def test_parse_any(self):
        assert parse_type_annotation("Any") == T_ANY
        assert parse_type_annotation(None) == T_ANY

    def test_parse_json_string(self):
        result = parse_type_annotation('{"type": "string"}')
        assert result.tag == TypeTag.JSON


# ============================================================
# 6. Operator Type Checking (>> at construction time)
# ============================================================

class TestOperatorTypeChecking:

    def test_rshift_with_types_pass(self):
        """f >> g passes when output(f) <: input(g)"""
        f = Tool("to_int", lambda x: int(x))
        f.output_type = T_INT
        g = Tool("to_str", lambda x: str(x))
        g.input_type = T_FLOAT  # Int <: Float
        result = f >> g
        assert isinstance(result, Compose)

    def test_rshift_with_types_fail(self):
        """f >> g fails when output(f) ≮: input(g)"""
        f = Tool("to_int", lambda x: int(x))
        f.output_type = T_INT
        g = Tool("to_str", lambda x: str(x))
        g.input_type = T_STR  # Int ≮: Str
        with pytest.raises(AgentTypeError):
            f >> g

    def test_rshift_any_type_no_check(self):
        """f >> g with Any types should not trigger type check"""
        f = Tool("a", lambda x: x)
        g = Tool("b", lambda x: x)
        result = f >> g  # Both have T_ANY, no error
        assert isinstance(result, Compose)

    def test_pair_output_type(self):
        """Pair(f, g) has output type (output(f), output(g))"""
        f = Tool("a", lambda x: x)
        f.output_type = T_STR
        g = Tool("b", lambda x: x)
        g.output_type = T_INT
        p = Pair(f, g)
        assert p.output_type == T_TUPLE(T_STR, T_INT)


# ============================================================
# 7. Type Inference from Values
# ============================================================

class TestTypeInference:

    def test_infer_str(self):
        assert infer_type_from_value("hello") == T_STR

    def test_infer_int(self):
        assert infer_type_from_value(42) == T_INT

    def test_infer_float(self):
        assert infer_type_from_value(3.14) == T_FLOAT

    def test_infer_bool(self):
        assert infer_type_from_value(True) == T_BOOL

    def test_infer_dict(self):
        result = infer_type_from_value({"key": "val"})
        assert result.tag == TypeTag.JSON

    def test_infer_tuple(self):
        result = infer_type_from_value(("a", 1))
        assert result.tag == TypeTag.TUPLE


# ============================================================
# 8. Type Repr
# ============================================================

class TestTypeRepr:

    def test_basic_repr(self):
        assert repr(T_STR) == "Str"
        assert repr(T_INT) == "Int"
        assert repr(T_ANY) == "Any"

    def test_json_repr(self):
        t = T_JSON({"type": "object", "properties": {"name": {"type": "string"}}})
        assert "Json" in repr(t)

    def test_tuple_repr(self):
        t = T_TUPLE(T_STR, T_INT)
        assert "(Str, Int)" == repr(t)

    def test_agent_type_repr(self):
        at = AgentType(T_STR, T_INT, "llm(claude)")
        assert "Str →^llm(claude) Int" == repr(at)
