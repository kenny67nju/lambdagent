"""
lambdagent.primitives — Lambda 演算的 6 个核心原语

Lam      λ 抽象       创建 Agent（数据集 → 函数）
Compose  函数组合     f >> g = λx. g(f(x))
If       Church 条件  IF cond THEN a ELSE b
Loop     Y 组合子     递归/迭代直到不动点
Pair     Church 对    PAIR = λa.λb.λf. f a b
Fst/Snd  投影         FST = λp. p TRUE / SND = λp. p FALSE
Tool     原语/Oracle  将 Python 函数提升为 Lambda 项
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union

from .core import Term, Context, TraceEntry


# ============================================================
# Lam: λ 抽象 — 核心构造，创建 Agent
# ============================================================

class Lam(Term):
    """
    Lambda 抽象: λx.body

    通过 prompt/数据集定义 Agent 行为。
    调用 = β-规约 = 自回归解码。

        Lam("summarizer", "Summarize concisely.")
        ≡ λ_D . F_{M,D}
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        output_parser: Callable[[str], Any] | None = None,
    ):
        super().__init__(name)
        self.prompt = prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_parser = output_parser or (lambda x: x)
        self._client = None

    def _get_client(self):
        if self._client is None:
            provider = self._detect_provider()
            if provider == 'anthropic':
                import anthropic
                self._client = ('anthropic', anthropic.Anthropic())
            elif provider == 'openai':
                import openai
                self._client = ('openai', openai.OpenAI())
            elif provider == 'ollama':
                self._client = ('ollama', None)
            else:
                self._client = ('dashscope', None)
        return self._client

    def _detect_provider(self) -> str:
        m = self.model.lower()
        # Explicit provider prefix takes priority
        if m.startswith('dashscope/'):
            return 'dashscope'
        if m.startswith('anthropic/'):
            return 'anthropic'
        if m.startswith('openai/'):
            return 'openai'
        if m.startswith('ollama/'):
            return 'ollama'
        # Auto-detect by model name
        if 'claude' in m or 'anthropic' in m:
            return 'anthropic'
        elif 'gpt' in m or 'openai' in m:
            return 'openai'
        elif 'qwen' in m or 'glm' in m or 'llama' in m or 'ollama' in m:
            # 检测 Ollama 是否在运行
            try:
                import urllib.request
                with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
                    return 'ollama'
            except Exception:
                pass
        return 'dashscope'

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """β-规约: (λ_D x) → F_{M,D}(x)

        Paper III §6: 如果有活跃的效果处理器，委托给 handler.handle_llm()。
        """
        ctx = ctx or Context()
        t0 = time.time()

        # S17: Enforce token budget hard limit before LLM call
        token_budget = ctx.bindings.get("__token_budget__") if hasattr(ctx, 'bindings') else None
        if token_budget is not None:
            estimated = token_budget.estimate_cost(str(input)) if hasattr(token_budget, 'estimate_cost') else 0
            token_budget.enforce_before_call(estimated)

        # Paper III: 检查效果处理器
        # DESIGN-01: Use PassthroughHandler base class for isinstance check
        from .handlers import get_current_handler, PassthroughHandler
        handler = get_current_handler()
        if handler is not None and not isinstance(handler, PassthroughHandler):
            # 使用 handler 处理 LLM 效果（如 TestHandler 的 Mock）
            raw = handler.handle_llm(
                self.prompt, str(input), self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        else:
            raw = self._call_llm(str(input))

        duration = (time.time() - t0) * 1000
        result = self.output_parser(raw)
        ctx.log(self._name, self._trace_id, input, result, duration, self.model)

        # Report cost to TraceHandler if active
        from .handlers import TraceHandler
        if handler is not None and isinstance(handler, TraceHandler):
            handler.handle_cost(0, duration, self.model)

        return result

    def _call_llm(self, input_text: str) -> str:
        """自回归解码 = 一步 β-规约. 支持 Anthropic/OpenAI/DashScope."""
        provider, client = self._get_client()

        if provider == 'anthropic':
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.prompt,
                messages=[{"role": "user", "content": input_text}],
            )
            return response.content[0].text.strip()

        elif provider == 'openai':
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": input_text},
                ],
            )
            return response.choices[0].message.content.strip()

        elif provider == 'ollama':
            # Ollama local model via HTTP
            import json, urllib.request
            url = "http://localhost:11434/api/chat"
            body = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": input_text},
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("message", {}).get("content", "").strip()

        else:
            # DashScope via HTTP
            import os, json, urllib.request
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY not set")
            model_name = self.model.replace("dashscope/", "")
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            body = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": input_text},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


# ============================================================
# Compose: 函数组合 f >> g = λx. g(f(x))
# ============================================================

class Compose(Term):
    """
    函数组合: λx. g(f(x))

        f >> g >> h  =  Compose(f, g, h)
        pipeline(x)  =  h(g(f(x)))

    每个 >> 步骤 = 一次 β-规约，结果传递给下一步。
    """

    def __init__(self, *stages: Term):
        name = " >> ".join(s._name for s in stages)
        super().__init__(name)
        self.stages = list(stages)
        # Paper III: Compose 的类型 = input(first) → output(last)
        if stages:
            self._input_type = stages[0]._input_type
            self._output_type = stages[-1]._output_type

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        result = input
        for stage in self.stages:
            result = stage.apply(result, ctx)
        return result

    def __rshift__(self, other: Term) -> Compose:
        """展平嵌套组合: (f >> g) >> h = Compose(f, g, h)"""
        if isinstance(other, Compose):
            return Compose(*self.stages, *other.stages)
        return Compose(*self.stages, other)


# ============================================================
# If: Church 布尔条件分支
# ============================================================

class If(Term):
    """
    Church 条件: IF cond THEN then_ ELSE else_

    Lambda: IF ≡ λc.λt.λe. c t e
    cond 返回 truthy → 执行 then_
    cond 返回 falsy → 执行 else_
    """

    def __init__(
        self,
        cond: Term | Callable[[Any], bool],
        then_: Term,
        else_: Term,
    ):
        cond_name = getattr(cond, '_name', 'cond')
        super().__init__(f"If({cond_name})")
        self.cond = cond
        self.then_ = then_
        self.else_ = else_

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        if isinstance(self.cond, Term):
            condition_result = self.cond.apply(input, ctx)
            branch = self._is_truthy(condition_result)
        else:
            branch = self.cond(input)

        if branch:
            return self.then_.apply(input, ctx)
        else:
            return self.else_.apply(input, ctx)

    @staticmethod
    def _is_truthy(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().upper() in ("TRUE", "YES", "1")
        return bool(val)


# ============================================================
# Loop: Y 组合子 — 递归/迭代
# ============================================================

class Loop(Term):
    """
    Y 组合子: 递归自应用直到不动点。

    Lambda: Y = λf. (λx. f(x x)) (λx. f(x x))
    DSL:    Loop(body, stop_condition)

    每次迭代 = 一步 CoT = 一次 β-规约展开。
    输出反馈为输入，直到 condition 返回 True 或达到 max_steps。
    """

    def __init__(
        self,
        body: Term,
        condition: Callable[[Any, int], bool],
        max_steps: int = 10,
    ):
        super().__init__(f"Loop({body._name})")
        self.body = body
        self.condition = condition
        self.max_steps = max_steps

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        ctx = ctx or Context()
        result = input
        for step in range(self.max_steps):
            result = self.body.apply(result, ctx)
            if self.condition(result, step):
                break
        return result


# ============================================================
# Pair / Fst / Snd: Church 有序对
# ============================================================

class Pair(Term):
    """
    Church 对: PAIR = λa.λb.λf. f a b

    对同一输入运行两个 Agent，打包结果为元组。
    """

    def __init__(self, first: Term, second: Term):
        super().__init__(f"Pair({first._name}, {second._name})")
        self.first = first
        self.second = second
        # Paper III: Pair 输出类型 = (output(first), output(second))
        from .lam_types import T_TUPLE
        self._output_type = T_TUPLE(first.output_type, second.output_type)

    def apply(self, input: Any, ctx: Context | None = None) -> tuple:
        ctx = ctx or Context()
        a = self.first.apply(input, ctx)
        b = self.second.apply(input, ctx)
        return (a, b)


class Fst(Term):
    """FST = λp. p TRUE — 取第一个元素"""
    def __init__(self):
        super().__init__("Fst")

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        # FIX-03: 类型检查，防止非 tuple 输入导致 IndexError
        if not isinstance(input, (tuple, list)) or len(input) < 1:
            raise TypeError(
                f"Fst expects a tuple/list with at least 1 element, "
                f"got {type(input).__name__}: {str(input)[:100]}"
            )
        return input[0]


class Snd(Term):
    """SND = λp. p FALSE — 取第二个元素"""
    def __init__(self):
        super().__init__("Snd")

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        # FIX-03: 类型检查，防止非 tuple 输入导致 IndexError
        if not isinstance(input, (tuple, list)) or len(input) < 2:
            raise TypeError(
                f"Snd expects a tuple/list with at least 2 elements, "
                f"got {type(input).__name__}: {str(input)[:100]}"
            )
        return input[1]


# ============================================================
# Tool: 外部函数 → Lambda 项
# ============================================================

class Tool(Term):
    """
    将 Python 函数提升为 Lambda 项（原语/Oracle）。

        Tool("double", lambda x: int(x) * 2)
        ≡ λx. double(x)

    这是外部工具（搜索、代码执行、数据库）进入 Lambda 世界的入口。
    """

    def __init__(self, name: str, fn: Callable):
        super().__init__(name)
        self.fn = fn

    def apply(self, input: Any, ctx: Context | None = None) -> Any:
        """Paper III §6: 如果有活跃的效果处理器，委托给 handler.handle_tool()。"""
        ctx = ctx or Context()
        t0 = time.time()

        # Paper III: 检查效果处理器
        from .handlers import get_current_handler, ProductionHandler
        handler = get_current_handler()
        if handler is not None and not isinstance(handler, ProductionHandler):
            result = handler.handle_tool(self._name, self.fn, input)
        else:
            result = self.fn(input)

        duration = (time.time() - t0) * 1000
        ctx.log(self._name, self._trace_id, input, result, duration)
        return result
