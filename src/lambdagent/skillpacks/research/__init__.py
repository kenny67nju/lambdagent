"""
research skill pack — 科研流程技能包

封装 "读论文→复现→实验→记录" 四阶段研究流程为可复用 Skill。

4 个独立 Skill:
  paper-reader    读论文: 搜索 → 阅读 → 提取关键信息
  code-reproducer 复现: 理解方法 → 写代码 → 对齐论文结果
  experimenter    实验: 设计实验 → 执行 → 分析结果
  lab-notebook    记录: 整理笔记 → 结构化报告 → 保存到知识库

1 个组合 Skill:
  research-pipeline  完整流程: paper-reader >> code-reproducer >> experimenter >> lab-notebook

用法:
  # 方式 1: 注册后按需调用单个 skill
  from lambdagent.skillpacks.research import register_all
  register_all()

  registry = SkillRegistry()
  reader = registry.get("paper-reader")
  result = reader.apply("Attention Is All You Need")

  # 方式 2: 运行完整 pipeline
  pipeline = registry.get("research-pipeline")
  result = pipeline.apply("论文标题或URL")

  # 方式 3: 在 agent67v2 中通过协调者调用
  # 协调者会自动发现 research-pipeline skill
"""

from .skills import register_all, SKILL_PACK_NAME

__all__ = ["register_all", "SKILL_PACK_NAME"]
