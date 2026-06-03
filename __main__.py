"""
python -m lambdagent <command> [args]

Lambda 演算 Agent DSL 的命令行入口。

等价于: lambdagent <command> [args]

Lambda 语义:
    CLI 本身是一个 Lambda 项:
    main = λargs. dispatch(parse(args))
    每个子命令是一个分支:
    dispatch = CASE command [
        ("compile", compile_handler),
        ("run",     run_handler),
        ("repl",    repl_handler),
        ("lint",    lint_handler),
        ...
    ]
"""

from lambdagent.cli.main import main

if __name__ == "__main__":
    main()
