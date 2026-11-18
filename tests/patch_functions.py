"""在 CPU 上直接测试固定补丁中新增的函数。"""
import ast
from pathlib import Path
import textwrap

PATCH = Path(__file__).resolve().parents[1] / 'patches/sglang-hicache-experiment.patch'


def added_function(name, scope=None):
    """直接执行交付补丁里完整新增的函数，不复制一份实现用于测试。"""
    lines = PATCH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith('+') and line[1:].lstrip().startswith(f'def {name}('))
    added = []
    for line in lines[start:]:
        if not line.startswith('+'):
            break
        added.append(line[1:])
    tree = ast.parse(textwrap.dedent('\n'.join(added)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(scope or {})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATCH), 'exec'), namespace)
    return namespace[name]
