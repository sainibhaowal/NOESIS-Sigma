# Verifier/dsl_ast.py
# NOESIS-S -- Verifier DSL AST (D4)

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from Core.Verifier.errors import E_DSL_001, VerifierError

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
)
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.Not, ast.USub, ast.UAdd)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


@dataclass(frozen=True)
class CompiledExpr:
    src: str
    code: Any
    node_count: int


def _validate(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"Unsupported node: {type(n).__name__}")
        if isinstance(n, ast.BinOp) and not isinstance(n.op, _ALLOWED_BINOPS):
            raise ValueError(f"Unsupported binop: {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp) and not isinstance(n.op, _ALLOWED_UNARYOPS):
            raise ValueError(f"Unsupported unaryop: {type(n.op).__name__}")
        if isinstance(n, ast.BoolOp) and not isinstance(n.op, _ALLOWED_BOOLOPS):
            raise ValueError(f"Unsupported boolop: {type(n.op).__name__}")
        if isinstance(n, ast.Compare):
            for op in n.ops:
                if not isinstance(op, _ALLOWED_CMPOPS):
                    raise ValueError(f"Unsupported cmpop: {type(op).__name__}")
        if isinstance(n, ast.Call) and not isinstance(n.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")


def compile_expr(expr: str) -> tuple[CompiledExpr | None, VerifierError | None]:
    try:
        node = ast.parse(expr, mode="eval")
        _validate(node)
        code = compile(node, "<verifier_dsl>", "eval")
        node_count = sum(1 for _ in ast.walk(node))
        return CompiledExpr(src=expr, code=code, node_count=node_count), None
    except Exception as e:
        return None, VerifierError(
            code=E_DSL_001, message="DSL syntax/AST not allowed", detail=str(e)
        )
