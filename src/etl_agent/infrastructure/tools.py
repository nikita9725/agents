"""Read-only LangChain tools with bounded, validated JSON observations."""

import ast
import math
import operator
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import Field

from etl_agent.domain import Model, NonEmptyText

MAX_BYTES = 65536


class CalculatorInput(Model):
    expression: NonEmptyText = Field(
        max_length=512, description="Arithmetic using + - * / and parentheses"
    )


class CalculatorOutput(Model):
    result: float


class FileInput(Model):
    path: NonEmptyText = Field(description="Relative UTF-8 file path within TOOL_FILES_ROOT")


class FileOutput(Model):
    path: str
    content: str
    truncated: bool


class SearchInput(FileInput):
    query: NonEmptyText = Field(max_length=512, description="Case-sensitive literal text")


class Match(Model):
    line: int
    text: str


class SearchOutput(Model):
    path: str
    matches: list[Match]
    truncated: bool


class ToolError(Model):
    error: str


def expected_error(exc: ToolException) -> str:
    return ToolError(error=str(exc)).model_dump_json()


def read_file(root: Path, path: str) -> FileOutput:
    try:
        relative = Path(path)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise ToolException("Path must remain inside TOOL_FILES_ROOT")
        if not resolved.is_file():
            raise ToolException("File not found or not a regular file")
        with resolved.open("rb") as source:
            data = source.read(MAX_BYTES + 1)
        truncated = len(data) > MAX_BYTES
        # Decode incrementally so a truncated multibyte character isn't a false encoding error.
        import codecs

        content = codecs.getincrementaldecoder("utf-8")().decode(
            data[:MAX_BYTES], final=not truncated
        )
        return FileOutput(path=path, content=content, truncated=truncated)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ToolException("File cannot be read as UTF-8") from exc


def create_tools(root: Path) -> list[BaseTool]:
    root = root.resolve()

    @tool(args_schema=CalculatorInput)
    def calculator(expression: str) -> str:
        """Calculate an arithmetic expression; never execute Python code."""
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 100:
                raise ToolException("Expression is too complex")
            operations: dict[type[ast.operator], Callable[[float, float], float]] = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }

            def evaluate(node: ast.AST) -> float:
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    assert isinstance(node.value, (int, float))
                    return float(node.value)
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    value = evaluate(node.operand)
                    return -value if isinstance(node.op, ast.USub) else value
                if isinstance(node, ast.BinOp) and type(node.op) in operations:
                    return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
                raise ToolException("Only numbers, parentheses and + - * / are supported")

            result = evaluate(tree.body)
            if not math.isfinite(result):
                raise ToolException("Result must be finite")
            return CalculatorOutput(result=result).model_dump_json()
        except (SyntaxError, ArithmeticError, ValueError, RecursionError) as exc:
            raise ToolException("Invalid arithmetic expression or division by zero") from exc

    @tool(args_schema=FileInput)
    def read_local_file(path: str) -> str:
        """Read a UTF-8 ETL file inside TOOL_FILES_ROOT, up to 64 KiB."""
        return read_file(root, path).model_dump_json()

    @tool(args_schema=SearchInput)
    def text_search(path: str, query: str) -> str:
        """Find literal case-sensitive text in one ETL file, returning up to 50 lines."""
        file = read_file(root, path)
        matches = [
            Match(line=n, text=line)
            for n, line in enumerate(file.content.splitlines(), 1)
            if query in line
        ]
        return SearchOutput(
            path=path, matches=matches[:50], truncated=file.truncated or len(matches) > 50
        ).model_dump_json()

    tools: list[BaseTool] = [calculator, read_local_file, text_search]
    for item in tools:
        item.handle_tool_error = expected_error
        item.handle_validation_error = ToolError(error="Invalid tool arguments").model_dump_json()
    return tools
