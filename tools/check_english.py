"""Check source comments/docstrings and Markdown for untranslated CJK prose."""

import ast
import io
from pathlib import Path
import re
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u9fff]")
SKIP = {".git", ".venv", "node_modules", "__pycache__", "dist", ".pytest_cache"}


def main():
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in {".py", ".md", ".mo", ".css", ".ts", ".tsx"}:
            continue
        source = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".md":
            texts = [source]
        elif path.suffix == ".py":
            texts = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            texts += [
                ast.get_docstring(node) or ""
                for node in ast.walk(ast.parse(source))
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
            ]
        else:
            tokens = re.findall(
                r'"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|//[^\n]*|/\*[\s\S]*?\*/', source
            )
            texts = [token for token in tokens if token.startswith(("//", "/*"))]
        if any(CJK.search(text) for text in texts):
            failures.append(str(path.relative_to(ROOT)))
    if failures:
        print("Non-English source prose detected:\n" + "\n".join(failures))
        return 1
    print("English documentation check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
