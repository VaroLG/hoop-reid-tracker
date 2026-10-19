"""The court chapter and the tracker must stay independent (see hooptrack/court/__init__.py)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "hooptrack"
TRACKER_PARTS = ("tracking", "reid", "detection", "eval", "pipeline", "cli", "viz", "data")


def _imports(path: Path) -> set[str]:
    """Absolute module names imported by a file (relative imports resolved inside hooptrack)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg = path.relative_to(SRC.parent).with_suffix("").parts[:-1]      # ('hooptrack', 'court')
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                                 # relative: from ..tracking import x
                base = pkg[: len(pkg) - node.level + 1]
                out.add(".".join((*base, node.module)) if node.module else ".".join(base))
            elif node.module:
                out.add(node.module)
    return out


def test_court_does_not_import_tracker():
    for f in (SRC / "court").rglob("*.py"):
        bad = {m for m in _imports(f) if any(m.startswith(f"hooptrack.{p}") for p in TRACKER_PARTS)}
        assert not bad, f"{f.name} imports tracker modules: {bad}"


def test_tracker_does_not_import_court():
    for f in SRC.rglob("*.py"):
        rel = f.relative_to(SRC).parts
        if rel[0] in ("court", "integrations"):
            continue
        bad = {m for m in _imports(f) if m.startswith("hooptrack.court")}
        assert not bad, f"{'/'.join(rel)} imports the court chapter directly: {bad}"
