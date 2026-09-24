from pathlib import Path

# Folders that only hold artifacts (data, environments, outputs) and never help the code reader
SKIPPED_DIRS = {
    "data",
    "venv",
    ".venv",
    "runs",
    "results",
    "outputs",
    "figures",
    "__pycache__",
    "archive",
    "node_modules",
}


def build_project_map(codebase_path: str) -> str:
    """
    Builds a compact two-level map of the codebase (folders, plus top-level code and docs files),
    so the code reader can target its tool calls instead of spending them on exploration.
    """
    lines = []
    for top in sorted(Path(codebase_path).iterdir()):
        if not top.is_dir() or top.name.startswith(".") or top.name in SKIPPED_DIRS:
            continue
        children = [
            child.name + ("/" if child.is_dir() else "")
            for child in sorted(top.iterdir())
            if not child.name.startswith((".", "_"))
            and not child.name.startswith("runs")
            and child.name not in SKIPPED_DIRS
            and (child.is_dir() or child.suffix in (".py", ".md"))
        ]
        lines.append(f"{top.name}/: {', '.join(children)}")
    return "\n".join(lines)
