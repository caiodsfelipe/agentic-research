import json
import os
from datetime import datetime
from pathlib import Path

from agentic_research.state import AppState

# Research history: an append-only JSON Lines file, one finished run per line.
# It is a record for people to browse (e.g. in the UI sidebar); it is never fed back into the agents,
# so a past mistake can't influence new runs.


def _history_path() -> Path:
    # Read at call time, so it can be redirected (e.g. by tests)
    default = Path(__file__).resolve().parents[2] / "research_history.jsonl"
    return Path(os.getenv("RESEARCH_HISTORY_PATH", default))


def save_research(state: AppState, question: str) -> None:
    # Save the research question, comparison, and ideas to the history file
    record = {
        "date": datetime.now().isoformat(timespec="minutes"),
        "question": question,
        "comparison": state.get("comparison", ""),
        "ideas": state.get("ideas", ""),
    }
    with _history_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history() -> list[dict]:
    # All saved runs, newest first
    path = _history_path()
    if not path.exists():
        return []
    records = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # blank or partially written line (e.g. interrupted run): skip it rather than fail
    return records
