"""Query the EXPERIENCE.md practitioner knowledge base."""
from __future__ import annotations

import re
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "EXPERIENCE.md"


def _parse_experience(path: Path) -> list[dict]:
    """Parse EXPERIENCE.md into sections with entries.

    Returns [{"section": str, "entries": [str, ...]}, ...]
    """
    text = path.read_text()
    sections = []
    current_section = None
    current_entries: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections.append({"section": current_section, "entries": current_entries})
            current_section = line[3:].strip()
            current_entries = []
        elif line.startswith("- **") and current_section is not None:
            current_entries.append(line[2:].strip())
        elif current_entries and line.startswith("  ") and not line.startswith("- **"):
            # Continuation of previous entry
            current_entries[-1] += " " + line.strip()

    if current_section is not None:
        sections.append({"section": current_section, "entries": current_entries})

    return sections


def query_experience(topic: str, experience_path: Path | None = None) -> list[str]:
    """Return EXPERIENCE.md entries matching the topic keywords.

    Case-insensitive. Matches if any topic word appears in section header
    or entry text.
    """
    path = experience_path or _DEFAULT_PATH
    if not path.exists():
        return []

    sections = _parse_experience(path)
    words = [w.lower() for w in re.split(r"\s+", topic.strip()) if w]
    if not words:
        return []

    results = []
    for section in sections:
        section_lower = section["section"].lower()
        section_matches = any(w in section_lower for w in words)
        for entry in section["entries"]:
            entry_lower = entry.lower()
            if section_matches or any(w in entry_lower for w in words):
                results.append(entry)

    return results
