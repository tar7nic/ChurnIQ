"""
fix_encoding.py
---------------
Replaces all non-cp1252-safe unicode characters in src/*.py with ASCII
equivalents and ensures sys.stdout.reconfigure(utf-8) is present in each file.
Run once from the project root, then delete.
"""

import pathlib

SRC = pathlib.Path("src")

REPLACEMENTS = {
    "[✓]": "[OK]",
    "✓":   "OK",
    "✗":   "X",
    "×":   "x",
    "→":   "->",
    "←":   "<-",
    "★":   "*",
    "📊":  "",
    "📈":  "",
    "🧪":  "",
    "📡":  "",
    "✅":  "[OK]",
    "⚠️":  "[!]",
    "⚠":   "[!]",
    "🚨":  "[!!]",
    "🏆":  "",
    "📋":  "",
    "📅":  "",
    "💰":  "",
    "⭐":  "",
    "🎫":  "",
    "💳":  "",
    "🔗":  "",
    "🔴":  "",
    "🟠":  "",
    "🟡":  "",
    "🟢":  "",
    "💡":  "",
    "🎯":  "",
}

RECONFIGURE_LINE = "sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"

for fp in sorted(SRC.glob("*.py")):
    text = fp.read_text(encoding="utf-8")
    original = text

    # Apply unicode replacements
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    # Ensure sys is imported
    if "import sys\n" not in text and "import sys\r\n" not in text:
        text = text.replace("import os\n", "import os\nimport sys\n", 1)
        if "import sys\n" not in text:
            # fallback: prepend after docstring ends
            lines = text.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    lines.insert(i, "import sys\n")
                    break
            text = "".join(lines)

    # Ensure reconfigure call is present (after imports)
    if "sys.stdout.reconfigure" not in text:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, "\n" + RECONFIGURE_LINE)
        text = "".join(lines)

    if text != original:
        fp.write_text(text, encoding="utf-8")
        print(f"  [Fixed ] {fp.name}")
    else:
        print(f"  [Clean ] {fp.name}")

print("\nDone. All source files sanitized.")
