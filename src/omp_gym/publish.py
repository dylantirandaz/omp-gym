"""Publish the ledger report as a static page.

Writes docs/index.html (a rendered report page). With --push it
commits and pushes only that file. It never changes repository
settings: enabling GitHub Pages stays a manual decision, because a
one-command publish pipeline can amplify a data leak.
"""

import html
import shutil
import subprocess
from pathlib import Path

from .report import report_from_ledger


def _git_binary() -> str:
    """Find a working git. Prefer PATH, then the macOS CLT copy."""
    candidate = shutil.which("git")
    if candidate:
        probe = subprocess.run(
            [candidate, "--version"], capture_output=True
        )
        if probe.returncode == 0:
            return candidate
    clt = "/Library/Developer/CommandLineTools/usr/bin/git"
    probe = subprocess.run([clt, "--version"], capture_output=True)
    if probe.returncode == 0:
        return clt
    raise RuntimeError("no working git binary found")


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>omp-gym report</title>
<style>
body {{ background: #0d1117; color: #e6edf3; max-width: 900px;
       margin: 40px auto; padding: 0 16px;
       font: 15px/1.6 -apple-system, monospace; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
td, th {{ border: 1px solid #30363d; padding: 6px 10px;
          text-align: left; }}
th {{ color: #8b949e; font-weight: normal; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; color: #8b949e; }}
code {{ color: #d29922; }}
</style></head><body>{body}</body></html>
"""


def _markdown_to_html(markdown: str) -> str:
    """Minimal markdown subset: headings, tables, lists, code."""
    lines_out: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", " ", ":"} for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                lines_out.append("<table>")
                in_table = True
            lines_out.append(
                "<tr>"
                + "".join(
                    f"<{tag}>{html.escape(c)}</{tag}>" for c in cells
                )
                + "</tr>"
            )
            continue
        if in_table:
            lines_out.append("</table>")
            in_table = False
        if line.startswith("## "):
            lines_out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            lines_out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "):
            lines_out.append(f"<p>• {html.escape(line[2:])}</p>")
        elif line.strip():
            lines_out.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        lines_out.append("</table>")
    return "\n".join(lines_out)


def publish_report(
    repo_root: Path, ledger_path: Path, push: bool
) -> dict:
    """Render docs/index.html and push only that file on request."""
    report = report_from_ledger(ledger_path)
    docs = repo_root / "docs"
    docs.mkdir(exist_ok=True)
    page = docs / "index.html"
    page.write_text(_PAGE.format(body=_markdown_to_html(report)))

    result = {"page": str(page), "pushed": False}
    if not push:
        return result

    git_bin = _git_binary()

    def git(*args: str) -> None:
        subprocess.run(
            [git_bin, *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )

    git("add", "docs/index.html")
    commit = subprocess.run(
        [git_bin, "commit", "-m", "Publish ledger report page"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if commit.returncode == 0:
        git("push", "-q", "origin", "main")
        result["pushed"] = True
    return result
