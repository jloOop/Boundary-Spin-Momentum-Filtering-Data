"""Deterministic, read-only CSV-to-Markdown presentation of the Supplement."""
import csv
from decimal import Decimal, InvalidOperation, localcontext
import os
from pathlib import Path
import re

CATALOG = Path("data/centered/tables/README.md")


def sections(root):
    from .supplement_tables_a import build_sections as first
    from .supplement_tables_b import build_sections as second
    items = first(root) + second(root)
    expected = [f"table-s{n}" for n in range(1, 13)] + ["long-time-summary"]
    if [item["id"] for item in items] != expected:
        raise ValueError("Supplement sections must be S1–S12 followed by the unnumbered long-time summary")
    return items


def escape(value):
    return str(value).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def cell(raw, column):
    """Round decimal CSV tokens only at presentation time; never alter the CSV."""
    raw = str(raw).strip()
    mapping = column.get("values", {})
    if raw in mapping:
        return escape(mapping[raw])
    spec = column.get("format", "text")
    if spec in ("text", "raw"):
        return escape(raw)
    if not raw or raw.lower() in ("nan", "na", "n/a"):
        if column.get("allow_missing", False):
            return "—"
        raise ValueError(f"Missing numeric value for {column['key']}")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric CSV field {column['key']}: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"Non-finite numeric CSV field {column['key']}")
    with localcontext() as ctx:
        ctx.prec = 50
        if spec == "d":
            if value != value.to_integral_value():
                raise ValueError(f"Non-integral count in {column['key']}")
            return str(int(value))
        if not re.fullmatch(r"(?:\.\d+)?[feg]", spec):
            raise ValueError(f"Unsupported display format {spec!r}")
        return format(value, spec)


def panel_rows(root, panel):
    source = (root / panel["csv"]).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError(f"CSV must exist within checkout: {panel['csv']}")
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("Presentation summary exceeds 1 MiB; do not downsample native series to fix this")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)):
            raise ValueError(f"Missing or duplicate CSV headers: {panel['csv']}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Inconsistent CSV row structure: {panel['csv']}")
    if "expected_total_rows" in panel and len(rows) != panel["expected_total_rows"]:
        raise ValueError(f"Unexpected total row count: {panel['csv']}")
    if "row_slice" in panel:
        rows = rows[slice(*panel["row_slice"])]
    if len(rows) != panel["expected_rows"]:
        raise ValueError(f"Expected {panel['expected_rows']} rows, got {len(rows)}: {panel['csv']}")
    columns = panel["columns"]
    for column in columns:
        if column["key"] not in header:
            raise ValueError(f"Unknown column {column['key']}: {panel['csv']}")
    if panel.get("row_key"):
        keys = [tuple(row[key] for key in panel["row_key"]) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate row identifiers: {panel['csv']}")
    return rows


def render(root):
    items = sections(root)
    lines = [
        "# Supplementary tables",
        "",
        "Readable Tables S1–S12 for the **28 October 2026** manuscript snapshot. "
        "Each panel links its canonical CSV. The CSV retains its available precision; only the Markdown display is rounded. "
        "Printed digits do not establish numerical accuracy. Probabilities, fractions and spin polarizations are dimensionless; "
        "other quantities use the manuscript's simulation units, with definitions below each table.",
        "",
        "[Figure gallery](../README.md) · [Main results](../../../README.md) · "
        "[Reproduction instructions](../../../REPRODUCING.md) · "
        "[Source discrepancies](../../../PROJECT_SCOPE.md) · [Provenance](../provenance/manifest.json)",
        "",
        "**Evidence labels:** native-record recomputation, archived derived results, and manuscript-reported rounded "
        "values describe different capabilities. Implementation labels (centered, legacy one-sided, or mixed) are stated separately. "
        "Missing native inputs are never counted as independently reproduced results.",
        "",
        "| Item | Result |",
        "|---|---|",
    ]
    for item in items:
        label = item.get("paper_item", "Long-time summary" if item["id"] == "long-time-summary" else item["id"].replace("table-s", "Table S"))
        lines.append(f"| [{label}](#{item['id']}) | {escape(item['title'])} |")
    for item in items:
        label = item.get("paper_item", "Long-time summary" if item["id"] == "long-time-summary" else item["id"].replace("table-s", "Table S"))
        lines.extend(["", f'<a id="{item["id"]}"></a>', "", f"## {label} — {item['title']}", "", item["intro"], ""])
        for panel in item["panels"]:
            rows = panel_rows(root, panel)
            if panel.get("label"):
                lines.extend([f"### {panel['label']}", ""])
            link = Path(os.path.relpath(root / panel["csv"], (root / CATALOG).parent)).as_posix()
            lines.extend([f"[Numerical CSV]({link})", ""])
            cols = panel["columns"]
            lines.append("| " + " | ".join(escape(c["label"]) for c in cols) + " |")
            lines.append("| " + " | ".join("---" if c.get("format", "text") in ("text", "raw") else "---:" for c in cols) + " |")
            lines.extend("| " + " | ".join(cell(row[c["key"]], c) for c in cols) + " |" for row in rows)
            if panel.get("note"):
                lines.extend(["", panel["note"]])
            lines.append("")
        for field, heading in [("dictionary", "Definitions"), ("interpretation", "Interpretation"), ("implementation", "Implementation"), ("status", "Source and reproduction status")]:
            if item.get(field):
                lines.extend([f"**{heading}.** {item[field]}", ""])
        lines.extend(["[Back to table index](#supplementary-tables)", ""])
    lines.extend(["<!-- Generated by Loaders/render_supplement_tables.py. Edit the CSVs/specifications, then regenerate. -->", ""])
    return "\n".join(lines)


def presentation_checks(root):
    """Checks used by the existing centered verifier; no writes or scientific recomputation."""
    checks = []
    try:
        expected = render(root)
        checks.append({"name": "supplement CSV schemas, row counts, numeric fields and S1–S12 identifiers", "passed": True})
        target = root / CATALOG
        checks.append({"name": "supplement Markdown equals canonical CSVs after declared rounding", "passed": target.is_file() and target.read_text(encoding="utf-8") == expected})
        sources = {p["csv"] for item in sections(root) for p in item["panels"]}
        checks.append({"name": "supplement canonical summary CSVs are compact", "passed": all((root / p).stat().st_size <= 1024 * 1024 for p in sources), "actual": len(sources)})
    except (ValueError, KeyError, OSError) as exc:
        checks.append({"name": "supplement presentation validation", "passed": False, "actual": str(exc)})
    return checks
