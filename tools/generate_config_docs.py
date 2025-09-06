from __future__ import annotations
import json, pathlib, textwrap

SCHEMA_PATH = pathlib.Path("pyfoamclone/configuration/config_schema.json")
OUT_PATH = pathlib.Path("docs/configuration_guide.md")

def main():
    data = json.loads(SCHEMA_PATH.read_text())
    props = data.get("properties", {})
    lines = ["# Configuration Guide", "", "Generated from schema.", ""]
    lines.append("| Name | Type | Allowed/Enum | Default | Description |")
    lines.append("|------|------|-------------|---------|-------------|")
    for name, spec in props.items():
        typ = spec.get("type", "-")
        enum = ",".join(map(str, spec.get("enum", []))) if spec.get("enum") else "-"
        default = spec.get("default", "-")
        desc = spec.get("description", "")
        lines.append(f"| {name} | {typ} | {enum} | {default} | {desc} |")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
