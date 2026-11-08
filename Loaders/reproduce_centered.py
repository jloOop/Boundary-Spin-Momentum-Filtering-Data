#!/usr/bin/env python3
"""Recompute/replot the selected centered saved-data products on a CPU."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_safe(v) for v in value]
    if hasattr(value,"tolist"): return json_safe(value.tolist())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group",choices=("main","diagnostics","controls","all"),default="all")
    p.add_argument("--output",type=Path,help="Default build/GROUP; never a tracked input directory")
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    output=(args.output or root/"build"/args.group).resolve()
    if output==root or (output.is_relative_to(root) and not output.is_relative_to(root/"build")):
        p.error("Inside the checkout, output must be under ignored build/; all input paths are read-only")
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR",str(output/".matplotlib"))
    groups=("main","diagnostics","controls") if args.group=="all" else (args.group,)
    reports={}
    for group in groups:
        dest=output/group if args.group=="all" else output;dest.mkdir(parents=True,exist_ok=True)
        reports[group]=json_safe(importlib.import_module("centered."+group).run(root,dest))
        print(f"{group}: outputs written to {dest}")
    (output/"reproduction_report.json").write_text(json.dumps(reports,indent=2,allow_nan=False)+"\n")
    failed=[f"{group}: {c.get('name','unnamed check')}" for group,r in reports.items() for c in r.get("checks",[]) if isinstance(c,dict) and c.get("passed") is False]
    if failed: raise SystemExit("Failed checks:\n"+"\n".join(failed))
if __name__=="__main__": main()
