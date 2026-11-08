#!/usr/bin/env python3
"""Verify committed centered inputs and execute the offline saved-data checks."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlsplit
import numpy as np
from reproduce_centered import json_safe


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def verify(root, output):
    manifest=json.loads((root/"data/centered/provenance/manifest.json").read_text())
    checks=[]
    def check(name, passed, actual=None, tolerance=None):
        checks.append(dict(name=name,passed=bool(passed),actual=actual,tolerance=tolerance))
    seen=set()
    for asset in manifest["assets"]:
        relative=asset["path"];p=(root/relative).resolve()
        check("unique asset: "+relative,relative not in seen);seen.add(relative)
        if not p.is_relative_to(root) or not p.is_file():
            check("present inside checkout: "+relative,False);continue
        check("SHA256: "+relative,digest(p)==asset["sha256"])
        check("size: "+relative,p.stat().st_size==asset["size_bytes"],p.stat().st_size)
        if "array_sha256" in asset:
            with np.load(p,allow_pickle=False) as z:
                check("NPZ keys: "+relative,set(z.files)==set(asset["array_sha256"]))
                for key,expected in asset["array_sha256"].items():
                    arr=z[key]
                    check("array identity: "+relative+":"+key,hashlib.sha256(arr.tobytes(order="C")).hexdigest()==expected)
                    if key in asset.get("shapes",{}):check("shape: "+relative+":"+key,list(arr.shape)==asset["shapes"][key])
                    if key in asset.get("dtypes",{}):check("dtype: "+relative+":"+key,asset["dtypes"][key] in (arr.dtype.str,str(arr.dtype)))
            if "source_column_equality" in asset:check("recorded source-column equality: "+relative,asset["source_column_equality"])
    protected=manifest["protected_media"]
    directory=root/"data/3DGifs_Simulations"
    actual={p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    expected={f["path"] for f in protected["files"]}
    check("protected recursive path set (including untracked)",actual==expected,sorted(actual))
    for f in protected["files"]:
        p=root/f["path"]
        check("protected regular file: "+f["path"],p.is_file() and not p.is_symlink())
        if p.is_file():
            check("protected SHA256: "+f["path"],digest(p)==f["sha256"])
            check("protected mode: "+f["path"],oct(stat.S_IMODE(p.stat().st_mode))==f["mode"])
    for path in manifest.get("legacy_paths",[]):check("legacy path preserved: "+path,(root/path).is_file())
    for f in manifest.get("legacy_code",[]):check("legacy code unchanged: "+f["path"],digest(root/f["path"])==f["sha256"])
    # Link validation intentionally excludes the protected legacy README.
    docs=["README.md","REPRODUCING.md","PROJECT_SCOPE.md","Loaders/README.md","Solvers/README.md","data/README.md","data/centered/README.md","data/centered/tables/README.md","paper/README.md",
          "data/processed-summary-diagnostics-tables/README_S4_TimeWindow.md",
          "data/processed-summary-diagnostics-tables/ReadMe_S7_Boundary_Layer_Duhamel.md",
          "data/processed-summary-diagnostics-tables/representative-diagnostics/ReadMe.md"]
    for name in docs:
        p=root/name
        check("documentation exists: "+name,p.is_file())
        if not p.is_file():continue
        for match in re.finditer(r"\]\(([^)\s]+)\)",p.read_text()):
            target=match.group(1)
            if urlsplit(target).scheme:continue
            rel=unquote(target.split("#",1)[0]);dest=(p.parent/rel).resolve() if rel else p.resolve()
            check("local link: "+name+" -> "+target,dest.exists() and dest.is_relative_to(root))
            if "#" in target and dest.is_file() and dest.suffix==".md":
                fragment=unquote(target.split("#",1)[1])
                body=dest.read_text()
                headings=re.findall(r"^#{1,6} +(.+)$",body,re.M)
                slugs={re.sub(r"[^\w\- ]","",heading.lower()).replace(" ","-") for heading in headings}
                explicit=set(re.findall(r'id=["\']([^"\']+)["\']',body))
                check("local anchor: "+name+" -> "+target,fragment in slugs|explicit)
    from centered.supplement_tables import presentation_checks
    for c in presentation_checks(root):
        check(c["name"], c["passed"], c.get("actual"))
    suite=unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromName("centered.test_numerics"),
        unittest.defaultTestLoader.loadTestsFromName("centered.test_supplement_tables"),
    ])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    check("analytic edge-case tests",result.wasSuccessful(),result.testsRun)
    os.environ.setdefault("MPLBACKEND","Agg");os.environ.setdefault("MPLCONFIGDIR",str(output/".matplotlib"))
    groups={}
    for group in ("main","diagnostics","controls"):
        destination=output/group;destination.mkdir(parents=True,exist_ok=True)
        groups[group]=importlib.import_module("centered."+group).run(root,destination)
        for c in groups[group].get("checks",[]):
            check(group+": "+c.get("name","unnamed check"),c.get("passed",False),c.get("actual"),c.get("tolerance"))
    from centered.supplement_tables_a import summary_checks, manuscript_comparisons
    for c in summary_checks(root, output/"diagnostics"):
        check(c["name"], c["passed"], c.get("actual"), c.get("tolerance"))
    printed_comparisons=manuscript_comparisons(root)
    printed_conflicts=[c for c in printed_comparisons if not c["agrees_at_printed_precision"]]
    declared=manifest["supplementary_presentation"]["printed_value_conflicts_S1_S6"]
    conflict_key=lambda c:(c["table_id"],str(c["row_id"]),c["quantity"])
    check("printed-value discrepancies explicitly retained in provenance", {conflict_key(c) for c in printed_conflicts}=={conflict_key(c) for c in declared}, len(printed_conflicts))
    from centered.supplement_tables_b import check_generated_inputs
    check("long-time support summary matches bundled derived bins and native sampled norms", check_generated_inputs(root))
    for item in manifest["paper_items"]:
        if item["paper_item"].startswith(("Table S", "Fig. S")):
            presentation=item.get("presentation", "")
            check("presentation mapped: "+item["paper_item"], bool(presentation))
            if presentation:
                page, _, anchor=presentation.partition("#")
                target=root/page
                check("presentation target: "+item["paper_item"], target.is_file() and (not anchor or ('id="'+anchor+'"') in target.read_text()))
            if item["paper_item"].startswith("Fig. S"):
                from PIL import Image
                image_path=item.get("displayed_asset")
                check("figure preview mapped: "+item["paper_item"], bool(image_path))
                if image_path and (root/image_path).is_file():
                    with Image.open(root/image_path) as im:
                        check("readable figure dimensions: "+item["paper_item"], im.width>=800 and im.height>=400, [im.width,im.height])
                        im.verify()
            else:
                csv_paths=item.get("canonical_csvs",[])
                check("canonical table CSVs mapped: "+item["paper_item"], bool(csv_paths) and all((root/p).is_file() for p in csv_paths))
    covered=" ".join(str(p.get("paper_item","")) for p in manifest["paper_items"])
    for item in ["Fig. "+str(n) for n in range(1,4)]+["Fig. S"+str(n) for n in range(1,14)]+["Table S"+str(n) for n in range(1,13)]:
        check("coverage: "+item,any(item==str(x.get("paper_item")) for x in manifest["paper_items"]))
    report=json_safe({"checks":checks,"groups":groups,"printed_value_comparisons":printed_comparisons,"printed_value_discrepancies":printed_conflicts,"unavailable_items":[x for x in manifest["paper_items"] if "not bundled" in str(x.get("reproduction_level","")) or "unavailable" in str(x.get("reproduction_level",""))],"environment":manifest["tested_environment"]})
    (output/"verification_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    failed=[x["name"] for x in checks if not x["passed"]]
    print(f"{len(checks)-len(failed)}/{len(checks)} checks passed; report: {output/'verification_report.json'}")
    if failed:raise SystemExit("FAILED:\n"+"\n".join(failed))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    output=(args.output or root/"build/verify").resolve()
    if output==root or (output.is_relative_to(root) and not output.is_relative_to(root/"build")):p.error("Output inside checkout must be under build/")
    output.mkdir(parents=True,exist_ok=True);verify(root,output)
if __name__=="__main__":main()
