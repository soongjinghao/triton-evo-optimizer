#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time

from collections import Counter
from datetime import datetime
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_previous(paths):
    passed = {}

    for path in paths:
        if not path.exists():
            continue

        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue

            if r.get("status") != "passed":
                continue

            key = (
                r.get("operator_name"),
                r.get("candidate_role"),
                r.get("source_sha256"),
            )

            if all(key):
                passed[key] = str(path)

    return passed


def kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def make_test(operator, source, target):
    content = source.read_text(encoding="utf-8")

    module_re = (
        rf"(?:kernel|{re.escape(operator)}"
        rf"(?:_\d+)?(?:\.[\w.]+)?)"
    )

    from_pattern = rf"^from\s+{module_re}\s+import"

    content = re.sub(
        from_pattern,
        f"from {operator} import",
        content,
        flags=re.MULTILINE,
    )

    import_pattern = (
        rf"^import\s+{module_re}\b(\s+as\s+\w+)?"
    )

    content = re.sub(
        import_pattern,
        lambda m: f"import {operator}{m.group(1) or ''}",
        content,
        flags=re.MULTILINE,
    )

    target.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/workspace/Agent",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--resume-from",
        action="append",
        default=[],
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets = root / "costModel/datasets"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        root / "costModel/audit/correctness_validation"
        / f"final_{stamp}"
    )

    log_dir = run_dir / "logs"
    work_root = run_dir / "work"

    log_dir.mkdir(parents=True)
    work_root.mkdir(parents=True)

    previous_paths = [
        root / "costModel/audit/correctness_validation/run_20261124_113808/results.jsonl",
        root / "costModel/audit/correctness_validation/resume_20261124_133146/results.jsonl",
    ]

    previous_paths += [
        Path(p) for p in args.resume_from
    ]

    previous = load_previous(previous_paths)

    print("=" * 76)
    print("FINAL COSTMODEL CORRECTNESS VALIDATION")
    print("=" * 76)
    print("Root       :", root)
    print("Timeout    :", args.timeout)
    print("MSProf     : NOT USED")
    print("IR capture : NOT USED")
    print("Resume pass:", len(previous))
    print("Run dir    :", run_dir)
    print("=" * 76)

    jobs = []

    for op_dir in sorted(datasets.iterdir()):
        meta_file = op_dir / "operator.json"

        if not meta_file.is_file():
            continue

        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        operator = op_dir.name
        test_file = op_dir / "test.py"
        variant_dir = op_dir / "variant"

        roles = meta.get("candidate_roles", [])

        if not roles:
            roles = [
                p.stem for p in sorted(variant_dir.glob("*.py"))
                if p.is_file()
            ]

        for role in roles:
            variant = variant_dir / f"{role}.py"

            jobs.append({
                "operator": operator,
                "role": role,
                "variant": variant,
                "test": test_file,
            })

    total = len(jobs)

    print("Selected candidates:", total)

    if total != 250:
        print("[WARNING] Expected 250 candidates, found", total)

    results = []
    results_path = run_dir / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as output:

        for index, job in enumerate(jobs, 1):

            operator = job["operator"]
            role = job["role"]
            variant = job["variant"]
            test = job["test"]

            record = {
                "operator_name": operator,
                "candidate_role": role,
                "source_path": str(variant),
                "test_path": str(test),
                "status": None,
            }

            if not variant.is_file():
                record["status"] = "variant_missing"

            elif not test.is_file():
                record["status"] = "test_missing"

            else:
                source_hash = sha256(variant)
                test_hash = sha256(test)

                record["source_sha256"] = source_hash
                record["test_sha256"] = test_hash

                key = (operator, role, source_hash)

                old_result = previous.get(key)

                if old_result:
                    record["status"] = "passed"
                    record["reused_previous_result"] = True
                    record["resumed_from"] = old_result

                else:
                    work = work_root / operator / role
                    work.mkdir(parents=True)

                    kernel_file = work / f"{operator}.py"
                    generated_test = work / f"test_{operator}.py"

                    kernel_file.write_text(
                        variant.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

                    make_test(
                        operator,
                        test,
                        generated_test,
                    )

                    home = work / "home"
                    cache = work / "triton_cache"

                    home.mkdir()
                    cache.mkdir()

                    env = os.environ.copy()

                    env["HOME"] = str(home)
                    env["TRITON_CACHE_DIR"] = str(cache)
                    env["XDG_CACHE_HOME"] = str(work / "xdg_cache")

                    old_pythonpath = env.get("PYTHONPATH", "")

                    env["PYTHONPATH"] = os.pathsep.join(
                        x for x in [
                            str(work),
                            str(root),
                            old_pythonpath,
                        ] if x
                    )

                    env["PYTHONUNBUFFERED"] = "1"

                    # Correctness-only: disable collection hooks.
                    for name in [
                        "COSTMODEL_CAPTURE_DIR",
                        "COSTMODEL_CAPTURE_KERNEL",
                        "COSTMODEL_HIVM_SAVE_TEMPS_DIR",
                    ]:
                        env.pop(name, None)

                    log_file = (
                        log_dir / operator / f"{role}.log"
                    )
                    log_file.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    start = time.time()
                    proc = None

                    try:
                        with log_file.open(
                            "w", encoding="utf-8"
                        ) as log:

                            proc = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-u",
                                    str(generated_test),
                                ],
                                cwd=str(work),
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )

                            try:
                                return_code = proc.wait(
                                    timeout=args.timeout
                                )
                                record["status"] = (
                                    "passed"
                                    if return_code == 0
                                    else "failed"
                                )
                                record["exit_code"] = return_code

                            except subprocess.TimeoutExpired:
                                kill_process_group(proc)
                                record["status"] = "timeout"

                    except Exception as exc:
                        if proc is not None and proc.poll() is None:
                            kill_process_group(proc)

                        record["status"] = "execution_error"
                        record["error"] = repr(exc)

                    record["duration_seconds"] = round(
                        time.time() - start, 3
                    )
                    record["stdout_log"] = str(log_file)

                    # Remove large per-candidate caches.
                    shutil.rmtree(home, ignore_errors=True)
                    shutil.rmtree(cache, ignore_errors=True)
                    shutil.rmtree(
                        work / "xdg_cache",
                        ignore_errors=True,
                    )

            results.append(record)

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output.flush()

            reused = record.get("reused_previous_result", False)

            suffix = " [REUSED]" if reused else ""

            print(
                f"[{index}/{total}] "
                f"{operator}/{role}: "
                f"{record['status']}{suffix}",
                flush=True,
            )

    counts = Counter(r["status"] for r in results)

    summary = {
        "schema_version": "costmodel_final_correctness_v1",
        "run_id": stamp,
        "selected": total,
        "recorded": len(results),
        "counts": dict(counts),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "timeout": counts.get("timeout", 0),
        "variant_missing": counts.get("variant_missing", 0),
        "test_missing": counts.get("test_missing", 0),
        "execution_error": counts.get("execution_error", 0),
        "msprof_used": False,
        "ir_capture_used": False,
        "results": str(results_path),
        "run_dir": str(run_dir),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (run_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator_name",
                "candidate_role",
                "status",
                "source_sha256",
                "duration_seconds",
                "reused_previous_result",
                "stdout_log",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 76)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 76)

    return 0 if (
        len(results) == total
        and all(r["status"] == "passed" for r in results)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
