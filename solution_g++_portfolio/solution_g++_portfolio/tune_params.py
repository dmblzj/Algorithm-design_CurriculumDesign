#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402


DEFAULT_CASES = [
    "case020.in",
    "case060.in",
    "case090.in",
    "case095.in",
    "case098.in",
    "case100.in",
]

DEFAULT_CONFIG = {
    "wait_weight": 1.0,
    "memory_weight": 1.0,
    "finish_weight": 1.0,
    "fragment_weight": 1.0,
    "scarcity_weight": 1.0,
    "tail_weight": 1.0,
    "duration_bias": 0.0001,
    "random_repair_ratio": 0.25,
    "repair_rounds": 0,
    "candidate_limit": 0,
    "local_search_time_small": 1.2,
    "local_search_time_medium": 2.5,
    "local_search_time_large": 5.0,
    "local_search_time_huge": 8.0,
    "proxy_wait_weight": 1.0,
    "proxy_memory_weight": 1.0,
    "proxy_finish_weight": 1.0,
}

FLOAT_RANGES = {
    "wait_weight": (0.55, 2.40),
    "memory_weight": (0.15, 2.80),
    "finish_weight": (0.35, 2.40),
    "fragment_weight": (0.0, 2.20),
    "scarcity_weight": (0.0, 2.20),
    "tail_weight": (0.25, 2.60),
    "duration_bias": (0.0, 0.0012),
    "random_repair_ratio": (0.05, 0.45),
    "local_search_time_small": (0.8, 1.8),
    "local_search_time_medium": (1.8, 4.0),
    "local_search_time_large": (3.5, 8.0),
    "local_search_time_huge": (6.0, 12.0),
    "proxy_wait_weight": (0.60, 1.70),
    "proxy_memory_weight": (0.45, 2.20),
    "proxy_finish_weight": (0.45, 2.20),
}

INT_RANGES = {
    "repair_rounds": (20, 95),
    "candidate_limit": (0, 50),
}

CSV_FIELDS = [
    "param_id",
    "legal_count",
    "case_count",
    "weighted_wait_sum",
    "memory_waste_sum",
    "max_finish_sum",
    "runtime_sum",
    "normalized_score",
    "config_path",
    "error",
]

CASE_CSV_FIELDS = [
    "param_id",
    "case",
    "legal",
    "runtime_seconds",
    "weighted_wait",
    "memory_waste",
    "max_finish",
    "error",
]


def load_config(path):
    config = dict(DEFAULT_CONFIG)
    if path:
        config.update(json.loads(Path(path).read_text()))
    return config


def log_uniform(rng, low, high):
    if low <= 0:
        return rng.uniform(low, high)
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def random_config(rng, base):
    config = dict(base)
    for key, (low, high) in FLOAT_RANGES.items():
        if key in {"duration_bias", "random_repair_ratio"} or low == 0:
            config[key] = rng.uniform(low, high)
        else:
            config[key] = log_uniform(rng, low, high)
    for key, (low, high) in INT_RANGES.items():
        config[key] = rng.randint(low, high)
    return config


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def build_solution(solution_dir):
    result = subprocess.run(["bash", str(solution_dir / "build.sh")], cwd=solution_dir, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"build failed with return code {result.returncode}")


def run_solution(solution_dir, input_path, output_path, timeout_seconds, config_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SCHED_CONFIG"] = str(config_path.resolve())
    start = time.perf_counter()

    with output_path.open("w") as output_file:
        result = subprocess.run(
            ["bash", str(solution_dir / "run.sh")],
            cwd=solution_dir,
            input=input_path.read_text(),
            text=True,
            stdout=output_file,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=env,
        )

    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"return code {result.returncode}")
    return elapsed


def evaluate_config(param_id, config_path, cases, solution_dir, output_dir, timeout):
    row = {
        "param_id": param_id,
        "legal_count": "0",
        "case_count": str(len(cases)),
        "weighted_wait_sum": "0",
        "memory_waste_sum": "0",
        "max_finish_sum": "0",
        "runtime_sum": "0.000000",
        "normalized_score": "",
        "config_path": str(config_path),
        "error": "",
    }

    weighted_wait_sum = 0
    memory_waste_sum = 0
    max_finish_sum = 0
    runtime_sum = 0.0
    legal_count = 0
    errors = []
    case_rows = []

    for input_path in cases:
        output_path = output_dir / param_id / f"{input_path.stem}.out"
        case_row = {
            "param_id": param_id,
            "case": input_path.name,
            "legal": "NO",
            "runtime_seconds": "",
            "weighted_wait": "",
            "memory_waste": "",
            "max_finish": "",
            "error": "",
        }
        try:
            runtime = run_solution(solution_dir, input_path, output_path, timeout, config_path)
            metrics = judge.validate_output(input_path, output_path)
            weighted_wait_sum += metrics["weighted_wait"]
            memory_waste_sum += metrics["memory_waste"]
            max_finish_sum += metrics["max_finish"]
            runtime_sum += runtime
            legal_count += 1
            case_row.update({
                "legal": "YES",
                "runtime_seconds": f"{runtime:.6f}",
                "weighted_wait": str(metrics["weighted_wait"]),
                "memory_waste": str(metrics["memory_waste"]),
                "max_finish": str(metrics["max_finish"]),
            })
        except subprocess.TimeoutExpired:
            errors.append(f"{input_path.name}: timeout")
            case_row["error"] = "timeout"
        except Exception as exc:
            errors.append(f"{input_path.name}: {exc}")
            case_row["error"] = str(exc)
        case_rows.append(case_row)

    row.update({
        "legal_count": str(legal_count),
        "weighted_wait_sum": str(weighted_wait_sum),
        "memory_waste_sum": str(memory_waste_sum),
        "max_finish_sum": str(max_finish_sum),
        "runtime_sum": f"{runtime_sum:.6f}",
        "error": " | ".join(errors[:5]),
    })
    return row, case_rows


def read_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as file:
        return list(csv.DictReader(file))


def metric_value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.inf


def is_full_legal(row):
    return row.get("legal_count") == row.get("case_count") and row.get("case_count") not in {"", "0", None}


def recalculate_scores(rows, case_rows, score_weights):
    legal_case_rows = [row for row in case_rows if row.get("legal") == "YES"]
    if not legal_case_rows:
        for row in rows:
            row["normalized_score"] = ""
        return

    best_by_case = {}
    for row in legal_case_rows:
        case = row["case"]
        current = best_by_case.setdefault(case, {
            "weighted_wait": math.inf,
            "memory_waste": math.inf,
            "max_finish": math.inf,
        })
        current["weighted_wait"] = min(current["weighted_wait"], metric_value(row, "weighted_wait"))
        current["memory_waste"] = min(current["memory_waste"], metric_value(row, "memory_waste"))
        current["max_finish"] = min(current["max_finish"], metric_value(row, "max_finish"))

    by_param = {}
    for row in case_rows:
        by_param.setdefault(row["param_id"], {})[row["case"]] = row

    a, b, c = score_weights
    for row in rows:
        if not is_full_legal(row):
            row["normalized_score"] = "inf"
            continue
        param_cases = by_param.get(row["param_id"], {})
        if len(param_cases) != int(row["case_count"]):
            row["normalized_score"] = "inf"
            continue

        score = 0.0
        used_cases = 0
        for case, case_row in param_cases.items():
            if case_row.get("legal") != "YES" or case not in best_by_case:
                score = math.inf
                break
            best = best_by_case[case]
            score += (
                a * metric_value(case_row, "weighted_wait") / max(1.0, best["weighted_wait"]) +
                b * metric_value(case_row, "memory_waste") / max(1.0, best["memory_waste"]) +
                c * metric_value(case_row, "max_finish") / max(1.0, best["max_finish"])
            )
            used_cases += 1
        if math.isfinite(score) and used_cases:
            score /= used_cases
        row["normalized_score"] = f"{score:.9f}"


def write_rows(csv_path, rows, fields):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dominates(a, b):
    keys = ["weighted_wait_sum", "memory_waste_sum", "max_finish_sum", "runtime_sum"]
    a_values = [metric_value(a, key) for key in keys]
    b_values = [metric_value(b, key) for key in keys]
    return all(x <= y for x, y in zip(a_values, b_values)) and any(x < y for x, y in zip(a_values, b_values))


def pareto_rows(rows):
    candidates = [row for row in rows if is_full_legal(row)]
    result = []
    for row in candidates:
        if not any(dominates(other, row) for other in candidates if other is not row):
            result.append(row)
    return result


def refresh_top_configs(rows, top_dir, keep_top):
    top_dir.mkdir(parents=True, exist_ok=True)
    for old_file in top_dir.glob("*.json"):
        old_file.unlink()

    legal_rows = [row for row in rows if is_full_legal(row)]
    legal_rows.sort(key=lambda row: metric_value(row, "normalized_score"))
    selected = []
    selected.extend(legal_rows[:keep_top])
    selected.extend(pareto_rows(rows))

    seen = set()
    for index, row in enumerate(selected, start=1):
        config_path = Path(row["config_path"])
        if not config_path.exists() or str(config_path) in seen:
            continue
        seen.add(str(config_path))
        prefix = f"rank{index:03d}"
        if row not in legal_rows[:keep_top]:
            prefix = f"pareto{index:03d}"
        shutil.copyfile(config_path, top_dir / f"{prefix}_{row['param_id']}.json")


def next_param_index(rows):
    highest = 0
    for row in rows:
        param_id = row.get("param_id", "")
        if len(param_id) > 1 and param_id[0] == "p" and param_id[1:].isdigit():
            highest = max(highest, int(param_id[1:]))
    return highest + 1


def resolve_cases(data_dir, args):
    if args.full:
        cases = sorted(data_dir.glob("case*.in"))
    else:
        names = args.cases or DEFAULT_CASES
        cases = [data_dir / name for name in names]

    missing = [path.name for path in cases if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing cases: {', '.join(missing)}")
    if not cases:
        raise FileNotFoundError(f"no case*.in files found in {data_dir}")
    return cases


def main():
    parser = argparse.ArgumentParser(description="Random-search scheduler parameters.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("solution_dir", type=Path)
    parser.add_argument("--cases", nargs="*", help="Case names for quick search")
    parser.add_argument("--full", action="store_true", help="Run all case*.in files")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--base-config", type=Path, help="Start random search around this config")
    parser.add_argument("--config-dir", type=Path, default=Path("solution_g++_portfolio/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_paramsearch"))
    parser.add_argument("--csv", type=Path, default=Path("paramsearch_results.csv"))
    parser.add_argument("--case-csv", type=Path, default=Path("paramsearch_case_results.csv"))
    parser.add_argument("--top-dir", type=Path, default=Path("solution_g++_portfolio/top_configs"))
    parser.add_argument("--keep-top", type=int, default=20)
    parser.add_argument("--score-wait", type=float, default=1.0)
    parser.add_argument("--score-memory", type=float, default=1.0)
    parser.add_argument("--score-finish", type=float, default=1.0)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    solution_dir = args.solution_dir.resolve()
    config_dir = args.config_dir.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = args.csv.resolve()
    case_csv_path = args.case_csv.resolve()
    top_dir = args.top_dir.resolve()

    cases = resolve_cases(data_dir, args)
    if not args.skip_build:
        build_solution(solution_dir)

    rows = read_rows(csv_path)
    case_rows = read_rows(case_csv_path)
    start_index = next_param_index(rows)
    rng = random.Random(args.seed + start_index)
    base_config = load_config(args.base_config)

    for offset in range(args.trials):
        param_index = start_index + offset
        param_id = f"p{param_index:04d}"
        config = dict(base_config) if offset == 0 and args.base_config else random_config(rng, base_config)
        if offset == 0 and not args.base_config:
            config = dict(DEFAULT_CONFIG)
        config_path = config_dir / f"{param_id}.json"
        write_config(config_path, config)

        print(f"running {param_id} on {len(cases)} cases")
        row, new_case_rows = evaluate_config(param_id, config_path, cases, solution_dir, output_dir, args.timeout)
        rows.append(row)
        case_rows.extend(new_case_rows)
        recalculate_scores(rows, case_rows, (args.score_wait, args.score_memory, args.score_finish))
        write_rows(csv_path, rows, CSV_FIELDS)
        write_rows(case_csv_path, case_rows, CASE_CSV_FIELDS)
        refresh_top_configs(rows, top_dir, args.keep_top)
        print(
            f"{param_id}: legal {row['legal_count']}/{row['case_count']}, "
            f"wait={row['weighted_wait_sum']}, memory={row['memory_waste_sum']}, "
            f"finish={row['max_finish_sum']}, runtime={row['runtime_sum']}s"
        )

    legal_rows = [row for row in rows if is_full_legal(row)]
    legal_rows.sort(key=lambda row: metric_value(row, "normalized_score"))
    if legal_rows:
        best = legal_rows[0]
        print(
            f"best: {best['param_id']} score={best['normalized_score']} "
            f"wait={best['weighted_wait_sum']} memory={best['memory_waste_sum']} "
            f"finish={best['max_finish_sum']}"
        )
    print(f"results: {csv_path}")
    print(f"case results: {case_csv_path}")
    print(f"top configs: {top_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
