"""Locust 부하테스트 결과 리포트 생성.

ramp     테스트 B(동시성 램프) 결과 — stage별 실측 RPS/p50/p95/p99/에러율을
         워밍업 구간 제외하고 집계하고, breakpoint 후보와 sentinel(첫 stage
         재현) 비교를 출력한다. 또한 Locust 로그에서 "CPU usage" 경고를 찾아
         load generator 자체가 병목이었을 가능성을 알려준다.
baseline 테스트 A(간격 N 여러 번 독립 실행) 결과 — N별 latency 비교표.

사용법:
    python -m milvus_migration.loadtest.report ramp \
        --request-log results/loadtest/ramp_requests.csv --logfile results/loadtest/ramp.log

    python -m milvus_migration.loadtest.report baseline \
        --stats results/loadtest/baseline_N10_stats.csv results/loadtest/baseline_N5_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

_CPU_WARNING_PATTERN = re.compile(r"cpu usage", re.IGNORECASE)
_DEGRADE_THRESHOLD = 1.2  # sentinel p95가 처음보다 20% 이상 나빠지면 경고
_BREAKPOINT_RATIO = 2.0   # 이전 stage 대비 p95가 이 배율 이상 뛰면 breakpoint 후보


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def _scan_cpu_warnings(logfile: str) -> list[tuple[int, str]]:
    """Locust 로그에서 CPU 경고 라인을 찾는다.

    Locust는 load generator 자체 CPU가 부족해지면
    "CPU usage above ...%! ..." / "CPU usage was too high at some point
    during the test!" 같은 경고를 로그에 남긴다(공식 소스: locust/runners.py).
    """
    path = Path(logfile)
    if not path.exists():
        return []
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if _CPU_WARNING_PATTERN.search(line):
            hits.append((lineno, line.strip()))
    return hits


def _load_request_log(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report_ramp(request_log_path: str, logfile: str | None, warmup_sec: float) -> None:
    rows = _load_request_log(request_log_path)

    by_stage: dict[str, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(row["stage_idx"], []).append(row)

    print("\n===== TEST B (동시성 램프) — stage별 요약 (워밍업 구간 제외) =====")
    print(f"{'stage':>5} {'users':>6} {'n':>10} {'실측RPS':>10} "
          f"{'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'에러율':>8}")

    stage_summary: dict[int, dict] = {}
    for stage_idx_str in by_stage:
        if stage_idx_str in ("", "None"):
            continue
        stage_idx = int(stage_idx_str)
        stage_rows = by_stage[stage_idx_str]
        steady = [r for r in stage_rows if float(r["elapsed_in_stage"] or 0) >= warmup_sec]
        if not steady:
            continue

        target_users = int(steady[0]["target_users"])
        rts = [float(r["response_time_ms"]) for r in steady]
        n_err = sum(1 for r in steady if r["exception"])
        span_sec = max(float(r["elapsed_in_stage"]) for r in steady) - warmup_sec
        rps = len(steady) / span_sec if span_sec > 0 else float("nan")

        stage_summary[stage_idx] = {
            "target_users": target_users,
            "n": len(steady),
            "rps": rps,
            "p50": _percentile(rts, 50),
            "p95": _percentile(rts, 95),
            "p99": _percentile(rts, 99),
            "error_rate": n_err / len(steady),
        }

    ordered = sorted(stage_summary.items())
    for stage_idx, s in ordered:
        print(f"{stage_idx:>5} {s['target_users']:>6} {s['n']:>10,} {s['rps']:>10.1f} "
              f"{s['p50']:>9.1f} {s['p95']:>9.1f} {s['p99']:>9.1f} {s['error_rate']*100:>7.2f}%")

    print("\n--- breakpoint 후보 (이전 stage 대비 p95가 "
          f"{_BREAKPOINT_RATIO}배 이상 뛰거나 에러 발생) ---")
    prev = None
    breakpoint_found = False
    for stage_idx, s in ordered:
        if prev is not None and prev["p95"] > 0:
            if s["p95"] >= prev["p95"] * _BREAKPOINT_RATIO or s["error_rate"] > 0:
                print(f"  stage {stage_idx} (동시 사용자 {s['target_users']}명): "
                      f"p95 {prev['p95']:.1f}ms → {s['p95']:.1f}ms"
                      f"{' / 에러 발생' if s['error_rate'] > 0 else ''} — 여기가 포화 지점일 가능성")
                breakpoint_found = True
        prev = s
    if not breakpoint_found:
        print("  뚜렷한 급등 구간 없음 — 테스트한 동시성 범위 내에서는 포화점을 못 찾음 "
              "(더 높은 동시성으로 재시도 고려)")

    if len(ordered) >= 2:
        first_idx, first = ordered[0]
        last_idx, last = ordered[-1]
        print(f"\n--- sentinel 검증 (stage {first_idx} vs stage {last_idx}, "
              f"둘 다 동시 사용자 {first['target_users']}명) ---")
        if first["target_users"] != last["target_users"]:
            print("  ⚠ 마지막 stage가 sentinel(첫 stage 반복)이 아닌 것 같습니다 — "
                  "STAGES 설정을 확인하세요.")
        else:
            ratio = last["p95"] / first["p95"] if first["p95"] else float("nan")
            print(f"  p95: {first['p95']:.1f}ms → {last['p95']:.1f}ms  (배율 {ratio:.2f}x)")
            if ratio >= _DEGRADE_THRESHOLD:
                print("  ⚠ 마지막이 처음보다 뚜렷이 나쁨 — 테스트 도중 자원 누수/고갈이 "
                      "의심되니 원인을 조사하세요 (load generator 자체 자원, Milvus 메모리 등).")
            elif ratio <= 1 / _DEGRADE_THRESHOLD:
                print("  처음보다 마지막이 더 좋음 — 캐시 워밍업 효과로 보이며 별도 조치는 불필요.")
            else:
                print("  처음과 비슷함 — 테스트 내내 누적된 오염 없이 결과를 신뢰할 수 있음.")

    if logfile:
        hits = _scan_cpu_warnings(logfile)
        print(f"\n--- Locust 로그 CPU 경고 스캔 ({logfile}) ---")
        if hits:
            print(f"  ⚠ 로그에서 CPU 경고가 {len(hits)}건 발견되었습니다 — 위 stage들 중 일부는 "
                  "'Milvus의 한계'가 아니라 'Locust load generator 자체의 한계'를 측정한 것일 "
                  "수 있습니다. 아래 줄 번호 근처의 run_time을 위 stage 표와 대조하고, 그 구간의 "
                  "kubectl top pod(Locust Pod) 기록을 같이 확인하세요. 재현되면 분산 모드(master+worker)로 "
                  "전환해서 load generator를 여러 Pod에 분산하세요.")
            for lineno, line in hits[:10]:
                print(f"    L{lineno}: {line}")
            if len(hits) > 10:
                print(f"    ... 외 {len(hits) - 10}건 (전체는 {logfile} 참고)")
        else:
            print("  CPU 경고 없음 — load generator 자체가 병목이었을 가능성은 낮습니다.")


def report_baseline(stats_paths: list[str]) -> None:
    print("\n===== TEST A (기준선, 간격 N별) 요약 =====")
    print(f"{'파일':<45} {'요청수':>8} {'평균(ms)':>10} {'중간값(ms)':>10} {'p95(ms)':>9} {'실패':>6}")
    for path in stats_paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        agg = next((r for r in rows if r.get("Name") in ("Aggregated", "Total")), None)
        if agg is None:
            print(f"{path:<45} (Aggregated 행을 찾을 수 없음)")
            continue
        n_req = agg.get("Request Count", "?")
        n_fail = agg.get("Failure Count", "?")
        print(f"{path:<45} {n_req:>8} {agg.get('Average Response Time', '?'):>10} "
              f"{agg.get('Median Response Time', '?'):>10} {agg.get('95%', '?'):>9} {n_fail:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Locust 부하테스트 결과 리포트")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_ramp = sub.add_parser("ramp", help="테스트 B(동시성 램프) 리포트")
    p_ramp.add_argument("--request-log", required=True)
    p_ramp.add_argument("--logfile", default=None, help="Locust --logfile로 저장한 로그 (CPU 경고 스캔용)")
    p_ramp.add_argument("--warmup-sec", type=float, default=20.0)

    p_base = sub.add_parser("baseline", help="테스트 A(기준선) 리포트")
    p_base.add_argument("--stats", nargs="+", required=True, help="Locust --csv=... 로 생성된 *_stats.csv 파일들")

    args = ap.parse_args()
    if args.mode == "ramp":
        report_ramp(args.request_log, args.logfile, args.warmup_sec)
    else:
        report_baseline(args.stats)


if __name__ == "__main__":
    main()
