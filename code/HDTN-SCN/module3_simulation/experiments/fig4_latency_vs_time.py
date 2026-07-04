# Module 3 — Fig. 4: PS-DT latency vs. time for HDTN-SCN and Benchmarks 1/2/3.
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module3_simulation import run_scheme, ALL_SCHEMES

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _p(*parts):
    return os.path.join(HERE, *parts)


def moving_average(x, k):
    if k <= 1 or len(x) < k:
        return x
    out, acc = [], 0.0
    from collections import deque
    q = deque()
    for v in x:
        q.append(v); acc += v
        if len(q) > k:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con", default="con1")
    ap.add_argument("--horizon", type=float, default=36000.0)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--smooth", type=int, default=20)
    ap.add_argument("--out", default=_p("results", "fig4.png"))
    args = ap.parse_args()

    scenario = _p("config", f"{args.con}.yaml")
    stations = _p("config", "stations.yaml")

    plt.figure(figsize=(8, 5))
    summaries = []
    for scheme in ALL_SCHEMES:
        res = run_scheme(scenario, stations, scheme, args.horizon,
                         threshold_ms=args.threshold, dt_s=args.dt)
        y = moving_average(res.mean_latency_series, args.smooth)
        plt.plot(res.times, y, label=f"{scheme} (avg {res.mean_latency:.0f} ms)")
        summaries.append(res.summary())
        print(res.summary())

    plt.xlabel("Time (s)")
    plt.ylabel("PS-DT latency (ms)")
    plt.title(f"Fig. 4 — PS-DT latency vs. time ({args.con})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
