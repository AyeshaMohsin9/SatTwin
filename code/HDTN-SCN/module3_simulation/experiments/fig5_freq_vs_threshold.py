# Module 3 — Fig. 5: migration frequency and average PS-DT latency vs. latency threshold.
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module3_simulation import sweep_threshold

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _p(*parts):
    return os.path.join(HERE, *parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con", nargs="+", default=["con1", "con2"])
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    ap.add_argument("--horizon", type=float, default=6000.0)
    ap.add_argument("--window", type=float, default=6000.0)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--out", default=_p("results", "fig5.png"))
    args = ap.parse_args()

    stations = _p("config", "stations.yaml")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for con in args.con:
        scenario = _p("config", f"{con}.yaml")
        rows = sweep_threshold(scenario, stations, args.thresholds,
                               args.horizon, args.window, dt_s=args.dt)
        thr = [r["threshold_ms"] for r in rows]
        freq = [r["migration_frequency"] for r in rows]
        lat = [r["mean_latency_ms"] for r in rows]
        ax1.plot(thr, freq, "o-", label=f"migration freq ({con})")
        ax2.plot(thr, lat, "s--", label=f"avg latency ({con})")
        for r in rows:
            print(con, r)

    ax1.set_xlabel("Latency threshold (ms)")
    ax1.set_ylabel("Migration frequency (per edge-DT per window)")
    ax2.set_ylabel("Average PS-DT latency (ms)")
    ax1.set_title("Fig. 5 — Migration frequency & latency vs. threshold")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
