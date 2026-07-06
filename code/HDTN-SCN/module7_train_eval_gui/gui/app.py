# Module 7 — Dash real-time dashboard: reward/latency curves + animated agent demo.
import argparse
import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from module7_train_eval_gui.event_bus import EventBus

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_app(run_dir):
    bus = EventBus(run_dir)
    app = dash.Dash(__name__)
    app.title = "SatTwin MAPPO Live"

    app.layout = html.Div(style={"backgroundColor": "#0e1117", "color": "#e6e6e6",
                                 "fontFamily": "monospace", "padding": "12px"},
                          children=[
        html.H2("SatTwin — MAPPO Digital-Twin Migration (live)"),
        html.Div(id="status", style={"marginBottom": "8px", "fontSize": "15px"}),
        html.Div(style={"display": "grid",
                        "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
            dcc.Graph(id="reward"),
            dcc.Graph(id="latency"),
            dcc.Graph(id="entropy_kl"),
            dcc.Graph(id="migrations"),
        ]),
        html.H3("Agent demonstration — satellite → ground-station assignment"),
        html.Div(id="demo_info", style={"marginBottom": "6px"}),
        dcc.Graph(id="constellation", style={"height": "560px"}),
        dcc.Interval(id="tick", interval=3000, n_intervals=0),
    ])

    def _line(xs, ys, name, color):
        return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                          line=dict(color=color, width=2))

    def _layout(title, ytitle):
        return dict(title=title, template="plotly_dark", height=280,
                    margin=dict(l=45, r=15, t=40, b=35),
                    xaxis_title="iteration", yaxis_title=ytitle,
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")

    @app.callback(
        [Output("status", "children"), Output("reward", "figure"),
         Output("latency", "figure"), Output("entropy_kl", "figure"),
         Output("migrations", "figure"), Output("constellation", "figure"),
         Output("demo_info", "children")],
        Input("tick", "n_intervals"))
    def update(_):
        rows = bus.read_metrics()
        status = bus.read_status() or {}
        its = [r["iter"] for r in rows]

        reward_fig = go.Figure(layout=_layout("Episode return (↑ better)", "return"))
        latency_fig = go.Figure(layout=_layout("Mean PS-DT latency vs greedy", "ms"))
        ek_fig = go.Figure(layout=_layout("Policy entropy & KL", "value"))
        mig_fig = go.Figure(layout=_layout("Migrations per rollout", "count"))

        if rows:
            reward_fig.add_trace(_line(its, [r["return"] for r in rows],
                                       "return", "#4ade80"))
            latency_fig.add_trace(_line(its, [r["mean_latency"] for r in rows],
                                        "MAPPO", "#60a5fa"))
            latency_fig.add_trace(_line(its, [r["greedy_latency"] for r in rows],
                                        "greedy baseline", "#f87171"))
            ek_fig.add_trace(_line(its, [r["entropy"] for r in rows],
                                   "entropy", "#fbbf24"))
            ek_fig.add_trace(_line(its, [r["kl"] for r in rows], "KL", "#a78bfa"))
            mig_fig.add_trace(_line(its, [r["migrations"] for r in rows],
                                    "migrations", "#f472b6"))

        running = status.get("running", False)
        badge = "🟢 RUNNING" if running else "⚪ idle/stopped"
        status_txt = (f"{badge}  |  iter {status.get('iter', '-')}  |  "
                      f"elapsed {status.get('elapsed_h', 0)} h  |  "
                      f"latency {status.get('mean_latency', '-')} ms  "
                      f"(greedy {status.get('greedy_latency', '-')} ms)")

        const_fig, demo_info = _demo_figure(bus)
        return status_txt, reward_fig, latency_fig, ek_fig, mig_fig, const_fig, demo_info

    return app


def _demo_figure(bus):
    demo = bus.read_demo()
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                      geo=dict(projection_type="natural earth", bgcolor="#0e1117",
                               landcolor="#1f2733", oceancolor="#0e1117",
                               showland=True, showocean=True, lakecolor="#0e1117"),
                      margin=dict(l=0, r=0, t=0, b=0))
    if not demo:
        return fig, "waiting for first demo rollout..."
    frames = demo.get("frames", [])
    if not frames:
        return fig, "waiting for frames..."
    stations = demo["stations"]
    st_lat = [s["lat"] for s in stations.values()]
    st_lon = [s["lon"] for s in stations.values()]
    st_txt = list(stations.keys())
    st_color = ["#f87171" if s["is_ncc"] else "#38bdf8" for s in stations.values()]

    plotly_frames = []
    for fr in frames:
        data = [
            go.Scattergeo(lat=st_lat, lon=st_lon, text=st_txt, mode="markers+text",
                          marker=dict(size=9, color=st_color, symbol="square"),
                          textposition="top center",
                          textfont=dict(size=8, color="#9ca3af"), name="ground stations"),
            go.Scattergeo(lat=[s["lat"] for s in fr["sats"]],
                          lon=[s["lon"] for s in fr["sats"]],
                          text=[f"{s['id']} -> {s['host']}" for s in fr["sats"]],
                          mode="markers",
                          marker=dict(size=6, color="#4ade80", symbol="circle"),
                          name="satellites"),
        ]
        for s in fr["sats"]:
            host = stations.get(s["host"])
            if host:
                data.append(go.Scattergeo(
                    lat=[s["lat"], host["lat"]], lon=[s["lon"], host["lon"]],
                    mode="lines", line=dict(width=0.6, color="rgba(148,163,184,0.35)"),
                    showlegend=False, hoverinfo="skip"))
        for mig in fr["migration_events"]:
            frm = stations.get(mig["from"]); to = stations.get(mig["to"])
            if frm and to:
                data.append(go.Scattergeo(
                    lat=[frm["lat"], to["lat"]], lon=[frm["lon"], to["lon"]],
                    mode="lines", line=dict(width=2.5, color="#fbbf24"),
                    showlegend=False, hoverinfo="skip"))
        plotly_frames.append(go.Frame(data=data, name=str(fr["step"])))

    fig.add_traces(plotly_frames[0].data)
    fig.frames = plotly_frames
    fig.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, x=0.05, y=0.05,
        buttons=[dict(label="▶ play", method="animate",
                      args=[None, dict(frame=dict(duration=350, redraw=True),
                                       fromcurrent=True)]),
                 dict(label="⏸ pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")])])])
    info = (f"iter {demo.get('iter', '-')} | demo mean latency "
            f"{demo.get('mean_latency', '-'):.2f} ms (greedy "
            f"{demo.get('greedy_latency', '-'):.2f} ms) | {len(frames)} frames "
            f"| yellow arcs = migration events")
    return fig, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "results", "mappo_run"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    app = build_app(args.run_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
