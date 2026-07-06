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
    palette = ["#38bdf8", "#4ade80", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c",
               "#22d3ee", "#facc15", "#e879f9", "#34d399", "#60a5fa", "#f87171",
               "#c084fc", "#2dd4bf", "#fdba74", "#93c5fd", "#fca5a5", "#86efac",
               "#d8b4fe", "#5eead4", "#fef08a", "#f9a8d4", "#a5b4fc", "#bbf7d0"]

    gid_order = list(stations.keys())

    def _cidx(gid):
        s = stations.get(gid)
        if s is None:
            return -1
        return s.get("cidx", gid_order.index(gid) if gid in gid_order else -1)

    def gcolor(gid):
        i = _cidx(gid)
        if i < 0:
            return "#6b7280"
        return palette[i % len(palette)]

    st_lat = [s["lat"] for s in stations.values()]
    st_lon = [s["lon"] for s in stations.values()]
    st_txt = list(stations.keys())
    st_color = [gcolor(g) for g in stations]
    st_symbol = ["star" if stations[g]["is_ncc"] else "square" for g in stations]

    plotly_frames = []
    cum = 0
    for fr in frames:
        sats = fr["sats"]
        events = fr.get("handover_events", fr.get("migration_events", []))
        handset = {e["sat"] for e in events}
        cum += len(events)
        cum_ho = fr.get("cum_handovers", cum)

        def _handed(s):
            return s.get("handover", s["id"] in handset)

        sat_colors = [gcolor(s["host"]) for s in sats]
        sat_sizes = [13 if _handed(s) else 7 for s in sats]
        sat_line = [dict(width=2.5, color="#ffffff") if _handed(s)
                    else dict(width=0.5, color="#0e1117") for s in sats]
        data = [
            go.Scattergeo(lat=st_lat, lon=st_lon, text=st_txt, mode="markers+text",
                          marker=dict(size=11, color=st_color, symbol=st_symbol,
                                      line=dict(width=1, color="#e6e6e6")),
                          textposition="top center",
                          textfont=dict(size=8, color="#9ca3af"),
                          name="ground stations (controllers)", hoverinfo="text"),
        ]
        for s in sats:
            host = stations.get(s["host"])
            if host:
                data.append(go.Scattergeo(
                    lat=[s["lat"], host["lat"]], lon=[s["lon"], host["lon"]],
                    mode="lines",
                    line=dict(width=1.0, color=gcolor(s["host"])),
                    opacity=0.45, showlegend=False, hoverinfo="skip"))
        data.append(go.Scattergeo(
            lat=[s["lat"] for s in sats], lon=[s["lon"] for s in sats],
            text=[f"{s['id']} ◀ {s['host']}"
                  + (f"  ({s['latency']} ms)" if s['latency'] is not None else "")
                  for s in sats],
            mode="markers",
            marker=dict(size=sat_sizes, color=sat_colors, symbol="circle",
                        line=dict(width=[l["width"] for l in sat_line],
                                  color=[l["color"] for l in sat_line])),
            name="satellites (colored by controller)", hoverinfo="text"))
        for h in events:
            frm = stations.get(h["from"]); to = stations.get(h["to"])
            if frm and to:
                data.append(go.Scattergeo(
                    lat=[frm["lat"], to["lat"]], lon=[frm["lon"], to["lon"]],
                    mode="lines", line=dict(width=3.0, color="#ffffff"),
                    opacity=0.9, showlegend=False, hoverinfo="skip"))
        plotly_frames.append(go.Frame(data=data, name=str(fr["step"]),
                                      layout=dict(title=dict(
                                          text=f"frame {fr['step']} · "
                                               f"handovers so far: {cum_ho}",
                                          font=dict(size=11, color="#e6e6e6")))))

    fig.add_traces(plotly_frames[0].data)
    fig.frames = plotly_frames
    steps = [dict(method="animate", label=str(fr["step"]),
                  args=[[str(fr["step"])],
                        dict(mode="immediate",
                             frame=dict(duration=0, redraw=True))])
             for fr in frames]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.03, y=0.05,
            buttons=[dict(label="▶ play", method="animate",
                          args=[None, dict(frame=dict(duration=450, redraw=True),
                                           fromcurrent=True,
                                           transition=dict(duration=200))]),
                     dict(label="⏸ pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.85, y=0.02,
                      currentvalue=dict(prefix="frame ", font=dict(size=10)),
                      steps=steps)])
    total_ho = cum_ho
    info = (f"iter {demo.get('iter', '-')} | demo latency "
            f"{demo.get('mean_latency', '-'):.2f} ms (greedy "
            f"{demo.get('greedy_latency', '-'):.2f} ms) | {len(frames)} frames | "
            f"{total_ho} total handovers | "
            f"dot color = controlling ground station · big white-ringed dot = handover this frame · "
            f"white arc = handover path")
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
