from __future__ import annotations
"""Live Dashboard (Phase 3 preview)

Plotly Dash application that launches the PyFlow CLI in a subprocess with
--json-stream, consumes the JSONL output, and renders live-updating charts:
  * Residual history (Ru, Rv if available, Rp optional)
  * Mean-free divergence (continuity residual)
  * 1D horizontal velocity profile along cavity vertical centerline

This module provides a single entrypoint run() and can be launched with:
    python -m pyflow.dashboard.live_dashboard --nx 64 --ny 64 --steps 500

Design Notes:
  * Keeps simulation fully decoupled (relies only on stable CLI JSON contract).
  * Non-blocking background thread reads subprocess stdout lines safely.
  * Uses in-memory ring buffers to bound memory usage.
  * If Plotly Dash is not installed, provides a helpful error message.

Dependencies (not yet in requirements.txt): dash, plotly
The caller must install these before launching dashboard.
"""
import argparse
import json
import threading
import queue
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

MAX_POINTS = 10_000  # cap history

@dataclass
class StreamState:
    iterations: List[int] = field(default_factory=list)
    continuity: List[float] = field(default_factory=list)
    ru: List[float] = field(default_factory=list)
    rv: List[float] = field(default_factory=list)
    rp: List[float] = field(default_factory=list)
    dt: List[float] = field(default_factory=list)
    cfl: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    velocity_profile_y: List[float] = field(default_factory=list)  # y coordinates
    velocity_profile_u: List[float] = field(default_factory=list)  # u at centerline
    last_raw: Dict[str, Any] = field(default_factory=dict)

    def append(self, obj: Dict[str, Any]):
        it = int(obj.get('iteration', len(self.iterations)))
        res = obj.get('residuals', {})
        diag = obj.get('diagnostics', {})
        self.iterations.append(it)
        self.continuity.append(float(obj.get('continuity', res.get('continuity', 0.0))))
        self.ru.append(float(res.get('Ru') or 0.0))
        self.rv.append(float(res.get('Rv') or 0.0))
        self.rp.append(float(res.get('Rp') or 0.0))
        self.dt.append(float(obj.get('dt') or 0.0))
        self.cfl.append(float(obj.get('CFL') or 0.0))
        self.wall_time.append(float(obj.get('wall_time') or 0.0))
        center_prof = diag.get('u_centerline')  # expected [y_list, u_list]
        if center_prof and isinstance(center_prof, (list, tuple)) and len(center_prof) == 2:
            self.velocity_profile_y = list(center_prof[0])
            self.velocity_profile_u = list(center_prof[1])
        # Trim ring buffers
        for lst in (self.iterations, self.continuity, self.ru, self.rv, self.rp, self.dt, self.cfl, self.wall_time):
            if len(lst) > MAX_POINTS:
                del lst[:len(lst)-MAX_POINTS]
        self.last_raw = obj


def launch_cli_stream(args: List[str], line_queue: queue.Queue[str]):
    """Launch CLI subprocess and push each JSON line into the queue."""
    proc = subprocess.Popen([sys.executable, '-m', 'pyflow.cli', *args, '--json-stream'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    def reader(stream, is_stderr=False):
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if is_stderr:
                # Forward stderr lines to parent stderr (diagnostics suppressed normally)
                sys.stderr.write(line + '\n')
                sys.stderr.flush()
            else:
                line_queue.put(line)
        stream.close()

    threading.Thread(target=reader, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=reader, args=(proc.stderr, True), daemon=True).start()
    return proc


def build_dash_app(state: StreamState, cli_args: List[str], proc):
    try:
        from dash import Dash, dcc, html
        import plotly.graph_objs as go
    except ImportError as e:  # pragma: no cover - requires optional deps
        print("Dash/Plotly not installed. Install with: pip install dash plotly")
        raise SystemExit(1)

    app = Dash(__name__)

    app.layout = html.Div([
        html.H2('PyFlow Live Dashboard'),
        html.Div(f"CLI Args: {' '.join(cli_args)}", style={'fontSize': '12px', 'color': '#555'}),
        html.Div([
            html.Button('Pause', id='pause-btn', n_clicks=0, style={'marginRight': '8px'}),
            html.Button('Resume', id='resume-btn', n_clicks=0, style={'marginRight': '8px'}),
            html.Button('Stop', id='stop-btn', n_clicks=0, style={'marginRight': '16px'}),
            html.Label('Scale:'),
            dcc.Dropdown([
                {'label': 'Log', 'value': 'log'},
                {'label': 'Linear', 'value': 'linear'}
            ], value='log', id='scale-mode', clearable=False, style={'width': '140px', 'display': 'inline-block', 'marginLeft': '6px'}),
            html.Label('Update (ms):', style={'marginLeft': '16px'}),
            dcc.Input(id='update-interval', type='number', value=1000, min=200, step=100, style={'width': '90px'}),
            html.Span(id='run-status', style={'marginLeft': '16px', 'fontWeight': 'bold'})
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
        html.Div([
            dcc.Graph(id='residuals-graph', style={'flex': '1', 'minWidth': '400px'}),
            dcc.Graph(id='continuity-graph', style={'flex': '1', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        html.Div([
            dcc.Graph(id='dt-graph', style={'flex': '1', 'minWidth': '400px'}),
            dcc.Graph(id='cfl-graph', style={'flex': '1', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        html.Div([
            dcc.Graph(id='u-centerline-graph', style={'flex': '1', 'minWidth': '400px'}),
            html.Div(id='summary-panel', style={'flex': '1', 'minWidth': '350px', 'padding': '12px', 'background': '#fafafa', 'border': '1px solid #ddd', 'fontFamily': 'monospace'})
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        dcc.Store(id='control-state', data={'paused': False, 'stopped': False, 'last_figures': {}}),
        dcc.Interval(id='tick', interval=1000, n_intervals=0),
    ], style={'fontFamily': 'Arial, sans-serif'})

    from dash import Output, Input, State as DashState, no_update

    @app.callback(
        Output('control-state', 'data'),
        [Input('pause-btn', 'n_clicks'), Input('resume-btn', 'n_clicks'), Input('stop-btn', 'n_clicks')],
        [DashState('control-state', 'data')]
    )
    def _control(pause_clicks, resume_clicks, stop_clicks, data):  # pragma: no cover
        # Determine most recent action (import dash lazily to avoid hard dependency at static analysis time)
        try:
            import dash  # type: ignore
            ctx = getattr(dash, 'callback_context', None)
            changed = [p['prop_id'] for p in ctx.triggered] if ctx and ctx.triggered else []
        except Exception:
            changed = []
        if not changed:
            return data
        trig = changed[0]
        if 'pause-btn' in trig:
            data['paused'] = True
        elif 'resume-btn' in trig:
            data['paused'] = False
        elif 'stop-btn' in trig:
            if not data.get('stopped'):
                try:
                    proc.terminate()
                except Exception:
                    pass
            data['stopped'] = True
        return data

    @app.callback(
        Output('tick', 'interval'),
        [Input('update-interval', 'value')]
    )
    def _update_interval(val):  # pragma: no cover
        try:
            if val and val >= 200:
                return int(val)
        except Exception:
            pass
        return 1000

    @app.callback(
        [
            Output('residuals-graph', 'figure'),
            Output('continuity-graph', 'figure'),
            Output('u-centerline-graph', 'figure'),
            Output('dt-graph', 'figure'),
            Output('cfl-graph', 'figure'),
            Output('run-status', 'children')
        ],
        [Input('tick', 'n_intervals'), Input('scale-mode', 'value')],
        [DashState('control-state', 'data')]
    )
    def _update(_n, scale_mode, control):  # pragma: no cover
        its = state.iterations
        status = 'Stopped' if control.get('stopped') else ('Paused' if control.get('paused') else 'Running')
        if control.get('paused') or control.get('stopped'):
            # Keep last figures (no updates) if we previously stored them
            figs = control.get('last_figures') or {}
            return (
                figs.get('res') or {'data': [], 'layout': {'title': 'Residual History'}},
                figs.get('cont') or {'data': [], 'layout': {'title': 'Continuity'}},
                figs.get('prof') or {'data': [], 'layout': {'title': 'Centerline u'}},
                figs.get('dt') or {'data': [], 'layout': {'title': 'dt'}},
                figs.get('cfl') or {'data': [], 'layout': {'title': 'CFL'}},
                status
            )
        def mk_trace(xs, ys, name, mode='lines'):
            return {'type': 'scatter', 'x': xs, 'y': ys, 'name': name, 'mode': mode}
        def yaxis(scale):
            return {'type': 'log' if scale == 'log' else 'linear', 'title': ''}
        # Residuals
        res_traces = []
        if any(v != 0 for v in state.ru): res_traces.append(mk_trace(its, state.ru, 'Ru'))
        if any(v != 0 for v in state.rv): res_traces.append(mk_trace(its, state.rv, 'Rv'))
        if any(v != 0 for v in state.rp): res_traces.append(mk_trace(its, state.rp, 'Rp'))
        fig_res = {'data': res_traces, 'layout': {'title': 'Residual History', 'xaxis': {'title': 'Iteration'}, 'yaxis': {'type': 'log' if scale_mode=='log' else 'linear', 'title': 'Residual'}}}
        fig_cont = {'data': [mk_trace(its, state.continuity, 'continuity')], 'layout': {'title': 'Continuity Residual', 'xaxis': {'title': 'Iteration'}, 'yaxis': {'type': 'log' if scale_mode=='log' else 'linear', 'title': '||div||'}}}
        fig_prof = {'data': [mk_trace(state.velocity_profile_u, state.velocity_profile_y, 'u(y)', 'lines+markers')] if state.velocity_profile_u else [], 'layout': {'title': 'Centerline Horizontal Velocity', 'xaxis': {'title': 'u'}, 'yaxis': {'title': 'y'}}}
        fig_dt = {'data': [mk_trace(its, state.dt, 'dt')], 'layout': {'title': 'dt vs Iteration', 'xaxis': {'title': 'Iteration'}, 'yaxis': {'type': 'linear', 'title': 'dt'}}}
        fig_cfl = {'data': [mk_trace(its, state.cfl, 'CFL')], 'layout': {'title': 'CFL vs Iteration', 'xaxis': {'title': 'Iteration'}, 'yaxis': {'type': 'linear', 'title': 'CFL'}}}
        # Store last figs in control (client side would need another callback; here we just return status)
        control['last_figures'] = {'res': fig_res, 'cont': fig_cont, 'prof': fig_prof, 'dt': fig_dt, 'cfl': fig_cfl}
        return fig_res, fig_cont, fig_prof, fig_dt, fig_cfl, status

    @app.callback(
        Output('summary-panel', 'children'),
        [Input('tick', 'n_intervals')],
        [DashState('control-state', 'data')]
    )
    def _summary(_n, control):  # pragma: no cover
        if not state.iterations:
            return 'Waiting for data...'
        idx = -1
        return (
            f"Iteration: {state.iterations[idx]}\n"
            f"Continuity: {state.continuity[idx]:.3e}\n"
            f"Ru: {state.ru[idx]:.3e}  Rv: {state.rv[idx]:.3e}  Rp: {state.rp[idx]:.3e}\n"
            f"dt: {state.dt[idx]:.4g}  CFL: {state.cfl[idx]:.3g}\n"
            f"Wall Time: {state.wall_time[idx]:.2f}s\n"
            f"Status: {'Stopped' if control.get('stopped') else ('Paused' if control.get('paused') else 'Running')}"
        )

    return app


def run(argv: Optional[List[str]] = None):  # pragma: no cover - integration function
    parser = argparse.ArgumentParser(description='PyFlow Live Dashboard')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--re', type=float, default=100.0)
    parser.add_argument('--lid-velocity', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--scheme', choices=['quick', 'upwind'], default='quick')
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, help='Additional args passed verbatim to pyflow.cli')
    args = parser.parse_args(argv)

    cli_args = [
        f"--nx={args.nx}", f"--ny={args.ny}", f"--re={args.re}", f"--lid-velocity={args.lid_velocity}",
        f"--steps={args.steps}", f"--scheme={args.scheme}", f"--cfl={args.cfl}"
    ]
    if args.extra_args:
        cli_args.extend(args.extra_args)

    line_queue: queue.Queue[str] = queue.Queue()
    state = StreamState()
    proc = launch_cli_stream(cli_args, line_queue)

    def ingest():  # pragma: no cover - background thread
        while proc.poll() is None:
            try:
                line = line_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get('type') == 'step':
                state.append(obj)
        # Drain remaining
        while not line_queue.empty():
            try:
                obj = json.loads(line_queue.get_nowait())
                if obj.get('type') == 'step':
                    state.append(obj)
            except Exception:
                pass

    threading.Thread(target=ingest, daemon=True).start()

    app = build_dash_app(state, cli_args, proc)
    print("[dashboard] Launching Dash server on port", args.port)
    app.run(host='127.0.0.1', port=args.port, debug=False)

if __name__ == '__main__':  # pragma: no cover
    run()
