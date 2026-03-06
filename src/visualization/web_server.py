#!/usr/bin/env python3
"""
EA-NN  —  Modern web interface
Real-time browser visualisation of the predator/prey neuroevolution simulation.
"""

import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template_string
from flask_socketio import SocketIO, emit


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _clean(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, dict):            return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):      return int(obj)
    if isinstance(obj, np.floating):     return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return obj


# --------------------------------------------------------------------------- #
# Server                                                                       #
# --------------------------------------------------------------------------- #

class EcosystemWebServer:
    """Flask + SocketIO server that streams simulation state to the browser."""

    def __init__(self, canvas):
        self.canvas   = canvas          # _Wrapper from main.py
        self.running  = False
        self.speed_ms = 100            # ms between simulation steps (10 steps/sec default)

        self.app      = Flask(__name__)
        self.app.config["SECRET_KEY"] = "ea-nn"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*",
                                 async_mode="threading")

        self._setup_routes()
        self._setup_events()

    # ----------------------------------------------------------------------- #
    # Routes                                                                   #
    # ----------------------------------------------------------------------- #

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/health")
        def health():
            n = len(getattr(getattr(self.canvas, "env", None), "agents", []))
            return jsonify({"status": "ok", "agents": n})

    # ----------------------------------------------------------------------- #
    # WebSocket events                                                         #
    # ----------------------------------------------------------------------- #

    def _setup_events(self):
        @self.socketio.on("connect")
        def on_connect():
            emit("state", _clean(self._state()))

        @self.socketio.on("start")
        def on_start():
            if not self.running:
                self.running = True
                threading.Thread(target=self._loop, daemon=True).start()
            self.socketio.emit("running", True)

        @self.socketio.on("stop")
        def on_stop():
            self.running = False
            self.socketio.emit("running", False)

        @self.socketio.on("set_speed")
        def on_speed(data):
            self.speed_ms = max(0, min(2000, int(data.get("ms", 100))))

        @self.socketio.on("inspect")
        def on_inspect(data):
            details = self._agent_details(data.get("id"))
            emit("agent_details", _clean(details))

    # ----------------------------------------------------------------------- #
    # Simulation loop (background thread)                                      #
    # ----------------------------------------------------------------------- #

    def _loop(self):
        while self.running:
            try:
                self.canvas.step()
                self.socketio.emit("state", _clean(self._state()))
            except Exception as exc:
                print(f"[EA-NN] Simulation error: {exc}")
            time.sleep(self.speed_ms / 1000.0)

    # ----------------------------------------------------------------------- #
    # Data helpers                                                             #
    # ----------------------------------------------------------------------- #

    def _state(self) -> dict:
        try:
            agents = self.canvas.get_agent_data()
            stats  = self.canvas.get_stats()
            return {"agents": agents, "stats": stats}
        except Exception:
            empty = {"x": [], "y": [], "fitness": [], "ids": []}
            return {"agents": {"herbivores": empty, "carnivores": empty,
                               "food": {"x": [], "y": []}},
                    "stats": {}}

    def _agent_details(self, agent_id) -> dict | None:
        try:
            env = self.canvas.env
            for agent in env.agents:
                if agent.agent_id != agent_id or not agent.is_alive:
                    continue
                info  = agent.get_info()
                brain = agent.brain

                last_in  = getattr(agent, "_last_inputs",  np.zeros(19))
                last_out = getattr(agent, "_last_outputs", np.full(4, 0.5))

                info["neural"] = {
                    "input_size"   : int(brain.config.input_size),
                    "hidden_size"  : int(brain.hidden_size),
                    "output_size"  : int(brain.config.output_size),
                    "has_recurrent": bool(brain.has_recurrent),
                    "h_state"      : brain.h_state.tolist(),
                    "W1"           : brain.W1.tolist(),
                    "W2"           : brain.W2.tolist(),
                    "last_inputs"  : np.asarray(last_in).tolist(),
                    "last_outputs" : np.asarray(last_out).tolist(),
                    "input_labels" : [
                        "Energy", "Age", "Can Repro",
                        "Food1 Dist", "Food1 Angle",
                        "Food2 Dist", "Food2 Angle",
                        "Food3 Dist", "Food3 Angle",
                        "Target1 Dist", "Target1 Angle",
                        "Target2 Dist", "Target2 Angle",
                        "Target3 Dist", "Target3 Angle",
                        "X Boundary", "Y Boundary",
                        "Hunger",
                        "Pack1 Dist", "Pack1 Angle",
                        "Pack2 Dist", "Pack2 Angle",
                        "Pack3 Dist", "Pack3 Angle",
                    ],
                    "output_labels": ["Move X", "Move Y", "Speed", "Reproduce"],
                    "decisions"    : int(brain.decisions),
                }
                return info
        except Exception as exc:
            print(f"[EA-NN] Inspector error: {exc}")
        return None

    # ----------------------------------------------------------------------- #
    # Start                                                                    #
    # ----------------------------------------------------------------------- #

    def start_server(self, host: str = "localhost", port: int = 5000,
                     debug: bool = False):
        print(f"[EA-NN]  http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug,
                          allow_unsafe_werkzeug=True)


# =========================================================================== #
# HTML / CSS / JS template                                                     #
# =========================================================================== #

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EA-NN Neural Evolution</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117; --surface:#161b22; --surface2:#1c2128;
  --border:#30363d; --border2:#21262d;
  --text:#e6edf3; --muted:#7d8590;
  --herb:#3fb950; --herb-dim:#1a4d2e;
  --carn:#f85149; --carn-dim:#4d1a1a;
  --food:#d29922; --food-dim:#4d3800;
  --accent:#58a6ff; --purple:#a371f7;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.5}

/* Layout */
.layout{display:grid;grid-template-rows:52px 1fr;height:100vh}
.main{display:grid;grid-template-columns:1fr 340px;overflow:hidden;min-height:0}

/* Header */
header{
  display:flex;align-items:center;gap:14px;padding:0 20px;
  background:var(--surface);border-bottom:1px solid var(--border);
  user-select:none;flex-shrink:0;
}
.logo{font-size:15px;font-weight:600;letter-spacing:.5px;white-space:nowrap}
.logo em{font-style:normal;color:var(--accent)}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--muted);
  transition:background .3s,box-shadow .3s;flex-shrink:0}
.status-dot.running{background:var(--herb);box-shadow:0 0 7px var(--herb)}
.status-label{font-size:12px;color:var(--muted);white-space:nowrap}
.spacer{flex:1}
.gen-badge{
  font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);
  background:var(--bg);border:1px solid var(--border2);border-radius:5px;
  padding:3px 9px;white-space:nowrap;
}
.speed-wrap{display:flex;align-items:center;gap:8px;white-space:nowrap}
.speed-wrap label{font-size:11px;color:var(--muted)}
input[type=range]{
  -webkit-appearance:none;width:88px;height:3px;cursor:pointer;
  background:var(--border);border-radius:2px;outline:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:13px;height:13px;border-radius:50%;
  background:var(--accent);cursor:pointer;transition:transform .1s;
}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}
.speed-val{font-family:'JetBrains Mono',monospace;font-size:11px;
  color:var(--accent);width:44px;text-align:left}
button{
  display:inline-flex;align-items:center;gap:6px;
  padding:6px 14px;border-radius:6px;border:1px solid var(--border);
  background:var(--surface2);color:var(--text);
  font-size:12px;font-family:'Inter',sans-serif;font-weight:500;
  cursor:pointer;transition:all .15s;white-space:nowrap;
}
button:hover{border-color:var(--accent);color:var(--accent);background:var(--bg)}
button.start{background:var(--herb-dim);border-color:var(--herb);color:var(--herb)}
button.start:hover{background:var(--herb);color:#000}
button.stop{background:var(--carn-dim);border-color:var(--carn);color:var(--carn)}
button.stop:hover{background:var(--carn);color:#fff}

/* Canvas area */
.canvas-wrap{
  display:flex;align-items:center;justify-content:center;
  background:var(--bg);overflow:hidden;padding:12px;
}
#sim{border-radius:8px;display:block;cursor:crosshair}

/* Sidebar */
.sidebar{
  background:var(--surface);border-left:1px solid var(--border);
  overflow-y:auto;display:flex;flex-direction:column;
}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* Cards */
.card{padding:14px 16px;border-bottom:1px solid var(--border2);flex-shrink:0}
.card-title{
  font-size:10px;font-weight:600;letter-spacing:.9px;
  text-transform:uppercase;color:var(--muted);margin-bottom:12px;
}

/* Population grid */
.pop-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:12px}
.pop-cell{
  background:var(--bg);border-radius:7px;border:1px solid var(--border2);
  padding:10px 6px;text-align:center;
}
.pop-num{font-size:26px;font-weight:600;font-family:'JetBrains Mono',monospace;line-height:1.1}
.pop-num.h{color:var(--herb)} .pop-num.c{color:var(--carn)} .pop-num.f{color:var(--food)}
.pop-label{font-size:9px;color:var(--muted);margin-top:3px;letter-spacing:.5px;text-transform:uppercase}

/* Info rows */
.info-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.ik{font-size:11px;color:var(--muted)}
.iv{font-family:'JetBrains Mono',monospace;font-size:12px}

/* Chart */
.chart-wrap{height:120px;position:relative}

/* Inspector */
.empty-state{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;padding:28px 16px;color:var(--muted);font-size:12px;text-align:center;
}
.empty-state svg{opacity:.35}

/* Agent header */
.agent-hdr{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.badge{
  padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600;
  letter-spacing:.5px;text-transform:uppercase;
}
.badge.h{background:var(--herb-dim);color:var(--herb);border:1px solid var(--herb)}
.badge.c{background:var(--carn-dim);color:var(--carn);border:1px solid var(--carn)}
.aid{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted)}

/* Energy bar */
.ebar-hdr{display:flex;justify-content:space-between;margin-bottom:5px}
.ebar-label{font-size:11px;color:var(--muted)}
.ebar-val{font-family:'JetBrains Mono',monospace;font-size:11px}
.ebar-track{height:5px;background:var(--bg);border-radius:3px;overflow:hidden;margin-bottom:13px}
.ebar-fill{height:100%;border-radius:3px;transition:width .4s}

/* Agent stats grid */
.agent-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:13px}
.ag-cell{background:var(--bg);border-radius:6px;padding:8px 10px}
.ag-key{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.ag-val{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500}

/* Sensor bars */
.section-title{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;
  margin-bottom:7px;padding-top:4px}
.sbar-row{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.sbar-name{font-size:9px;color:var(--muted);width:78px;flex-shrink:0;
  overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.sbar-track{flex:1;height:4px;background:var(--bg);border-radius:2px;overflow:hidden}
.sbar-fill{height:100%;border-radius:2px}
.sbar-num{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);
  width:28px;text-align:right;flex-shrink:0}

/* NN canvas */
#nn-canvas{display:block;width:100%;border-radius:6px;background:var(--bg)}

/* Tooltip */
#tip{
  position:fixed;pointer-events:none;display:none;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;padding:5px 10px;font-size:11px;
  font-family:'JetBrains Mono',monospace;color:var(--text);
  z-index:200;box-shadow:0 4px 16px rgba(0,0,0,.5);white-space:nowrap;
}
</style>
</head>
<body>
<div class="layout">

<!-- Header -->
<header>
  <span class="logo">EA&#8209;<em>NN</em></span>
  <div class="status-dot" id="dot"></div>
  <span class="status-label" id="slabel">Stopped</span>
  <div class="spacer"></div>
  <span class="gen-badge" id="gbadge">Gen &mdash; &nbsp;|&nbsp; Step &mdash;</span>
  <div class="speed-wrap">
    <label>Speed</label>
    <input type="range" id="speed" min="0" max="5" step="1" value="3"
           list="speed-ticks"/>
    <datalist id="speed-ticks">
      <option value="0"><option value="1"><option value="2">
      <option value="3"><option value="4"><option value="5">
    </datalist>
    <span class="speed-val" id="sval">10/s</span>
  </div>
  <button class="start" id="btn-start">&#9654;&nbsp;Start</button>
  <button class="stop"  id="btn-stop"  style="display:none">&#9646;&#9646;&nbsp;Stop</button>
</header>

<!-- Main -->
<div class="main">
  <div class="canvas-wrap"><canvas id="sim"></canvas></div>

  <div class="sidebar">

    <!-- Population -->
    <div class="card">
      <div class="card-title">Population</div>
      <div class="pop-grid">
        <div class="pop-cell"><div class="pop-num h" id="nh">0</div><div class="pop-label">Herbivores</div></div>
        <div class="pop-cell"><div class="pop-num c" id="nc">0</div><div class="pop-label">Carnivores</div></div>
        <div class="pop-cell"><div class="pop-num f" id="nf">0</div><div class="pop-label">Food</div></div>
      </div>
      <div class="info-row"><span class="ik">Step</span><span class="iv" id="vstep">0</span></div>
      <div class="info-row"><span class="ik">Generation</span><span class="iv" id="vgen">—</span></div>
      <div class="info-row"><span class="ik">Avg Herb Energy</span><span class="iv" id="vhe">—</span></div>
      <div class="info-row"><span class="ik">Avg Carn Energy</span><span class="iv" id="vce">—</span></div>
    </div>

    <!-- Chart -->
    <div class="card">
      <div class="card-title">Population History</div>
      <div class="chart-wrap"><canvas id="pop-chart"></canvas></div>
    </div>

    <!-- Inspector -->
    <div class="card" style="flex:1;min-height:200px">
      <div class="card-title">Agent Inspector</div>
      <div id="inspector">
        <div class="empty-state">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
          Click an agent to inspect it
        </div>
      </div>
    </div>

  </div>
</div>
</div>
<div id="tip"></div>

<script>
// ── Constants ──────────────────────────────────────────────────────────────
const WORLD = 100;
const C_HERB = '#3fb950', C_CARN = '#f85149', C_FOOD = '#d29922';

// ── Socket ─────────────────────────────────────────────────────────────────
const socket = io();
let lastState  = null;
let selectedId = null;

socket.on('running',      v  => setRunning(v));
socket.on('state',        s  => onState(s));
socket.on('agent_details',d  => renderInspector(d));

// ── Controls ───────────────────────────────────────────────────────────────
document.getElementById('btn-start').onclick = () => socket.emit('start');
document.getElementById('btn-stop') .onclick = () => socket.emit('stop');

const speedSlider = document.getElementById('speed');
const speedLabel  = document.getElementById('sval');

// Discrete speed presets: position → [ms delay, display label]
const SPEED_PRESETS = [
  [1000, '1/s'],   // 0 — very slow, watch individual decisions
  [200,  '5/s'],   // 1
  [100,  '10/s'],  // 2
  [50,   '20/s'],  // 3  ← default
  [20,   '50/s'],  // 4
  [0,    'MAX'],   // 5  — as fast as Python can run
];
function sliderToMs(v){ return SPEED_PRESETS[v][0]; }
function fmtSpeed(v)  { return SPEED_PRESETS[v][1]; }
speedSlider.oninput = () => {
  const v  = parseInt(speedSlider.value);
  const ms = sliderToMs(v);
  speedLabel.textContent = fmtSpeed(v);
  socket.emit('set_speed', {ms});
};
speedLabel.textContent = fmtSpeed(parseInt(speedSlider.value));

function setRunning(v){
  document.getElementById('dot')     .classList.toggle('running', v);
  document.getElementById('slabel')  .textContent = v ? 'Running' : 'Stopped';
  document.getElementById('btn-start').style.display = v ? 'none' : '';
  document.getElementById('btn-stop') .style.display = v ? ''     : 'none';
}

// ── Simulation canvas ──────────────────────────────────────────────────────
const sim = document.getElementById('sim');
const sctx = sim.getContext('2d');
let scale  = 6;

function resizeSim(){
  const wrap = sim.parentElement;
  const sz   = Math.min(wrap.clientWidth - 24, wrap.clientHeight - 24);
  sim.width  = sz;
  sim.height = sz;
  scale = sz / WORLD;
  if (lastState) renderSim(lastState.agents);
}
window.addEventListener('resize', resizeSim);
resizeSim();

function renderSim(data){
  const ctx = sctx, W = sim.width, H = sim.height;
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#161b22'; ctx.lineWidth = 1;
  for(let i = 0; i <= WORLD; i += 10){
    const p = i * scale;
    ctx.beginPath(); ctx.moveTo(p, 0);  ctx.lineTo(p, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p);  ctx.lineTo(W, p); ctx.stroke();
  }

  // Food
  if(data.food){ const xs=data.food.x, ys=data.food.y;
    ctx.fillStyle = C_FOOD;
    for(let i=0;i<xs.length;i++){
      const px=xs[i]*scale, py=ys[i]*scale;
      ctx.fillRect(px-2, py-2, 4, 4);
    }
  }
  drawSpecies(ctx, data.herbivores, C_HERB, false);
  drawSpecies(ctx, data.carnivores, C_CARN, true);
}

function drawSpecies(ctx, data, color, carn){
  if(!data||!data.x) return;
  const {x:xs, y:ys, ids, fitness:fits} = data;
  for(let i=0;i<xs.length;i++){
    const px=xs[i]*scale, py=ys[i]*scale;
    const sel = ids[i] === selectedId;
    const r   = carn ? 5 : 4;

    if(sel){
      ctx.save();
      ctx.shadowColor='#ffffff88'; ctx.shadowBlur=12;
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(px,py,r+5,0,Math.PI*2); ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = color;
    if(carn){
      ctx.beginPath();
      ctx.moveTo(px, py-r-1); ctx.lineTo(px+r+1, py+r); ctx.lineTo(px-r-1, py+r);
      ctx.closePath(); ctx.fill();
    } else {
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI*2); ctx.fill();
    }
  }
}

// Click
sim.addEventListener('click', e => {
  if(!lastState) return;
  const rect=sim.getBoundingClientRect();
  const mx=e.clientX-rect.left, my=e.clientY-rect.top;
  let best=null, bestD=16;
  function check(data){
    if(!data||!data.x) return;
    for(let i=0;i<data.x.length;i++){
      const dx=data.x[i]*scale-mx, dy=data.y[i]*scale-my;
      const d=Math.sqrt(dx*dx+dy*dy);
      if(d<bestD){bestD=d; best=data.ids[i];}
    }
  }
  check(lastState.agents.herbivores);
  check(lastState.agents.carnivores);
  if(best!==null){ selectedId=best; socket.emit('inspect',{id:best}); }
  else           { selectedId=null; renderInspector(null); }
  renderSim(lastState.agents);
});

// Hover tooltip
const tip = document.getElementById('tip');
sim.addEventListener('mousemove', e => {
  if(!lastState){tip.style.display='none';return;}
  const rect=sim.getBoundingClientRect();
  const mx=e.clientX-rect.left, my=e.clientY-rect.top;
  let label=null, bestD=13;
  function check(data, sp){
    if(!data||!data.x) return;
    for(let i=0;i<data.x.length;i++){
      const dx=data.x[i]*scale-mx, dy=data.y[i]*scale-my;
      const d=Math.sqrt(dx*dx+dy*dy);
      if(d<bestD){bestD=d; label=`${sp} #${data.ids[i]}  fit:${data.fitness[i].toFixed(1)}`;}
    }
  }
  check(lastState.agents.herbivores,'Herb');
  check(lastState.agents.carnivores,'Carn');
  if(label){
    tip.textContent=label; tip.style.display='block';
    tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-8)+'px';
  } else tip.style.display='none';
});
sim.addEventListener('mouseleave',()=>{ tip.style.display='none'; });

// ── Population chart ───────────────────────────────────────────────────────
const chartCtx = document.getElementById('pop-chart').getContext('2d');
const N = 120;
const herbH = new Array(N).fill(null);
const carnH = new Array(N).fill(null);

const popChart = new Chart(chartCtx,{
  type:'line',
  data:{
    labels: herbH.map(()=>''),
    datasets:[
      {label:'Herb',data:[...herbH],borderColor:C_HERB,borderWidth:1.5,
       pointRadius:0,tension:.3,fill:true,backgroundColor:'rgba(63,185,80,.07)'},
      {label:'Carn',data:[...carnH],borderColor:C_CARN,borderWidth:1.5,
       pointRadius:0,tension:.3,fill:true,backgroundColor:'rgba(248,81,73,.07)'},
    ]
  },
  options:{
    responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{display:false},
      y:{min:0,grid:{color:'#21262d'},
         ticks:{color:'#7d8590',font:{size:10},maxTicksLimit:4},
         border:{color:'#30363d'}},
    }
  }
});

function pushChart(h,c){
  herbH.push(h); herbH.shift();
  carnH.push(c); carnH.shift();
  popChart.data.datasets[0].data=[...herbH];
  popChart.data.datasets[1].data=[...carnH];
  popChart.update('none');
}

// ── State handler ──────────────────────────────────────────────────────────
function onState(s){
  lastState = s;
  renderSim(s.agents);
  const st = s.stats||{};
  const nh = st.herbivores||0, nc = st.carnivores||0, nf = st.available_food||0;
  document.getElementById('nh').textContent    = nh;
  document.getElementById('nc').textContent    = nc;
  document.getElementById('nf').textContent    = nf;
  document.getElementById('vstep').textContent = (st.time_step||0).toLocaleString();
  document.getElementById('vgen').textContent  = st.generation??'—';
  document.getElementById('vhe').textContent   = st.avg_herbivore_energy!=null ? st.avg_herbivore_energy.toFixed(1) : '—';
  document.getElementById('vce').textContent   = st.avg_carnivore_energy!=null ? st.avg_carnivore_energy.toFixed(1) : '—';
  document.getElementById('gbadge').textContent = `Gen ${st.generation??'—'}  |  Step ${(st.time_step||0).toLocaleString()}`;
  pushChart(nh, nc);
  if(selectedId!==null) socket.emit('inspect',{id:selectedId});
}

// ── Inspector ──────────────────────────────────────────────────────────────
function renderInspector(d){
  const el = document.getElementById('inspector');
  if(!d){
    el.innerHTML=`<div class="empty-state">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>Click an agent to inspect it</div>`;
    return;
  }
  const isH = d.species==='herbivore';
  const maxE = isH ? 150 : 200;
  const pct  = Math.min(100,(d.energy/maxE)*100);
  const ePct = pct.toFixed(0);
  const eCol = pct>60?'#3fb950':pct>25?'#d29922':'#f85149';
  const nn   = d.neural||{};
  const dead = !d.is_alive;

  el.innerHTML = `
    <div class="agent-hdr">
      <span class="badge ${isH?'h':'c'}">${d.species}</span>
      <span class="aid">#${d.id}${dead?' &mdash; Dead':''}</span>
    </div>
    <div class="ebar-hdr">
      <span class="ebar-label">Energy</span>
      <span class="ebar-val" style="color:${eCol}">${d.energy.toFixed(1)}</span>
    </div>
    <div class="ebar-track"><div class="ebar-fill" style="width:${ePct}%;background:${eCol}"></div></div>
    <div class="agent-grid">
      <div class="ag-cell"><div class="ag-key">Age</div><div class="ag-val">${d.age}</div></div>
      <div class="ag-cell"><div class="ag-key">Fitness</div><div class="ag-val" style="color:#58a6ff">${d.fitness.toFixed(1)}</div></div>
      <div class="ag-cell"><div class="ag-key">Offspring</div><div class="ag-val">${d.offspring}</div></div>
      <div class="ag-cell"><div class="ag-key">${isH?'Food':'Prey'}</div><div class="ag-val">${isH?d.food_consumed.toFixed(0):d.prey_captures}</div></div>
      <div class="ag-cell"><div class="ag-key">Hidden</div><div class="ag-val">${nn.hidden_size??'—'}</div></div>
      <div class="ag-cell"><div class="ag-key">Recurrent</div><div class="ag-val">${nn.has_recurrent?'Yes':'No'}</div></div>
    </div>
    ${sensorBars(nn)}
    <div class="section-title">Network Diagram</div>
    <canvas id="nn-canvas" height="240"></canvas>`;

  requestAnimationFrame(()=>drawNN(nn));
}

function sensorBars(nn){
  const ins  = nn.last_inputs  || [];
  const outs = nn.last_outputs || [];
  const hs   = nn.h_state      || [];
  const iL   = nn.input_labels  || [];
  const oL   = nn.output_labels || [];

  function bar(v,color){
    const pct=Math.min(100,Math.max(0,v*100)).toFixed(0);
    return `<div class="sbar-track"><div class="sbar-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="sbar-num">${v.toFixed(2)}</span>`;
  }
  const inRows  = ins.map((v,i)=>`<div class="sbar-row"><span class="sbar-name">${iL[i]||i}</span>${bar(v,'#58a6ff')}</div>`).join('');
  const hidRows = hs.map((v,i)=>{ const nv=(v+1)/2;
    return `<div class="sbar-row"><span class="sbar-name">h${i}</span>${bar(nv,'#a371f7')}</div>`;
  }).join('');
  const outRows = outs.map((v,i)=>`<div class="sbar-row"><span class="sbar-name">${oL[i]||i}</span>${bar(v,'#3fb950')}</div>`).join('');

  return `<div class="section-title">Sensor Inputs</div>${inRows}
          <div class="section-title">Hidden State</div>${hidRows}
          <div class="section-title">Outputs</div>${outRows}`;
}

// ── Neural network diagram ─────────────────────────────────────────────────
function drawNN(nn){
  const cv = document.getElementById('nn-canvas');
  if(!cv) return;
  cv.width  = cv.offsetWidth || 308;
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height;
  const W1=nn.W1||[], W2=nn.W2||[];
  const ni=nn.input_size||19, nh=nn.hidden_size||12, no=nn.output_size||4;
  const ins  = nn.last_inputs  || new Array(ni).fill(0.5);
  const hs   = nn.h_state      || new Array(nh).fill(0);
  const outs = nn.last_outputs || new Array(no).fill(0.5);
  const oL   = nn.output_labels || ['MX','MY','Spd','Rep'];

  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const PAD=14, r=3.5;
  const cx = [PAD+6, W/2, W-PAD-6];
  function ny(i,n){ const u=H-PAD*2; return PAD + (n>1 ? i*(u/(n-1)) : u/2); }

  // Layer labels
  ctx.fillStyle='#7d8590'; ctx.font='9px Inter'; ctx.textAlign='center';
  ctx.fillText('INPUT', cx[0], H-3);
  ctx.fillText('HIDDEN',cx[1], H-3);
  ctx.fillText('OUTPUT',cx[2], H-3);

  // W1 edges — top by magnitude
  const e1=[];
  for(let i=0;i<ni;i++) for(let j=0;j<nh;j++) if(W1[i]&&W1[i][j]!=null) e1.push([Math.abs(W1[i][j]),i,j,W1[i][j]]);
  e1.sort((a,b)=>b[0]-a[0]);
  for(const [mag,i,j,w] of e1.slice(0,55)){
    const a=Math.min(.5,mag/2), col=w>0?`rgba(88,166,255,${a})`:`rgba(248,81,73,${a})`;
    ctx.strokeStyle=col; ctx.lineWidth=Math.min(1.2,mag*.35);
    ctx.beginPath(); ctx.moveTo(cx[0],ny(i,ni)); ctx.lineTo(cx[1],ny(j,nh)); ctx.stroke();
  }

  // W2 edges
  const e2=[];
  for(let i=0;i<nh;i++) for(let j=0;j<no;j++) if(W2[i]&&W2[i][j]!=null) e2.push([Math.abs(W2[i][j]),i,j,W2[i][j]]);
  e2.sort((a,b)=>b[0]-a[0]);
  for(const [mag,i,j,w] of e2.slice(0,40)){
    const a=Math.min(.65,mag/2), col=w>0?`rgba(88,166,255,${a})`:`rgba(248,81,73,${a})`;
    ctx.strokeStyle=col; ctx.lineWidth=Math.min(1.5,mag*.45);
    ctx.beginPath(); ctx.moveTo(cx[1],ny(i,nh)); ctx.lineTo(cx[2],ny(j,no)); ctx.stroke();
  }

  // Input nodes
  for(let i=0;i<ni;i++){
    const v=ins[i]||0, y=ny(i,ni);
    ctx.fillStyle=`rgba(88,166,255,${.25+v*.75})`;
    ctx.beginPath(); ctx.arc(cx[0],y,r,0,Math.PI*2); ctx.fill();
  }
  // Hidden nodes
  for(let i=0;i<nh;i++){
    const v=(hs[i]+1)/2, y=ny(i,nh);
    ctx.fillStyle=`rgba(163,113,247,${.25+v*.75})`;
    ctx.beginPath(); ctx.arc(cx[1],y,r,0,Math.PI*2); ctx.fill();
    if(nn.has_recurrent){
      ctx.strokeStyle='rgba(163,113,247,.15)'; ctx.lineWidth=.8;
      ctx.beginPath(); ctx.arc(cx[1],y,r+4,0,Math.PI*2); ctx.stroke();
    }
  }
  // Output nodes + labels
  ctx.font='9px Inter'; ctx.textAlign='left';
  for(let i=0;i<no;i++){
    const v=outs[i]||0, y=ny(i,no);
    ctx.fillStyle=`rgba(63,185,80,${.25+v*.75})`;
    ctx.beginPath(); ctx.arc(cx[2],y,r+1,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#7d8590';
    ctx.fillText(oL[i]||i, cx[2]+8, y+3);
  }
}
</script>
</body>
</html>
"""
