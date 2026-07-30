"""Build a self-contained HTML replay viewer.

``build_html(replay_json_path, output_html)`` reads a recorded replay and
emits a single ``.html`` file you can open directly in any browser.  The
board is drawn on a canvas with play/pause, step, and a scrubber.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _frame_to_grid(frames: list[dict]) -> tuple[int, int]:
    """Infer grid size from wall_map in the first frame."""
    if not frames:
        return 16, 16
    wall = frames[0]["wall_map"]
    return len(wall), len(wall[0]) if wall else 16


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SnakeGo Replay</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     display:flex;flex-direction:column;align-items:center;padding:16px;min-height:100vh}
h1{font-size:16px;margin-bottom:6px}
.meta{color:#8b949e;font-size:13px;margin-bottom:12px}
.wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
canvas{border:2px solid #30363d;border-radius:4px;background:#010409;image-rendering:pixelated}
.ctrl{display:flex;align-items:center;gap:8px}
button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;
       padding:6px 14px;font-size:13px;cursor:pointer}
button:hover{background:#30363d}
button:disabled{opacity:.4;cursor:default}
input[type=range]{width:280px;accent-color:#58a6ff}
.info{font-size:12px;color:#8b949e;min-height:18px}
.legend{display:flex;gap:14px;font-size:11px;color:#8b949e;margin-top:4px}
.legend span{display:inline-flex;align-items:center;gap:4px}
.dot{width:12px;height:12px;border-radius:2px;display:inline-block}
</style>
</head>
<body>
<h1 id="title">SnakeGo Replay</h1>
<div class="meta" id="meta"></div>
<div class="wrap">
  <canvas id="board" width="480" height="480"></canvas>
  <div class="ctrl">
    <button id="prev">&#9664;&#9664;</button>
    <button id="play">&#9654;</button>
    <button id="next">&#9654;&#9654;</button>
    <input type="range" id="seek" min="0" max="0" value="0">
    <button id="speed">1x</button>
  </div>
  <div class="info" id="info"></div>
  <div class="legend">
    <span><i class="dot" style="background:#3fb950"></i> P0 snake</span>
    <span><i class="dot" style="background:#58a6ff"></i> P1 snake</span>
    <span><i class="dot" style="background:#2ea043"></i> P0 wall</span>
    <span><i class="dot" style="background:#1f6feb"></i> P1 wall</span>
    <span><i class="dot" style="background:#f0883e"></i> item</span>
  </div>
</div>
<script>
const REPLAY = __REPLAY_JSON__;
const L = REPLAY.length, W = REPLAY.width;
const CELL = 28;
const canvas = document.getElementById('board');
canvas.width = W * CELL; canvas.height = L * CELL;
const ctx = canvas.getContext('2d');
const frames = REPLAY.frames;
let idx = 0, playing = false, speed = 1, timer = null;

const CAMPS = {
  0: {snake:'#3fb950', head:'#7ee787', wall:'#2ea043'},
  1: {snake:'#58a6ff', head:'#79c0ff', wall:'#1f6feb'},
};
const ACT_NAMES = {1:'→(+x)',2:'↑(+y)',3:'←(-x)',4:'↓(-y)',5:'RAILGUN',6:'SPLIT'};

function draw(f){
  ctx.fillStyle = '#010409';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  // grid lines
  ctx.strokeStyle = '#161b22'; ctx.lineWidth = 1;
  for(let i=0;i<=L;i++){ctx.beginPath();ctx.moveTo(i*CELL,0);ctx.lineTo(i*CELL,L*CELL);ctx.stroke();}
  for(let j=0;j<=W;j++){ctx.beginPath();ctx.moveTo(0,j*CELL);ctx.lineTo(W*CELL,j*CELL);ctx.stroke();}
  // walls
  for(let x=0;x<L;x++)for(let y=0;y<W;y++){
    const w=f.wall_map[x][y];
    if(w>=0){ctx.fillStyle=CAMPS[w].wall;ctx.fillRect(y*CELL+1,x*CELL+1,CELL-2,CELL-2);}
  }
  // items
  for(let x=0;x<L;x++)for(let y=0;y<W;y++){
    if(f.item_map[x][y]>=0){
      ctx.fillStyle='#f0883e';
      ctx.beginPath();ctx.arc(y*CELL+CELL/2,x*CELL+CELL/2,CELL*0.28,0,Math.PI*2);ctx.fill();
    }
  }
  // snakes
  for(const s of f.snakes){
    const c=CAMPS[s.camp]; if(!c) continue;
    for(let i=0;i<s.coor.length;i++){
      const [x,y]=s.coor[i];
      if(x<0||x>=L||y<0||y>=W) continue;
      ctx.fillStyle = i===0 ? c.head : c.snake;
      ctx.fillRect(y*CELL+2,x*CELL+2,CELL-4,CELL-4);
      if(i===0){
        ctx.strokeStyle='#fff'; ctx.lineWidth=1.5;
        ctx.strokeRect(y*CELL+2,x*CELL+2,CELL-4,CELL-4);
      }
    }
  }
  // info
  const aname = f.action>=1 ? ACT_NAMES[f.action] : '—';
  const pname = f.player===0 ? REPLAY.p0_name : (f.player===1 ? REPLAY.p1_name : '—');
  document.getElementById('info').textContent =
    `step ${f.step}/${frames.length-1}  turn ${f.turn}  ${pname}  ${aname}  → ${f.result}`;
  document.getElementById('seek').value = idx;
}

function go(n){
  idx = Math.max(0, Math.min(frames.length-1, n));
  draw(frames[idx]);
}
function play(){
  playing = !playing;
  document.getElementById('play').textContent = playing ? '❚❚' : '▶';
  if(playing){
    const delay = Math.max(40, 600/speed);
    timer = setInterval(()=>{
      if(idx<frames.length-1) go(idx+1);
      else { playing=false; document.getElementById('play').textContent='▶'; clearInterval(timer); }
    }, delay);
  } else clearInterval(timer);
}

document.getElementById('prev').onclick=()=>go(idx-1);
document.getElementById('next').onclick=()=>go(idx+1);
document.getElementById('play').onclick=play;
document.getElementById('seek').oninput=e=>go(parseInt(e.target.value));
document.getElementById('speed').onclick=function(){
  const speeds=[1,2,4,8,16]; speed=speeds[(speeds.indexOf(speed)+1)%speeds.length];
  this.textContent=speed+'x'; if(playing){clearInterval(timer);play();play();}
};

document.getElementById('title').textContent =
  `SnakeGo: ${REPLAY.p0_name} vs ${REPLAY.p1_name}`;
const winName = REPLAY.winner===0 ? REPLAY.p0_name : (REPLAY.winner===1 ? REPLAY.p1_name : 'draw');
document.getElementById('meta').textContent =
  `seed ${REPLAY.seed}  •  winner: ${winName}  •  score ${REPLAY.scores[0]}-${REPLAY.scores[1]}  •  ${frames.length} steps`;
document.getElementById('seek').max = frames.length-1;
go(0);
</script>
</body>
</html>
"""


def build_html(replay_path: str | Path, output: str | Path | None = None) -> Path:
    """Embed replay JSON into a self-contained HTML file.

    Args:
        replay_path: path to a ``.json`` replay file.
        output: output ``.html`` path (default: same stem, ``.html``).
    """
    replay_path = Path(replay_path)
    with open(replay_path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    frames = data.get("frames", [])
    L, W = _frame_to_grid(frames)
    data["length"] = L
    data["width"] = W

    html = _HTML_TEMPLATE.replace("__REPLAY_JSON__", json.dumps(data))

    if output is None:
        output = replay_path.with_suffix(".html")
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output


__all__ = ["build_html"]
