"""
Dining Philosophers (4 philosophers) - Monitor solution with condition variables.

Just run:   python dining_philosophers_simulation.py

1. The real monitor runs with 4 threads and prints the output in the console.
2. When it finishes, an animated replay page is created and opened in your
   browser automatically (no Tkinter or extra libraries needed).
"""
import json
import os
import threading
import time
import webbrowser

N = 4          # number of philosophers
MEALS = 5      # meals each philosopher eats
SPEED = float(os.environ.get("DP_SPEED", "1"))   # 1 = normal timing

THINKING = "THINKING"
HUNGRY = "HUNGRY"
EATING = "EATING"

T0 = time.monotonic()
print_lock = threading.Lock()
EVENTS = []   # state changes for the animation
LOGS = []     # console lines for the animation's output panel


def now():
    return time.monotonic() - T0


def log(message):
    with print_lock:
        print(message, flush=True)
        LOGS.append({"t": round(now(), 3), "text": message})


class DiningMonitor:
    """Monitor: one lock + one condition variable per philosopher."""

    def __init__(self):
        self.lock = threading.Lock()
        self.condition = [threading.Condition(self.lock) for _ in range(N)]
        self.state = [THINKING] * N
        self.meals = [0] * N
        self.ticket_counter = 0          # fairness: longest-waiting goes first
        self.ticket = [None] * N

    def left(self, i):
        return (i + N - 1) % N

    def right(self, i):
        return (i + 1) % N

    def record(self, i):
        EVENTS.append({"t": round(now(), 3), "i": i,
                       "s": self.state[i], "m": self.meals[i]})

    def older_neighbour_waiting(self, i):
        for j in (self.left(i), self.right(i)):
            if self.state[j] == HUNGRY and self.ticket[j] < self.ticket[i]:
                return True
        return False

    def test(self, i):
        # Call with self.lock held.
        if (self.state[i] == HUNGRY
                and self.state[self.left(i)] != EATING
                and self.state[self.right(i)] != EATING
                and not self.older_neighbour_waiting(i)):
            self.state[i] = EATING
            self.ticket[i] = None
            self.record(i)
            log(f"Philosopher {i + 1} starts eating")
            self.condition[i].notify()

    def pickup_forks(self, i):
        with self.lock:
            self.state[i] = HUNGRY
            self.ticket[i] = self.ticket_counter
            self.ticket_counter += 1
            self.record(i)
            log(f"Philosopher {i + 1} is hungry")
            self.test(i)
            while self.state[i] != EATING:      # loop guards spurious wakeups
                self.condition[i].wait()

    def putdown_forks(self, i):
        with self.lock:
            self.state[i] = THINKING
            self.meals[i] += 1
            self.record(i)
            log(f"Philosopher {i + 1} finishes eating ({self.meals[i]}/{MEALS})")
            self.test(self.left(i))
            self.test(self.right(i))
            self.test(self.left(self.left(i)))
            self.test(self.right(self.right(i)))


def philosopher(monitor, i):
    for _ in range(MEALS):
        log(f"Philosopher {i + 1} is thinking")
        time.sleep(1 * SPEED)
        monitor.pickup_forks(i)
        time.sleep(1 * SPEED)                   # eating
        monitor.putdown_forks(i)
    log(f"Philosopher {i + 1} has finished all meals")


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dining Philosophers - Monitor Solution</title>
<style>
 body{font-family:Arial,sans-serif;background:#fff;margin:0;padding:12px;text-align:center}
 h2{margin:4px 0 8px}
 #wrap{display:flex;flex-wrap:wrap;justify-content:center;gap:16px}
 canvas{border:1px solid #ddd;background:#fff}
 #log{width:340px;height:560px;overflow-y:auto;text-align:left;background:#111;color:#9f9;
      font:12px/1.5 Consolas,monospace;padding:8px;box-sizing:border-box;border-radius:4px}
 #status{font-weight:bold;font-size:16px;margin:8px}
 button,select{font-size:14px;margin:2px;padding:4px 10px}
</style></head><body>
<h2>Dining Philosophers - Monitor Solution (4 philosophers)</h2>
<div id="wrap">
 <canvas id="cv" width="650" height="560"></canvas>
 <div id="log"></div>
</div>
<div id="status"></div>
<div>
 <button id="pp">Pause</button><button id="rs">Restart</button>
 Speed: <select id="sp"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
 <option value="2">2x</option><option value="4">4x</option></select>
</div>
<script>
const N=__N__, MEALS=__MEALS__, EVENTS=__EVENTS__, LOGS=__LOGS__;
const DUR=Math.max(EVENTS.length?EVENTS[EVENTS.length-1].t:0, LOGS.length?LOGS[LOGS.length-1].t:0);
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const logEl=document.getElementById('log'), statusEl=document.getElementById('status');
let T=0, playing=true, speed=1, last=null, shownLogs=0;
const COL={THINKING:'#87CEEB',HUNGRY:'#FFD700',EATING:'#90EE90'};

function stateAt(t){
  const s=Array(N).fill('THINKING'), m=Array(N).fill(0);
  for(const e of EVENTS){ if(e.t>t)break; s[e.i]=e.s; m[e.i]=e.m; }
  return [s,m];
}
function fork(cx,cy,a,color,w){
  const x1=cx+120*Math.cos(a), y1=cy+120*Math.sin(a), x2=cx+165*Math.cos(a), y2=cy+165*Math.sin(a);
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=w; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  ctx.beginPath(); ctx.arc(x2,y2,5,0,7); ctx.fill();
}
function text(t,x,y,font,color){ctx.font=font;ctx.fillStyle=color||'#000';ctx.textAlign='center';ctx.fillText(t,x,y);}
function render(){
  const [s,m]=stateAt(T), cx=325, cy=270, step=2*Math.PI/N;
  ctx.clearRect(0,0,650,560);
  ctx.beginPath(); ctx.arc(cx,cy,115,0,7); ctx.fillStyle='#D2B48C'; ctx.fill();
  ctx.lineWidth=3; ctx.strokeStyle='#795548'; ctx.stroke();
  text('DINING',cx,cy-4,'bold 16px Arial'); text('TABLE',cx,cy+16,'bold 16px Arial');
  for(let k=0;k<N;k++){
    const nx=(k+1)%N;
    if(s[k]==='EATING') fork(cx,cy,-Math.PI/2+k*step+0.2,'#228B22',5);
    else if(s[nx]==='EATING') fork(cx,cy,-Math.PI/2+nx*step-0.2,'#228B22',5);
    else fork(cx,cy,-Math.PI/2+(k+0.5)*step,'#555555',4);
  }
  for(let i=0;i<N;i++){
    const a=-Math.PI/2+i*step, px=cx+205*Math.cos(a), py=cy+205*Math.sin(a);
    ctx.beginPath(); ctx.ellipse(px,py,48,35,0,0,7); ctx.fillStyle=COL[s[i]]; ctx.fill();
    ctx.lineWidth=2; ctx.strokeStyle='#000'; ctx.stroke();
    text('P'+(i+1),px,py-4,'bold 12px Arial'); text(s[i],px,py+14,'9px Arial');
    text('Meals: '+m[i]+'/'+MEALS,px,py+56,'11px Arial');
  }
  text('BLUE = THINKING    YELLOW = HUNGRY    GREEN = EATING',325,510,'bold 11px Arial');
  text('Green forks = in use    Grey forks = on the table',325,535,'10px Arial');
  // output panel
  let n=0; while(n<LOGS.length && LOGS[n].t<=T) n++;
  if(n<shownLogs){logEl.textContent='';shownLogs=0;}
  for(;shownLogs<n;shownLogs++){
    const d=document.createElement('div'); d.textContent=LOGS[shownLogs].text; logEl.appendChild(d);
  }
  logEl.scrollTop=logEl.scrollHeight;
  const done=m.every(x=>x>=MEALS);
  statusEl.textContent=done?'All philosophers finished eating!':'Monitor is controlling access to the forks';
}
function frame(ts){
  if(last===null)last=ts;
  if(playing){T+=(ts-last)/1000*speed; if(T>DUR+0.5){T=DUR+0.5;playing=false;pp.textContent='Play';}}
  last=ts; render(); requestAnimationFrame(frame);
}
const pp=document.getElementById('pp');
pp.onclick=()=>{playing=!playing;pp.textContent=playing?'Pause':'Play';};
document.getElementById('rs').onclick=()=>{T=0;playing=true;pp.textContent='Pause';};
document.getElementById('sp').onchange=e=>{speed=parseFloat(e.target.value);};
requestAnimationFrame(frame);
</script></body></html>
"""


def build_html(path):
    events = sorted(EVENTS, key=lambda e: e["t"])
    logs = sorted(LOGS, key=lambda e: e["t"])
    html = (HTML.replace("__N__", str(N))
                .replace("__MEALS__", str(MEALS))
                .replace("__EVENTS__", json.dumps(events))
                .replace("__LOGS__", json.dumps(logs)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("Starting Dining Philosophers (monitor solution)...\n")
    monitor = DiningMonitor()
    threads = [threading.Thread(target=philosopher, args=(monitor, i),
                                name=f"Philosopher-{i + 1}")
               for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nFinal meal counts:", monitor.meals)
    print("Dining Philosophers simulation completed.")

    path = os.path.abspath("dining_philosophers_animation.html")
    build_html(path)
    print(f"\nAnimation saved to: {path}")
    print("Opening it in your browser... (if it doesn't open, open that file manually)")
    webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()