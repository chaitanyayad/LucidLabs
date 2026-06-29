import streamlit as st
from pathlib import Path

_ICON = Path(__file__).parent / "assets" / "lucidlabs-icon.svg"

st.set_page_config(
    page_title="LucidLabss",
    page_icon=str(_ICON),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Hide all Streamlit chrome on landing */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stSidebarNav"],
[data-testid="stStatusWidget"],
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Background — grid merged into background-image to avoid z-index covering iframe */
.stApp {
    background:
      linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px),
      radial-gradient(ellipse 60% 40% at 15% 0%,   rgba(124,58,237,0.12) 0%, transparent 70%),
      radial-gradient(ellipse 50% 35% at 85% 100%, rgba(6,182,212,0.08)  0%, transparent 70%),
      linear-gradient(180deg, #080B14 0%, #0A0F1E 60%, #080B14 100%);
    background-size: 48px 48px, 48px 48px, auto, auto, auto;
    min-height: 100vh;
}
.block-container {
    padding: 1rem 1rem !important;
    max-width: 100% !important;
}
/* CTA button reveal — fades in after content settles (~2s) */
[data-testid="stVerticalBlock"] > div:last-child {
    opacity: 0;
    animation: btnReveal 0.9s ease 2s forwards;
}
@keyframes btnReveal { to { opacity: 1; } }

/* CTA button styles */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6366F1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2.8rem !important;
    font-family: 'Exo 2', 'Segoe UI', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 28px rgba(124,58,237,0.45) !important;
    transition: all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
    min-height: 52px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #4F46E5) !important;
    box-shadow: 0 8px 40px rgba(124,58,237,0.65) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

/* Animated gradient blobs */
@keyframes _b1{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(300px,-200px) scale(1.25)}66%{transform:translate(-180px,260px) scale(.8)}}
@keyframes _b2{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(-240px,180px) scale(1.2)}66%{transform:translate(200px,-250px) scale(1.25)}}
@keyframes _b3{0%,100%{transform:translate(-50%,-50%) scale(1)}33%{transform:translate(calc(-50% + 320px),calc(-50% + 220px)) scale(.75)}66%{transform:translate(calc(-50% - 240px),calc(-50% - 180px)) scale(1.2)}}
.grad-blob{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0;}
.gb1{width:750px;height:650px;background:radial-gradient(circle,rgba(124,58,237,.28) 0%,transparent 65%);top:-200px;left:-200px;animation:_b1 9s ease-in-out infinite;}
.gb2{width:650px;height:550px;background:radial-gradient(circle,rgba(6,182,212,.2) 0%,transparent 65%);bottom:-160px;right:-160px;animation:_b2 11s ease-in-out infinite;}
.gb3{width:550px;height:480px;background:radial-gradient(circle,rgba(99,102,241,.15) 0%,transparent 65%);top:50%;left:50%;transform:translate(-50%,-50%);animation:_b3 13s ease-in-out infinite;}
</style>
<div class="grad-blob gb1"></div>
<div class="grad-blob gb2"></div>
<div class="grad-blob gb3"></div>
""", unsafe_allow_html=True)

LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: transparent;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 24px 16px;
  color: #F1F5F9;
}

/* Hero card */
.hero {
  position: relative;
  text-align: center;
  max-width: 820px;
  width: 100%;
  padding: 2.5rem 3rem 2rem;
  background: rgba(15, 20, 40, 0.85);
  border-radius: 24px;
  border: 1px solid rgba(124, 58, 237, 0.30);
  box-shadow: 0 0 60px rgba(124,58,237,0.15), 0 24px 48px rgba(0,0,0,0.4);
  animation: heroIn 0.6s ease both;
}
@keyframes heroIn {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}

/* Top accent */
.accent-line {
  position: absolute;
  top: 0; left: 12%; right: 12%;
  height: 2px;
  background: linear-gradient(90deg, transparent, #7C3AED 35%, #06B6D4 65%, transparent);
  border-radius: 2px;
}

/* Badge */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(124,58,237,0.14);
  border: 1px solid rgba(124,58,237,0.35);
  color: #A5B4FC;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  padding: 5px 18px;
  border-radius: 999px;
  margin-bottom: 1.5rem;
  animation: fadeUp 0.5s ease 0.2s both;
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #06B6D4;
  box-shadow: 0 0 8px #06B6D4;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

/* Title */
.title {
  font-size: 2.6rem;
  font-weight: 800;
  line-height: 1.2;
  color: #E2D9FF;
  margin-bottom: 1.1rem;
  animation: fadeUp 0.6s ease 0.4s both;
}
.title em {
  font-style: normal;
  color: #A78BFA;
}

/* Typewriter lines */
.typewriter {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  border-right: 3px solid #A78BFA;
  width: max-content;
  max-width: 0;
  animation: typing 1.4s steps(22,end) 0.6s forwards,
             cursor 0.6s step-end 2s 2,
             hideCursor 0.01s linear 3.2s forwards;
}
.typewriter2 {
  display: block;
  color: #C4B5FD;
  white-space: nowrap;
}
#tw2c {
  display: inline-block;
  border-right: 3px solid #A78BFA;
  height: 0.8em;
  vertical-align: middle;
  margin-left: 2px;
  opacity: 0;
}
@keyframes typing    { to { max-width: 50rem; } }
@keyframes cursor    { 50% { border-color: transparent; } }
@keyframes hideCursor { to { border-color: transparent; } }

/* Subtitle */
.subtitle {
  font-size: 0.93rem;
  color: #94A3B8;
  line-height: 1.8;
  margin-bottom: 1.75rem;
  max-width: 580px;
  margin-left: auto; margin-right: auto;
  animation: fadeUp 0.6s ease 1s both;
}

/* Stats */
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: rgba(124,58,237,0.18);
  border: 1px solid rgba(124,58,237,0.22);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 1.5rem;
  animation: fadeUp 0.6s ease 1.2s both;
}
.stat {
  padding: 0.85rem 0.5rem;
  background: rgba(10,14,30,0.75);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.stat-num {
  font-size: 1.55rem;
  font-weight: 800;
  color: #A78BFA;
  line-height: 1;
}
.stat-label {
  font-size: 0.62rem;
  color: #64748B;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  font-weight: 600;
}

/* Chips */
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  justify-content: center;
  margin-bottom: 1.25rem;
  animation: fadeUp 0.6s ease 1.4s both;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(99,102,241,0.09);
  border: 1px solid rgba(99,102,241,0.24);
  border-radius: 7px;
  padding: 4px 12px;
  font-size: 0.77rem;
  font-weight: 500;
  color: #A5B4FC;
}
.chip::before { content:'✓'; color:#06B6D4; font-size:0.68rem; font-weight:800; }

/* Footer */
.footer {
  font-size: 0.68rem;
  color: #334155;
  margin-top: 0.5rem;
  letter-spacing: 0.5px;
  animation: fadeUp 0.5s ease 1.6s both;
}
.footer span { color: #475569; }

@keyframes fadeUp {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}
</style>
</head>
<body>
<div class="hero">
  <div class="accent-line"></div>

  <div class="badge">
    <svg fill="#A78BFA" width="14" height="14" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
      <path d="M27,24a2.9609,2.9609,0,0,0-1.2854.3008L21.4141,20H18v2h2.5859l3.7146,3.7148A2.9665,2.9665,0,0,0,24,27a3,3,0,1,0,3-3Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,28Z"/>
      <path d="M27,13a2.9948,2.9948,0,0,0-2.8157,2H18v2h6.1843A2.9947,2.9947,0,1,0,27,13Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,17Z"/>
      <path d="M27,2a3.0033,3.0033,0,0,0-3,3,2.9657,2.9657,0,0,0,.3481,1.373L20.5957,10H18v2h3.4043l4.3989-4.2524A2.9987,2.9987,0,1,0,27,2Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,6Z"/>
      <path d="M18,6h2V4H18a3.9756,3.9756,0,0,0-3,1.3823A3.9756,3.9756,0,0,0,12,4H11a9.01,9.01,0,0,0-9,9v6a9.01,9.01,0,0,0,9,9h1a3.9756,3.9756,0,0,0,3-1.3823A3.9756,3.9756,0,0,0,18,28h2V26H18a2.0023,2.0023,0,0,1-2-2V8A2.0023,2.0023,0,0,1,18,6ZM12,26H11a7.0047,7.0047,0,0,1-6.92-6H6V18H4V14H7a3.0033,3.0033,0,0,0,3-3V9H8v2a1.0009,1.0009,0,0,1-1,1H4.08A7.0047,7.0047,0,0,1,11,6h1a2.0023,2.0023,0,0,1,2,2v4H12v2h2v4H12a3.0033,3.0033,0,0,0-3,3v2h2V21a1.0009,1.0009,0,0,1,1-1h2v4A2.0023,2.0023,0,0,1,12,26Z"/>
    </svg>
    LucidLabs
  </div>

  <div class="title">
    <span class="typewriter">Learn Machine Learning</span>
    <span class="typewriter2"><span id="tw2t"></span><span id="tw2c"></span></span>
  </div>

  <p class="subtitle">
    Upload your dataset or pick one from the UCI repository.
    Auto-detects your task, runs full EDA, trains your algorithm,
    and explains exactly how it works &mdash; no code required.
  </p>

  <div class="stats">
    <div class="stat"><div class="stat-num" data-target="11">--</div><div class="stat-label">Algorithms</div></div>
    <div class="stat"><div class="stat-num" data-target="8">--</div><div class="stat-label">Datasets</div></div>
    <div class="stat"><div class="stat-num" data-target="8">--</div><div class="stat-label">EDA Tabs</div></div>
    <div class="stat"><div class="stat-num" data-target="6">--</div><div class="stat-label">Steps</div></div>
  </div>

  <div class="chips">
    <span class="chip">Auto Task Detection</span>
    <span class="chip">Full EDA Dashboard</span>
    <span class="chip">UCI Dataset Browser</span>
    <span class="chip">Hyperparameter Tuning</span>
    <span class="chip">Algorithm Explanations</span>
    <span class="chip">Interactive Visualizations</span>
  </div>

  <div class="footer">
    Powered by <span>scikit-learn</span> &middot; <span>Streamlit</span> &middot; <span>Plotly</span>
  </div>
</div>
<script>
// Number scramble effect — runs when stats animate in (~1.2s delay)
function scramble(el) {
  const target = parseInt(el.dataset.target, 10);
  const chars = '0123456789';
  const duration = 900;   // ms total
  const interval = 50;    // ms per frame
  const frames = duration / interval;
  let frame = 0;
  const timer = setInterval(() => {
    frame++;
    const progress = frame / frames;
    if (progress >= 1) {
      el.textContent = target;
      clearInterval(timer);
      return;
    }
    // More random early, locks in near the end
    const lockChance = Math.pow(progress, 2);
    if (Math.random() < lockChance) {
      el.textContent = target;
    } else {
      // Show random digit(s) matching length of target
      const len = String(target).length;
      el.textContent = Array.from({length: len}, () =>
        chars[Math.floor(Math.random() * 10)]
      ).join('');
    }
  }, interval);
}

// Start scramble after the stats fade-in animation begins (1.2s)
setTimeout(() => {
  document.querySelectorAll('.stat-num[data-target]').forEach((el, i) => {
    setTimeout(() => scramble(el), i * 120);
  });
}, 1300);

// ── Typewriter sparkles ────────────────────────────────────────────────────
(function () {
  const css = document.createElement('style');
  css.textContent = `
    @keyframes _spk {
      0%   { transform: translate(0,0) scale(1); opacity: 1; }
      100% { transform: translate(var(--sx), var(--sy)) scale(0); opacity: 0; }
    }
    .spk {
      position: fixed; border-radius: 50%;
      pointer-events: none; z-index: 9999;
      animation: _spk .65s ease-out forwards;
    }
  `;
  document.head.appendChild(css);

  const PAL = ['#A78BFA','#C4B5FD','#06B6D4','#7C3AED','#38BDF8','#ffffff'];

  function burst(x, y) {
    const n = 3;
    for (let i = 0; i < n; i++) {
      const el  = document.createElement('div');
      const sz  = Math.random() * 1.5 + 1.5;
      // fan upward: angles between -60° and +60° from straight up (-90°)
      const ang = (-90 + (Math.random() - 0.5) * 120) * Math.PI / 180;
      const d   = Math.random() * 14 + 8;
      el.className = 'spk';
      el.style.cssText = `
        left:${x + (Math.random()-0.5)*4}px;
        top:${y  + (Math.random()-0.5)*4}px;
        width:${sz}px; height:${sz}px;
        background:${PAL[Math.floor(Math.random()*PAL.length)]};
        box-shadow:0 0 ${sz*1.5}px ${PAL[Math.floor(Math.random()*PAL.length)]};
        --sx:${Math.cos(ang)*d}px; --sy:${Math.sin(ang)*d}px;
      `;
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 650);
    }
  }

  function cursorXY(sel) {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.right, y: r.top + r.height * 0.5 };
  }

  function runPhase(sel, delayMs, durationMs) {
    setTimeout(() => {
      const deadline = Date.now() + durationMs;
      (function tick() {
        const p = cursorXY(sel);
        if (p) burst(p.x, p.y);
        if (Date.now() < deadline) setTimeout(tick, 110);
      })();
    }, delayMs);
  }

  // .typewriter line 1
  runPhase('.typewriter', 600, 1400);

  // .typewriter2 sparkles — track the JS cursor element #tw2c
  setTimeout(() => {
    const deadline = Date.now() + 900;
    (function tick() {
      const cur = document.getElementById('tw2c');
      if (cur) {
        const r = cur.getBoundingClientRect();
        burst(r.left, r.top + r.height * 0.5);
      }
      if (Date.now() < deadline) setTimeout(tick, 110);
    })();
  }, 2100);
})();

// ── Typewriter line 2 — JS driven (avoids CSS slide-in on centered text) ──
setTimeout(() => {
  const txt = 'by Doing It.';
  const el  = document.getElementById('tw2t');
  const cur = document.getElementById('tw2c');
  if (!el || !cur) return;
  cur.style.opacity = '1';
  let i = 0;
  const iv = setInterval(() => {
    el.textContent = txt.slice(0, ++i);
    if (i >= txt.length) {
      clearInterval(iv);
      let b = 0;
      const bv = setInterval(() => {
        cur.style.opacity = cur.style.opacity === '0' ? '1' : '0';
        if (++b >= 6) { clearInterval(bv); cur.style.opacity = '0'; }
      }, 360);
    }
  }, Math.round(900 / txt.length));
}, 2100);
</script>
</body>
</html>
"""

st.components.v1.html(LANDING_HTML, height=600, scrolling=False)

col = st.columns([1, 2, 1])[1]
with col:
    if st.button("Enter Platform  →", type="primary", use_container_width=True):
        st.switch_page("pages/Dashboard.py")
