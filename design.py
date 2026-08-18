"""Visual/design layer for Message Guard: colors, CSS, and the reusable
Streamlit rendering helpers (hero section, theming, page headers, chart
styling, KPI cards). Kept separate from app.py so styling can be edited
without touching any application logic."""
import json

import streamlit as st
import streamlit.components.v1 as components

CATEGORY_ICONS = {
    "Legitimate Message": "✅",
    "Phishing Attempt": "🎣",
    "Spam / Promotional": "📣",
    "Urgent Action Scam": "⏰",
    "Suspicious Message": "⚠️",
}
CATEGORY_COLORS = {
    "Legitimate Message": "#39ff88",
    "Phishing Attempt": "#ff3b5c",
    "Spam / Promotional": "#ffb020",
    "Urgent Action Scam": "#ff6b3b",
    "Suspicious Message": "#00e5ff",
}


GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    :root {
        --mg-bg: #05070a;
        --mg-surface: #0c1016;
        --mg-border: rgba(0, 229, 255, 0.18);
        --mg-cyan: #00e5ff;
        --mg-orange: #f6821f;
        --mg-text: #d7dee8;
        --mg-text-dim: #7d8b9c;
        --mg-low: #39ff88;
        --mg-medium: #ffb020;
        --mg-high: #ff3b5c;
    }

    html, body, [class*="css"] {
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    }

    /* themed scrollbar so the browser's default (light) scrollbar doesn't
       clash against the dark terminal background */
    html {
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 229, 255, 0.45) #05070a;
    }
    html::-webkit-scrollbar,
    body::-webkit-scrollbar,
    *::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    html::-webkit-scrollbar-track,
    body::-webkit-scrollbar-track,
    *::-webkit-scrollbar-track {
        background: #0a0e14;
    }
    html::-webkit-scrollbar-thumb,
    body::-webkit-scrollbar-thumb,
    *::-webkit-scrollbar-thumb {
        background: rgba(0, 229, 255, 0.45);
        border-radius: 2px;
        border: 2px solid #0a0e14;
    }
    html::-webkit-scrollbar-thumb:hover,
    body::-webkit-scrollbar-thumb:hover,
    *::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 229, 255, 0.7);
    }

    .stApp {
        background-color: var(--mg-bg);
    }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    hr {
        border: none;
        border-top: 1px solid var(--mg-border);
        margin: 2rem 0;
    }

    /* fade-in-up animation applied to page sections as they render */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cf-fade {
        animation: fadeInUp 0.5s ease-out both;
    }

    /* Consistent terminal-style page header used across every inner page */
    .mg-page-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .mg-page-header .mg-prompt {
        color: var(--mg-cyan);
        font-weight: 700;
        font-size: 1.6rem;
    }
    .mg-page-header .mg-title {
        color: var(--mg-text);
        font-weight: 800;
        font-size: 1.6rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .mg-page-subtitle {
        color: var(--mg-text-dim);
        font-size: 0.9rem;
        margin: 0.15rem 0 1.6rem 0;
        border-left: 2px solid var(--mg-border);
        padding-left: 0.6rem;
    }

    /* Get Started / primary buttons: sharp, neon-edged, animated on hover */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f6821f, #ff9d3d);
        border: 1px solid rgba(255, 157, 61, 0.6);
        border-radius: 2px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.0rem;
        letter-spacing: 0.03em;
        padding: 0.75rem 1.6rem;
        box-shadow: 0 0 16px rgba(246, 130, 31, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 26px rgba(246, 130, 31, 0.55);
        background: linear-gradient(135deg, #e0740f, #f6821f);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px);
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 2px;
        border: 1px solid var(--mg-border);
        background: var(--mg-surface);
        color: var(--mg-text);
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--mg-cyan);
        color: var(--mg-cyan);
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
    }

    div[data-testid="stMetric"] {
        background: var(--mg-surface);
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        padding: 0.9rem 1rem;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--mg-cyan);
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.18);
    }
    div[data-testid="stMetricValue"] {
        font-family: "JetBrains Mono", monospace;
    }

    /* Generic terminal-panel card, used for the analysis verdict etc. */
    .mg-terminal-card {
        background: var(--mg-surface);
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }
    .mg-terminal-card-bar {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 0.8rem;
        background: rgba(255,255,255,0.02);
        border-bottom: 1px solid var(--mg-border);
    }
    .mg-terminal-card-bar span.dot {
        width: 9px; height: 9px; border-radius: 50%;
        display: inline-block;
    }
    .mg-terminal-card-bar .dot.r { background: #ff5f56; }
    .mg-terminal-card-bar .dot.y { background: #ffbd2e; }
    .mg-terminal-card-bar .dot.g { background: #27c93f; }
    .mg-terminal-card-bar .label {
        color: var(--mg-text-dim);
        font-size: 0.78rem;
        margin-left: 0.4rem;
        letter-spacing: 0.03em;
    }
    .mg-terminal-card-body {
        padding: 1.4rem 1.5rem;
    }

    /* Circular risk-score gauge (pure CSS conic-gradient ring) */
    .mg-gauge {
        width: 108px;
        height: 108px;
        border-radius: 50%;
        background: conic-gradient(var(--gauge-color) calc(var(--gauge-pct) * 1%), rgba(255,255,255,0.08) 0);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 18px var(--gauge-glow);
    }
    .mg-gauge-inner {
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: var(--mg-surface);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .mg-gauge-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f1f4f8;
        line-height: 1;
    }
    .mg-gauge-label {
        font-size: 0.68rem;
        color: var(--mg-text-dim);
        margin-top: 2px;
    }

    /* small inline tags used for detected keywords / URLs / signals */
    .mg-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 5px 2px 0;
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        font-size: 0.72rem;
        color: var(--mg-cyan);
        background: rgba(0, 229, 255, 0.06);
    }
    .mg-badge.danger {
        color: var(--mg-high);
        border-color: rgba(255, 59, 92, 0.4);
        background: rgba(255, 59, 92, 0.08);
    }

    /* compact panel used in the dashboard-style Analyze grid */
    .mg-panel-title {
        color: var(--mg-text-dim);
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* give native Plotly / dataframe widgets the same bordered card look
       as the pure-HTML terminal cards, so charts and tables match the
       Analyze page's visual language */
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        background: var(--mg-surface);
        padding: 0.5rem;
    }

    /* compact KPI stat card (value + label, no dot-bar) */
    .mg-stat-card {
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        background: var(--mg-surface);
        padding: 0.9rem 1rem;
        height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .mg-stat-card .value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .mg-stat-card .label {
        color: var(--mg-text-dim);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.35rem;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Hero section: dark cybersecurity background + mouse-triggered glitch title
# Rendered as a self-contained HTML component (own CSS + JS, no Streamlit
# rerun involved) so the scramble animation is instant and client-side only.
# The hero now fills the entire viewport (full-bleed, edge-to-edge, no
# rounded corners) instead of sitting in a small centered card.
# ---------------------------------------------------------------------------
def render_hero(title: str = "EMAIL DETECTION", dark: bool = True, component_height: int = 900) -> None:
    # Theme tokens: plain white surface in light mode, plain black in dark
    # mode (no gradient wash) so the toggle reads clearly as two states.
    if dark:
        bg = "#000000"
        grid_line = "rgba(255, 255, 255, 0.07)"
        glyph_color = "rgba(255, 255, 255, 0.16)"
        glyph_flash = "rgba(246, 130, 31, 0.55)"
        title_color = "#f5f7fa"
        title_glow = "rgba(56, 189, 248, 0.25)"
        sub_color = "#aab3c2"
        icon_glow = "rgba(56, 189, 248, 0.45)"
    else:
        bg = "#ffffff"
        grid_line = "rgba(15, 23, 42, 0.06)"
        glyph_color = "rgba(15, 23, 42, 0.10)"
        glyph_flash = "rgba(246, 130, 31, 0.65)"
        title_color = "#14181f"
        title_glow = "rgba(246, 130, 31, 0.12)"
        sub_color = "#5b6472"
        icon_glow = "rgba(246, 130, 31, 0.35)"

    html = f"""
    <div class="hero-wrap">
      <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
        }}
        .hero-wrap {{
            position: relative;
            width: 100%;
            height: 100vh;
            min-height: 100%;
            border-radius: 0;
            overflow: hidden;
            background: {bg};
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            animation: heroFadeIn 0.8s ease-out both;
        }}
        @keyframes heroFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* faint circuit / grid overlay for the cybersecurity feel */
        .hero-grid {{
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient({grid_line} 1px, transparent 1px),
                linear-gradient(90deg, {grid_line} 1px, transparent 1px);
            background-size: 32px 32px;
            mask-image: radial-gradient(circle at 50% 40%, black 0%, transparent 75%);
        }}

        /* background "garbled code" layer: a grid of monospace glyphs.
           They idle on "/" and only scramble to other characters where
           the cursor has passed, like a glitch trail, then settle back
           down to "/" again once left alone. */
        .hero-glyphs {{
            position: absolute;
            inset: 0;
            display: grid;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 12px;
            line-height: 1;
            color: {glyph_color};
            user-select: none;
            pointer-events: none;
            mask-image: radial-gradient(circle at 50% 45%, transparent 0%, transparent 28%, black 60%, black 100%);
        }}
        .hero-glyphs span {{
            transition: color 1.1s ease-out;
        }}
        .hero-glyphs span.flash {{
            color: {glyph_flash};
            transition: color 0.05s ease-out;
        }}

        .hero-inner {{
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 2.5rem 1.25rem;
            max-width: 720px;
        }}

        .hero-icon {{
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 10px {icon_glow});
        }}

        .glitch-title {{
            font-size: clamp(2.1rem, 6vw, 3.6rem);
            font-weight: 800;
            letter-spacing: 0.06em;
            color: {title_color};
            margin: 0 0 0.9rem 0;
            cursor: default;
            text-shadow: 0 0 18px {title_glow};
            user-select: none;
        }}
        .glitch-title span.char {{
            display: inline-block;
            min-width: 0.15em;
        }}

        .hero-sub {{
            font-size: clamp(0.92rem, 2vw, 1.05rem);
            color: {sub_color};
            line-height: 1.6;
            margin: 0 auto;
        }}

        /* "Get Started" is a real link baked into the hero markup, styled
           to match the app's primary-button look, so it reads as one
           integrated piece with the background rather than a separate
           Streamlit widget sitting below it. */
        .hero-cta {{
            display: inline-block;
            margin-top: 2rem;
            background: linear-gradient(135deg, #f6821f, #ff9d3d);
            color: #ffffff;
            font-weight: 700;
            font-size: 1.05rem;
            text-decoration: none;
            padding: 0.85rem 2.3rem;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(246, 130, 31, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
            cursor: pointer;
        }}
        .hero-cta:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 24px rgba(246, 130, 31, 0.45);
            background: linear-gradient(135deg, #e0740f, #f6821f);
        }}
        .hero-cta:active {{
            transform: translateY(0) scale(0.98);
        }}

        @media (max-width: 640px) {{
            .hero-inner {{ padding: 1.75rem 1rem; }}
        }}
      </style>

      <div class="hero-grid"></div>
      <div class="hero-glyphs" id="heroGlyphs"></div>
      <div class="hero-inner">
        <div class="hero-icon">🛡️</div>
        <h1 class="glitch-title" id="glitchTitle"></h1>
        <p class="hero-sub">
          AI-powered spam and phishing detection system that analyses emails and
          messages using NLP and Machine Learning.
        </p>
        <a href="#" class="hero-cta" onclick="goToApp(); return false;">Get Started →</a>
      </div>
    </div>

    <script>
      // Navigates the *top* Streamlit page (not this component iframe) to
      // "?start=1". Built explicitly off window.parent.location rather than
      // a plain relative href, since relative URLs inside a srcdoc iframe
      // don't reliably resolve against the parent page.
      function goToApp() {{
        try {{
          const parentLoc = window.parent.location;
          const base = parentLoc.origin + parentLoc.pathname;
          parentLoc.href = base + "?start=1";
        }} catch (e) {{
          window.location.href = "?start=1";
        }}
      }}

      (function() {{
        const target = {json.dumps(title)};
        const el = document.getElementById('glitchTitle');
        const glitchChars = "!<>-_\\\\/[]{{}}—=+*^?#$%&0123456789";
        let frame = null;
        let running = false;

        function buildSpans(text) {{
            el.innerHTML = "";
            for (const ch of text) {{
                const span = document.createElement('span');
                span.className = 'char';
                span.textContent = ch === ' ' ? '\\u00A0' : ch;
                el.appendChild(span);
            }}
        }}

        function randomChar() {{
            return glitchChars[Math.floor(Math.random() * glitchChars.length)];
        }}

        function playScramble() {{
            if (running) return;
            running = true;
            const spans = Array.from(el.querySelectorAll('.char'));
            const total = spans.length;
            const revealDelayPerChar = 55; // ms between each letter locking in
            let startTime = performance.now();

            function tick(now) {{
                const elapsed = now - startTime;
                const revealCount = Math.min(total, Math.floor(elapsed / revealDelayPerChar));

                for (let i = 0; i < total; i++) {{
                    const original = target[i] === ' ' ? '\\u00A0' : target[i];
                    if (i < revealCount) {{
                        spans[i].textContent = original;
                    }} else if (original === '\\u00A0') {{
                        spans[i].textContent = original;
                    }} else {{
                        spans[i].textContent = randomChar();
                    }}
                }}

                if (revealCount < total) {{
                    frame = requestAnimationFrame(tick);
                }} else {{
                    running = false;
                }}
            }}

            frame = requestAnimationFrame(tick);
        }}

        function resetTitle() {{
            if (frame) cancelAnimationFrame(frame);
            running = false;
            buildSpans(target);
        }}

        buildSpans(target);
        el.addEventListener('mouseenter', playScramble);
        el.addEventListener('mouseleave', resetTitle);

        // ---- background glyph noise, reacts only to the cursor ----
        // Every cell idles on "/". Moving the mouse over the hero makes
        // nearby cells flicker through random characters (a glitch
        // trail); each touched cell keeps flickering on its own for a
        // little while afterwards, gradually slowing down, before
        // settling back to "/" again — even if the mouse has already
        // moved on or left.
        (function() {{
            const wrap = document.querySelector('.hero-wrap');
            const layer = document.getElementById('heroGlyphs');
            const cell = 22; // px per glyph cell
            const idleChar = "/";
            const noiseChars = "01AXF$#%&*<>/\\\\{{}}[]=+;:";
            let cols = 0, rows = 0;

            function buildGrid() {{
                cols = Math.ceil(wrap.clientWidth / cell);
                rows = Math.ceil(wrap.clientHeight / cell);
                layer.style.gridTemplateColumns = `repeat(${{cols}}, ${{cell}}px)`;
                layer.style.gridTemplateRows = `repeat(${{rows}}, ${{cell}}px)`;
                layer.innerHTML = "";
                const total = cols * rows;
                for (let i = 0; i < total; i++) {{
                    const span = document.createElement('span');
                    span.textContent = idleChar;
                    span.style.textAlign = 'center';
                    layer.appendChild(span);
                }}
            }}

            // Kicks off (or restarts) a decaying flicker on a single glyph
            // cell: it rapidly cycles through random characters, gradually
            // slowing down, then locks back to the idle "/" character.
            function triggerGlyph(span) {{
                if (span._glyphTimer) {{
                    clearTimeout(span._glyphTimer);
                }}
                const start = performance.now();
                const duration = 900 + Math.random() * 700; // total settle time

                (function step() {{
                    const elapsed = performance.now() - start;
                    if (elapsed > duration) {{
                        span.textContent = idleChar;
                        span.classList.remove('flash');
                        span._glyphTimer = null;
                        return;
                    }}
                    span.textContent = noiseChars[Math.floor(Math.random() * noiseChars.length)];
                    span.classList.add('flash');
                    setTimeout(() => span.classList.remove('flash'), 80);

                    const progress = elapsed / duration;
                    const nextDelay = 35 + progress * 150; // flicker slows as it settles
                    span._glyphTimer = setTimeout(step, nextDelay);
                }})();
            }}

            buildGrid();
            window.addEventListener('resize', buildGrid);

            let lastMove = 0;
            wrap.addEventListener('mousemove', function(e) {{
                const now = performance.now();
                if (now - lastMove < 35) return; // light throttle
                lastMove = now;

                const rect = wrap.getBoundingClientRect();
                const col = Math.floor((e.clientX - rect.left) / cell);
                const row = Math.floor((e.clientY - rect.top) / cell);
                const radius = 2;

                for (let dr = -radius; dr <= radius; dr++) {{
                    for (let dc = -radius; dc <= radius; dc++) {{
                        const rr = row + dr, cc = col + dc;
                        if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
                        if (Math.sqrt(dr * dr + dc * dc) > radius) continue;
                        if (Math.random() > 0.5) continue; // keep the trail sparse
                        const span = layer.children[rr * cols + cc];
                        if (!span) continue;
                        triggerGlyph(span);
                    }}
                }}
            }});
        }})();
      }})();

      // Resize this iframe to the real browser viewport height. "100vh" CSS
      // inside an iframe only ever refers to the iframe's own fixed height
      // (set below via the `height` prop passed to components.html), not the
      // actual browser window — without this, tall screens are left with
      // empty space below a hero sized to the smaller fallback height.
      (function() {{
        function resizeToViewport() {{
          try {{
            var target = window.parent.innerHeight;
            if (window.frameElement && target && window.frameElement.style.height !== target + 'px') {{
              window.frameElement.style.height = target + 'px';
              // let the glyph-grid's own resize listener recompute its
              // row/column count against the corrected height
              window.dispatchEvent(new Event('resize'));
            }}
          }} catch (e) {{}}
        }}
        resizeToViewport();
        window.addEventListener('resize', resizeToViewport);
      }})();
    </script>
    """
    components.html(html, height=component_height, scrolling=False)


def apply_theme() -> None:
    st.session_state.dark_mode = True
    st.html(
        GLOBAL_CSS
        + """
        <style>
        .stApp { background: #05070a; color: #d7dee8; }
        div[data-testid='stMetric'] { background:#0c1016; border-color: rgba(0, 229, 255, 0.18); }

        /* ---- sidebar shell ---- */
        section[data-testid="stSidebar"] {
            background: #030405;
            border-right: 1px solid rgba(0, 229, 255, 0.10);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        section[data-testid="stSidebar"] hr { border-top: 1px solid rgba(0, 229, 255, 0.10); }

        /* pin the footer status card to the bottom of the sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .mg-sidebar-spacer { flex-grow: 1; }

        /* ---- mini terminal titlebar at the top of the sidebar ---- */
        .mg-sidebar-titlebar {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 0.5rem 0.1rem 0.9rem 0.1rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid rgba(0, 229, 255, 0.10);
        }
        .mg-sidebar-titlebar span.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .mg-sidebar-titlebar .dot.r { background: #ff5f56; }
        .mg-sidebar-titlebar .dot.y { background: #ffbd2e; }
        .mg-sidebar-titlebar .dot.g { background: #27c93f; }
        .mg-sidebar-titlebar .brand {
            color: var(--mg-text);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-left: 0.35rem;
        }

        .mg-sidebar-label {
            color: #566373;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.2rem 0 0.5rem 0.1rem;
        }

        /* ---- nav buttons (one real st.button per item, so each can carry
           its own icon/badge and get a proper active-state accent bar —
           a native st.radio can't support that per-option styling) ---- */
        section[data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
            padding: 0.6rem 0.85rem;
            border-radius: 4px;
            font-size: 0.92rem;
            letter-spacing: 0;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            border: 1px solid transparent;
            border-left: 3px solid transparent;
            color: #7d8b9c;
            font-weight: 500;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: rgba(0, 229, 255, 0.06);
            border-left-color: rgba(0, 229, 255, 0.4);
            color: #f5f7fa;
            box-shadow: none;
            transform: none;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(246, 130, 31, 0.14) !important;
            border: 1px solid transparent !important;
            border-left: 3px solid #f6821f !important;
            color: #ffb066 !important;
            font-weight: 700;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: rgba(246, 130, 31, 0.22) !important;
            transform: none;
        }

        /* ---- status footer card, pinned to the bottom via the spacer above ---- */
        .mg-sidebar-footer {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.8rem 0.9rem;
            border: 1px dashed rgba(0, 229, 255, 0.3);
            border-radius: 6px;
            background: rgba(0, 229, 255, 0.03);
            margin: 0.8rem 0 1rem 0;
        }
        .mg-sidebar-footer .pulse {
            width: 8px; height: 8px; border-radius: 50%;
            background: #39ff88;
            box-shadow: 0 0 8px #39ff88;
            flex-shrink: 0;
        }
        .mg-sidebar-footer .meta { line-height: 1.3; }
        .mg-sidebar-footer .meta .model {
            color: var(--mg-text);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .mg-sidebar-footer .meta .status {
            color: #566373;
            font-size: 0.68rem;
            letter-spacing: 0.04em;
        }
        </style>
        """
    )


RISK_COLORS = {"low": "#39ff88", "medium": "#ffb020", "high": "#ff3b5c"}
RISK_ICONS = {"low": "✅", "medium": "⚠️", "high": "🚨"}


def page_header(icon: str, title: str, subtitle: str, extra_style: str = "") -> None:
    """Render the consistent terminal-style header used on every inner page.
    `extra_style` lets a page fold its own scoped <style> tag into this same
    element instead of issuing a separate, otherwise-empty st.markdown call."""
    st.html(
        f"""
        {extra_style}
        <div class="mg-page-header cf-fade">
            <span class="mg-prompt">{icon}</span>
            <span class="mg-title">{title}</span>
        </div>
        <div class="mg-page-subtitle cf-fade">{subtitle}</div>
        """
    )


def style_fig(fig):
    """Apply the dark terminal theme to a Plotly figure so charts blend with the app."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="JetBrains Mono, monospace",
        font_color="#d7dee8",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def stat_card(value, label: str, color: str = "#00e5ff", value_size: str = "1.5rem") -> None:
    """Render a compact KPI stat card matching the Analyze page's card language."""
    st.html(
        f"""
        <div class="mg-stat-card cf-fade">
            <div class="value" style="color:{color}; font-size:{value_size};">{value}</div>
            <div class="label">{label}</div>
        </div>
        """
    )