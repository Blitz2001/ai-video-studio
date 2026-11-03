"""Screen recording: Playwright drives a local 'studio' page (code editor on the
left, the live app in an iframe on the right, test results underneath) and
records it to video. The app really runs, the clicks are real, and the ticks in
the test panel are the real results from the build log."""
from __future__ import annotations

import difflib
import html
import json
import shutil
import subprocess
from pathlib import Path

from ...media import theme
from ...media.anim import ffmpeg_bin
from .browser import apply_action, launch, serve

STUDIO_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>studio</title><style>
:root{{--bg:{bg};--panel:#141413;--line:#2e2e2b;--text:{text};--muted:{muted};--accent:{accent};--hl:{hl};--ok:#22a06b}}
*{{box-sizing:border-box}} html,body{{margin:0;height:100%;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif;overflow:hidden}}
.top{{height:76px;display:flex;align-items:center;gap:18px;padding:0 36px;border-bottom:1px solid var(--line)}}
.badge{{background:var(--accent);font-weight:800;padding:8px 14px;border-radius:8px;letter-spacing:.06em}}
.pill{{color:var(--muted);font-size:24px}} .pill b{{color:var(--text)}}
.meta{{margin-left:auto;color:var(--muted);font:20px ui-monospace,Menlo,monospace}}
.main{{position:absolute;top:92px;left:32px;right:32px;bottom:230px;display:grid;grid-template-columns:1.1fr 1fr;gap:24px}}
.demo .main{{grid-template-columns:1fr}} .demo .editor{{display:none}}
.editor,.browser,.tests{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
.tabs{{display:flex;gap:4px;padding:10px 12px 0;border-bottom:1px solid var(--line)}}
.tab{{padding:8px 16px;font:18px ui-monospace,Menlo,monospace;color:var(--muted);border-radius:8px 8px 0 0}}
.tab.on{{background:var(--bg);color:var(--text)}}
pre{{margin:0;padding:14px 0;font:19px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;height:calc(100% - 50px);overflow:hidden}}
.ln{{display:block;padding:0 16px;white-space:pre}} .ln i{{display:inline-block;width:42px;color:#55554f;font-style:normal;text-align:right;margin-right:18px}}
.ln.new{{background:rgba(57,135,229,.10);box-shadow:inset 3px 0 var(--accent)}} .ln.old{{opacity:.45}}
.k{{color:#c792ea}} .s{{color:#c3e88d}} .c{{color:#6b6b63;font-style:italic}} .t{{color:#82aaff}} .n{{color:#f78c6c}}
.caret{{display:inline-block;width:10px;height:22px;background:var(--hl);vertical-align:middle;animation:b 1s steps(1) infinite}} @keyframes b{{50%{{opacity:0}}}}
.right{{display:grid;grid-template-rows:1fr auto;gap:18px;min-height:0}}
.bar{{height:44px;display:flex;align-items:center;gap:8px;padding:0 14px;border-bottom:1px solid var(--line)}}
.dot{{width:12px;height:12px;border-radius:50%;background:#3a3a37}} .url{{margin-left:10px;color:var(--muted);font:17px ui-monospace,monospace}}
iframe{{border:0;width:100%;height:calc(100% - 44px);background:#1a1a19}}
.tests{{padding:12px 18px;font-size:19px}} .tests h4{{margin:0 0 8px;color:var(--muted);font-weight:600;font-size:16px;letter-spacing:.12em}}
.test{{display:flex;gap:10px;align-items:center;color:var(--muted);margin:3px 0}} .test b{{width:22px;text-align:center}}
.test.ok{{color:var(--text)}} .test.ok b{{color:var(--ok)}} .test.bad b{{color:#e66767}}
#cursor{{position:fixed;left:0;top:0;width:30px;height:30px;pointer-events:none;z-index:9;transition:transform .45s cubic-bezier(.4,0,.2,1)}}
#cursor svg{{filter:drop-shadow(0 2px 3px rgba(0,0,0,.6))}} #cursor.click::after{{content:"";position:absolute;left:-14px;top:-14px;width:28px;height:28px;border-radius:50%;border:3px solid var(--hl);animation:r .4s ease-out}}
@keyframes r{{from{{transform:scale(.3);opacity:1}}to{{transform:scale(1.4);opacity:0}}}}
#key{{position:fixed;right:60px;top:110px;background:var(--hl);color:#111;font-weight:800;font-size:28px;padding:10px 22px;border-radius:12px;opacity:0;transition:opacity .2s;z-index:9}}
@media (max-aspect-ratio:1/1){{.main{{grid-template-columns:1fr;grid-template-rows:1fr 1.25fr;bottom:300px}} .demo .main{{grid-template-rows:1fr}} pre{{font-size:17px}} .tests{{display:none}} .meta{{display:none}}}}
</style></head><body class="{mode}">
<div class="top"><span class="badge">{brand}</span><span class="pill">{pill}</span><span class="meta">{meta}</span></div>
<div class="main">
 <div class="editor"><div class="tabs">{tabs}</div><pre id="code"></pre></div>
 <div class="right">
  <div class="browser"><div class="bar"><span class="dot"></span><span class="dot"></span><span class="dot"></span><span class="url">localhost / focus-timer</span></div>
   <iframe id="app" src="{app_src}"></iframe></div>
  <div class="tests"><h4>BROWSER TESTS</h4>{tests}</div>
 </div>
</div>
<div id="cursor"><svg width="30" height="30" viewBox="0 0 24 24"><path d="M3 2l7 19 2.5-7.5L20 11z" fill="#fff" stroke="#111" stroke-width="1.2"/></svg></div>
<div id="key"></div>
<script>
const LINES = {lines};
const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const TOKEN = /(\/\/.*$|\/\*.*?\*\/|<!--.*?-->)|("[^"]*"|'[^']*'|`[^`]*`)|(<\/?)([a-zA-Z0-9]+)|\b(const|let|function|return|if|else|new|document|this|null|true|false)\b|\b(\d+(?:\.\d+)?)\b/g;
function hl(s){{
  // single pass, so a later rule can never match inside markup an earlier rule produced
  let out = "", last = 0, m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(s)) !== null) {{
    out += esc(s.slice(last, m.index));
    if (m[1]) out += `<span class="c">${{esc(m[1])}}</span>`;
    else if (m[2]) out += `<span class="s">${{esc(m[2])}}</span>`;
    else if (m[4]) out += `${{esc(m[3])}}<span class="t">${{m[4]}}</span>`;
    else if (m[5]) out += `<span class="k">${{m[5]}}</span>`;
    else if (m[6]) out += `<span class="n">${{m[6]}}</span>`;
    last = TOKEN.lastIndex;
    if (m[0].length === 0) TOKEN.lastIndex++;
  }}
  return out + esc(s.slice(last));
}}
const code = document.getElementById("code");
function draw(typed){{
  let out = "", shown = 0;
  for (let i=0;i<LINES.length;i++){{
    const L = LINES[i];
    let text = L.text;
    if (L.isNew){{ if (shown >= typed) {{ if (shown === typed) out += `<span class="ln new"><i>${{i+1}}</i><span class="caret"></span></span>`; shown += 1; continue; }}
      const take = Math.min(text.length, typed - shown); text = text.slice(0, take); shown += L.text.length + 1; }}
    out += `<span class="ln ${{L.isNew?'new':'old'}}"><i>${{i+1}}</i>${{hl(text)}}</span>`;
  }}
  code.innerHTML = out;
  const target = [...code.querySelectorAll(".ln.new")].pop();
  if (target) code.scrollTop = Math.max(0, target.offsetTop - code.clientHeight * 0.55);
}}
const TOTAL = LINES.filter(l=>l.isNew).reduce((a,l)=>a+l.text.length+1,0);
window.typingDone = TOTAL === 0;
draw(TOTAL === 0 ? 0 : 0);
window.startTyping = (ms) => {{
  const t0 = performance.now();
  function f(now){{ const k = Math.min(1,(now-t0)/ms); draw(Math.floor(k*TOTAL)); if (k<1) requestAnimationFrame(f); else {{ draw(TOTAL+1); window.typingDone = true; }} }}
  requestAnimationFrame(f);
}};
window.loadApp = (src) => {{ document.getElementById("app").src = src; }};
const cur = document.getElementById("cursor"); cur.style.transform = "translate(1500px, 700px)";
window.moveCursor = (x,y) => {{ cur.style.transform = `translate(${{x}}px, ${{y}}px)`; }};
window.clickFx = () => {{ cur.classList.remove("click"); void cur.offsetWidth; cur.classList.add("click"); }};
window.showKey = (k) => {{ const e = document.getElementById("key"); e.textContent = k; e.style.opacity = 1; setTimeout(()=>e.style.opacity=0, 700); }};
window.markTest = (i, ok) => {{ const e = document.querySelectorAll(".test")[i]; e.classList.add(ok?"ok":"bad"); e.querySelector("b").textContent = ok?"✓":"✗"; }};
</script></body></html>"""


def code_lines(old: str, new: str) -> list[dict]:
    """Lines of the new file, flagging the ones this step added or changed."""
    old_lines, new_lines = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    flags = [False] * len(new_lines)
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for j in range(j1, j2):
                flags[j] = True
    return [{"text": t, "isNew": f} for t, f in zip(new_lines, flags)]


def studio_page(step: dict | None, snapshot_rel: str, prev_rel: str | None, brand: str, mode: str,
                total_steps: int, index: int) -> str:
    if step is None:  # demo of the finished app
        lines, tabs, tests, pill, meta = [], "", "", "<b>The finished app</b>", ""
    else:
        main = max(step["files"], key=lambda f: f["added"])
        new_text = (Path(snapshot_rel) / main["file"]).read_text() if Path(snapshot_rel).is_absolute() else ""
        lines = code_lines(step["previous"].get(main["file"], ""), new_text)
        tabs = "".join(f'<span class="tab {"on" if f is main else ""}">{html.escape(f["file"])}</span>' for f in step["files"])
        tests = "".join(f'<div class="test"><b>·</b>{html.escape(t["name"])}</div>' for t in step["tests"]["tests"][-5:])
        pill = f"Step {index} of {total_steps} · <b>{html.escape(step['title'])}</b>"
        meta = f"commit {step['commit']} · +{step['added']} −{step['removed']}"
    return STUDIO_HTML.format(bg=theme.SURFACE, text=theme.TEXT, muted=theme.MUTED, accent=theme.ACCENT, hl=theme.HIGHLIGHT,
                              mode=mode, brand=html.escape(brand.upper()), pill=pill, meta=meta, tabs=tabs, tests=tests,
                              app_src=prev_rel or "about:blank", lines=json.dumps(lines))


def _cursor_to(page, frame, selector: str) -> None:
    box = frame.locator(selector).bounding_box()
    if box:
        page.evaluate("([x, y]) => moveCursor(x, y)", [box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.6])
        page.wait_for_timeout(500)
        page.evaluate("clickFx()")


class Screencast:
    """Records a page through Chromium's DevTools screencast and encodes the
    frames with the system ffmpeg at a constant frame rate. This avoids
    Playwright's bundled-ffmpeg requirement, so it works anywhere Chromium runs."""

    def __init__(self, ctx, page, size):
        self.frames: list[tuple[float, bytes]] = []
        self.cdp = ctx.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._on_frame)
        W, H = size
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H,
                                               "everyNthFrame": 1})

    def _on_frame(self, ev):
        import base64

        self.frames.append((ev["metadata"]["timestamp"], base64.b64decode(ev["data"])))
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
        except Exception:
            pass

    def stop(self):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:
            pass

    def encode(self, out_mp4: Path, size, fps: int = 30, min_seconds: float = 0.0) -> Path:
        if not self.frames:
            raise RuntimeError("screencast captured no frames")
        work = out_mp4.parent / f"_frames_{out_mp4.stem}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        t0 = self.frames[0][0]
        lines = []
        for i, (ts, data) in enumerate(self.frames):
            f = work / f"{i:05d}.jpg"
            f.write_bytes(data)
            nxt = self.frames[i + 1][0] if i + 1 < len(self.frames) else max(ts + 0.5, t0 + min_seconds)
            lines += [f"file '{f.name}'", f"duration {max(0.001, nxt - ts):.4f}"]
        lines.append(f"file '{work / f'{len(self.frames) - 1:05d}.jpg'}'".replace(str(work) + "/", ""))
        (work / "list.txt").write_text("\n".join(lines) + "\n")
        W, H = size
        subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                        str(work / "list.txt"), "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", str(out_mp4)], check=True)
        shutil.rmtree(work, ignore_errors=True)
        return out_mp4


def record(run_dir: Path, page_rel: str, out_mp4: Path, size: tuple[int, int], seconds: float,
           app_rel: str, actions: list[dict], test_results: list[dict] | None, typing: bool) -> Path:
    """Record one studio session to mp4."""
    from playwright.sync_api import sync_playwright

    W, H = size
    with serve(run_dir) as base, sync_playwright() as pw:
        browser = launch(pw)
        ctx = browser.new_context(viewport={"width": W, "height": H})
        page = ctx.new_page()
        page.goto(f"{base}/{page_rel}")
        page.wait_for_load_state("networkidle")
        cast = Screencast(ctx, page, size)
        page.wait_for_timeout(300)
        ms_total = int(seconds * 1000)
        type_ms = int(ms_total * 0.42) if typing else 0
        if typing:
            page.evaluate(f"startTyping({type_ms})")
            page.wait_for_timeout(type_ms + 300)
        page.evaluate("src => loadApp(src)", f"{base}/{app_rel}")
        page.wait_for_timeout(700)
        frame = page.frame_locator("#app")
        for action in actions:
            (verb, arg), = action.items()
            if verb == "goto":
                page.evaluate("src => loadApp(src)", f"{base}/{Path(app_rel).parent.as_posix()}/{arg}")
                page.wait_for_timeout(600)
                continue
            if verb == "click":
                _cursor_to(page, frame, arg)
            if verb == "press":
                page.evaluate("k => showKey(k)", "␣ Space" if arg == "Space" else arg)
            apply_action(page, base, action, frame)
            if verb in ("click", "press"):
                page.wait_for_timeout(250)
        if test_results:
            for i, t in enumerate(test_results[-5:]):
                page.evaluate("([i, ok]) => markTest(i, ok)", [i, t["passed"]])
                page.wait_for_timeout(350)
        elapsed_guess = type_ms + 1300 + sum(a.get("wait", 0) for a in actions) + 900 * len(actions)
        page.wait_for_timeout(max(800, ms_total - elapsed_guess))
        cast.stop()
        ctx.close()
        browser.close()
    return cast.encode(out_mp4, size, min_seconds=seconds)
