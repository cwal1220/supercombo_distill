"""Render reports/experiments.json (every attempt to beat the shipped model, with its metrics and verdict) as
reports/experiments.html: the best model against the shipped one, then each attempt with its metrics coloured
against the shipped model's (lower is better for all).

usage: .venv/bin/python scripts/report.py
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"lat_acc": "lat_acc", "long_acc": "long_acc", "plan_y2s_cm": "plan_y2s (cm)", "lanes_vs_ld_cm": "lanes (cm)",
         "width_vs_ld_cm": "width (cm)", "lead_x_m": "lead_x (m)", "gas2": "gas2", "latency_ms": "latency (ms)",
         "false_lead_pct": "헛 lead (%)", "vx0_cms": "vx0 (cm/s)", "lanes_ldfix_cm": "lanes 경사보정 (cm)",
         "width_ldfix_cm": "width 경사보정 (cm)"}
VERDICT = {"success": ("성공", "ok"), "partial": ("부분 성공", "mid"), "fail": ("실패", "bad"), "info": ("진단", "info"),
           "running": ("진행 중", "run")}
CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e4e1da;--ok:#1f7a4d;--okbg:#e5f3ea;--bad:#b23a2b;
--badbg:#f8e6e2;--mid:#8a6400;--midbg:#f6efd9;--info:#2d5f9a;--infobg:#e4edf8;--run:#6b4fa3;--runbg:#eee8f7}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--card:#21211f;
--line:#34332f;--ok:#6fcf97;--okbg:#1c3327;--bad:#f08b7c;--badbg:#3a221e;--mid:#e3bd5c;--midbg:#332b16;--info:#8db7ea;
--infobg:#1b2a3c;--run:#b9a2e6;--runbg:#2a2338}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,
"Apple SD Gothic Neo","Noto Sans KR",sans-serif}main{max-width:1100px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:34px 0 10px}p{margin:6px 0}.mut{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}.better{color:var(--ok);font-weight:600}.worse{color:var(--bad);font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;white-space:nowrap}
.ok{color:var(--ok);background:var(--okbg)}.bad{color:var(--bad);background:var(--badbg)}.mid{color:var(--mid);background:var(--midbg)}
.info{color:var(--info);background:var(--infobg)}.run{color:var(--run);background:var(--runbg)}
.exp h3{font-size:16px;margin:0 0 4px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.meta{font-size:13px;color:var(--mut)}
table.txt td,table.txt th{white-space:normal;text-align:left;vertical-align:top}table.txt td:nth-child(2){white-space:nowrap}
.cap{font-size:13px;color:var(--mut);margin:12px 0 4px}figure{margin:12px 0 0}figure img{width:100%;border-radius:8px;display:block}
figcaption{font-size:13px;color:var(--mut);margin-top:6px}
"""


def cell(k, v, ref):
    if v is None:
        return "<td class=mut>–</td>"
    r = ref.get(k)
    cls = "" if r is None else ("better" if v < r else "worse" if v > r else "")
    fmt = f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return f"<td class='{cls}'>{fmt}</td>"


def five_routes(d, keys, head):
    """The 5-route table (from 2026-10-30): every attempt evaluated there, against the shipped model's 5-route values."""
    ref = d["shipped5"]
    rows = [e for e in d["experiments"] if e.get("metrics5")]
    out = ["<h2>평가 경로 5개 (현재 기준)</h2>", f"<p class=mut>{html.escape(d.get('metric_note5', ''))}</p>",
           f"<div class='card tw'><table><tr><th>모델</th>{head}</tr>",
           "<tr><td>출시 모델</td>" + "".join(cell(k, ref.get(k), {}) for k in keys) + "</tr>"]
    for e in rows:
        out.append(f"<tr><td>{html.escape(e['id'])}</td>" + "".join(cell(k, e["metrics5"].get(k), ref) for k in keys) + "</tr>")
    out.append("</table></div>")
    return out


def image(path: str) -> str:
    """A screenshot inlined as a JPEG data URI (the report is sent as a single file)."""
    import base64
    import io
    from PIL import Image
    im = Image.open(ROOT / path).convert("RGB")
    im.thumbnail((1100, 1100))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def section(s: dict) -> list:
    """A summary section: paragraphs, tables (badge_col: a column of verdicts shown as badges), bullets, an image."""
    out = [f"<h2>{html.escape(s['title'])}</h2><div class=card>"]
    out += [f"<p>{html.escape(p)}</p>" for p in s.get("paras", [])]
    for t in s.get("tables", []):
        if t.get("caption"):
            out.append(f"<p class=cap>{html.escape(t['caption'])}</p>")
        bc = t.get("badge_col")
        cls = " class=txt" if bc is not None else ""
        out.append(f"<div class=tw><table{cls}><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in t["head"]) + "</tr>")
        for r in t["rows"]:
            cells = [f"<td><span class='badge {VERDICT[c][1]}'>{VERDICT[c][0]}</span></td>" if i == bc else f"<td>{html.escape(c)}</td>"
                     for i, c in enumerate(r)]
            out.append("<tr>" + "".join(cells) + "</tr>")
        out.append("</table></div>")
    if s.get("bullets"):
        out.append("<ul>" + "".join(f"<li>{html.escape(b)}</li>" for b in s["bullets"]) + "</ul>")
    if s.get("image"):
        out.append(f"<figure><img src='{image(s['image'])}' alt='{html.escape(s.get('image_caption', ''))}'>"
                   f"<figcaption>{html.escape(s.get('image_caption', ''))}</figcaption></figure>")
    out.append("</div>")
    return out


def main():
    d = json.loads((ROOT / "reports/experiments.json").read_text())
    keys = d["metrics"]
    mk, rk = ("metrics5", "shipped5") if d.get("shipped5") else ("metrics", "shipped")   # the current (5-route) set
    ref = d[rk]
    scored = [e for e in d["experiments"] if e.get(mk) and e.get("candidate", True)]
    best = min(scored, key=lambda e: sum(e[mk].get(k, float("inf")) >= ref[k] for k in keys if ref.get(k)) * 100   # unmeasured = lost
               + e[mk].get("plan_y2s_cm", 99))
    wins = [k for k in keys if ref.get(k) is not None and best[mk].get(k) is not None and best[mk][k] < ref[k]]
    loses = [k for k in keys if ref.get(k) is not None and best[mk].get(k, float("inf")) >= ref[k]]
    head = "".join(f"<th>{NAMES.get(k, k)}</th>" for k in keys)
    out = [f"<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Student 개선 기록</title><style>{CSS}</style></head><body><main>",
           "<h1>Student 개선 기록</h1>", f"<p>{html.escape(d['goal'])}</p>", f"<p class=mut>{html.escape(d['metric_note'])}</p>",
           *(["<div class=card><b>현재 상황</b><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in d["status"]) + "</ul></div>"]
             if d.get("status") else []),
           *[x for s in d.get("sections", []) for x in section(s)],
           "<h2>현재 최고 모델과 출시 모델</h2><div class=card>",
           f"<p><b>{html.escape(best['title'])}</b> ({best['date']})</p>",
           f"<p>이기는 지표: {', '.join(NAMES.get(k, k) for k in wins) or '없음'}<br>지는 지표: "
           f"{', '.join(NAMES.get(k, k) for k in loses) or '없음 (목표 달성)'}</p>",
           *([f"<p class=mut>{html.escape(d['best_note'])}</p>"] if d.get("best_note") else []),
           f"<div class=tw><table><tr><th>모델 ({'평가 경로 5개' if mk == 'metrics5' else '평가 경로 4개'})</th>{head}</tr>",
           "<tr><td>출시 모델</td>" + "".join(cell(k, ref.get(k), {}) for k in keys) + "</tr>",
           f"<tr><td>{html.escape(best['id'])}</td>" + "".join(cell(k, best[mk].get(k), ref) for k in keys) + "</tr>",
           "</table></div></div>",
           *(five_routes(d, keys, head) if d.get("shipped5") else []),
           "<h2>전체 시도 요약 (평가 경로 4개)</h2><div class='card tw'><table>",
           f"<tr><th>시도</th><th>결과</th>{head}</tr>"]
    ref4 = d["shipped"]
    for e in d["experiments"]:
        lab, cls = VERDICT[e["verdict"]]
        m = e.get("metrics") or {}
        out.append(f"<tr><td>{html.escape(e['id'])}</td><td><span class='badge {cls}'>{lab}</span></td>" +
                   ("".join(cell(k, m.get(k), ref4) for k in keys) if m else f"<td colspan={len(keys)} class=mut>–</td>") + "</tr>")
    out.append("<tr><td>출시 모델</td><td></td>" + "".join(cell(k, ref4.get(k), {}) for k in keys) + "</tr></table></div>")
    if d.get("refs"):
        out.append("<h2>참고한 논문과 적용</h2><div class='card tw'><table><tr><th>논문</th><th>링크</th><th>적용</th></tr>" +
                   "".join(f"<tr><td style='white-space:normal'>{html.escape(t)}</td><td>{html.escape(l)}</td>"
                           f"<td style='white-space:normal;text-align:left'>{html.escape(n)}</td></tr>" for t, l, n in d["refs"]) +
                   "</table></div>")
    out.append("<h2>시도별 내용</h2>")
    for e in reversed(d["experiments"]):
        lab, cls = VERDICT[e["verdict"]]
        out.append(f"<div class='card exp'><h3>{html.escape(e['title'])} <span class='badge {cls}'>{lab}</span></h3>"
                   f"<div class=meta>{e['date']} · {html.escape(e['category'])} · {html.escape(e['id'])}</div>"
                   f"<p><b>한 것</b> {html.escape(e['what'])}</p><p><b>결과</b> {html.escape(e['result'])}</p>")
        extra = e.get("extra_metrics")
        if extra:
            ks = list(next(iter(extra.values())).keys())
            out.append("<div class=tw><table><tr><th></th>" + "".join(f"<th>{NAMES.get(k, k)}</th>" for k in ks) + "</tr>" +
                       "".join(f"<tr><td>{html.escape(n)}</td>" + "".join(cell(k, r.get(k), {}) for k in ks) + "</tr>"
                               for n, r in extra.items()) + "</table></div>")
        out.append("</div>")
    out.append("</main></body></html>")
    (ROOT / "reports/experiments.html").write_text("\n".join(out))
    print(ROOT / "reports/experiments.html", "best:", best["id"], "loses:", loses)


if __name__ == "__main__":
    main()
