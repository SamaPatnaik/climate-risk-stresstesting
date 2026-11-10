"""Build a static snapshot of the web app (no backend needed).

Every API response is precomputed, so each endpoint is written to a JSON file
under dist/static/data/ and the frontend reads those instead (window.API_STATIC).
The output can be hosted anywhere static (Claude Artifact, GitHub Pages, ...).

Run from project root:
    python -m scripts.build_static
"""
import json
import re
import shutil

import requests
from fastapi.testclient import TestClient

from api.main import create_app
from src.paths import ROOT

WEB = ROOT / "web"
DIST = ROOT / "dist" / "static"
# Hosts like Claude Artifacts only allow stylesheets from Google Fonts, so Leaflet's CSS is inlined.
LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"


def write_json(name: str, obj) -> None:
    (DIST / "data" / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def snapshot_api() -> None:
    client = TestClient(create_app())

    def get(path: str):
        r = client.get(path)
        r.raise_for_status()
        return r.json()

    scenarios = get("/api/scenarios")
    write_json("scenarios", scenarios)
    for name in ("regions", "risk", "sensitivity", "assumptions"):
        write_json(name, get(f"/api/{name}"))
    for s in scenarios["scenarios"]:
        write_json(f"losses_{s['id']}", get(f"/api/losses?scenario={s['id']}"))


def build_page() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    leaflet = requests.get(LEAFLET_CSS, timeout=30)
    leaflet.raise_for_status()
    html = html.replace(f'<link rel="stylesheet" href="{LEAFLET_CSS}">', f"<style>{leaflet.text}</style>")
    # The artifact host supplies doctype/html/head/body, charset and viewport itself.
    html = re.sub(r"<!doctype html>|</?html[^>]*>|</?head>|</?body>|<meta (charset|name=\"viewport\")[^>]*>",
                  "", html, flags=re.I)
    (DIST / "index.html").write_text(html.strip() + "\n", encoding="utf-8")


def main() -> None:
    shutil.rmtree(DIST, ignore_errors=True)
    (DIST / "data").mkdir(parents=True)
    snapshot_api()
    build_page()
    shutil.copy(WEB / "style.css", DIST / "style.css")
    shutil.copy(WEB / "app.js", DIST / "app.js")
    (DIST / "config.js").write_text("// Static snapshot: read data/*.json instead of the API.\n"
                                    "window.API_STATIC = true;\n", encoding="utf-8")
    print(f"wrote {DIST}")
    for p in sorted(DIST.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(DIST)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
