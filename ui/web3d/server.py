"""Local HTTP server that hosts the three.js 3D topology view. Runs in a
daemon background thread inside the same process as the Tkinter
Dashboard so it can read ``dashboard.sim`` / ``dashboard._net_canvas``
directly (no IPC, no extra dependency -- stdlib ``http.server`` only).
Zero-dependency by design: three.js itself is loaded by the *browser*
from a CDN when the page opens, not by this process, so nothing needs to
be pip- or npm-installed to use this feature.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .snapshot import build_snapshot

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Web3DServer:
    """Owns one background HTTP server bound to 127.0.0.1 on an
    OS-assigned free port. Call ``start()`` once, then ``url`` gives the
    address to open in a browser. Safe to leave running for the life of
    the process -- it's a daemon thread, so it never blocks shutdown."""

    def __init__(self, dashboard):
        self._dashboard = dashboard
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        dashboard = self._dashboard

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass  # silence the default per-request stderr spam

            def _send_bytes(self, body: bytes, content_type: str, cache: bool = True) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    path = "/index.html"

                if path == "/api/state":
                    try:
                        data = build_snapshot(dashboard)
                    except Exception as exc:  # keep the poll loop alive on transient errors
                        data = {"ready": False, "error": str(exc)}
                    self._send_bytes(json.dumps(data).encode("utf-8"),
                                      "application/json", cache=False)
                    return

                safe_rel = os.path.normpath(path).lstrip(os.sep)
                full = os.path.join(_STATIC_DIR, safe_rel)
                if not full.startswith(_STATIC_DIR) or not os.path.isfile(full):
                    self.send_error(404, "Not found")
                    return
                ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
                with open(full, "rb") as f:
                    self._send_bytes(f.read(), ctype)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                if path != "/api/move_node":
                    self.send_error(404, "Not found")
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    node_id = int(body["id"])
                    x, y = float(body["x"]), float(body["y"])
                    sim = getattr(dashboard, "sim", None)
                    node = sim.nodes[node_id] if sim is not None else None
                    # The AP is a fixed reference point everything else (roads,
                    # filler buildings, the plaza) is laid out around -- same
                    # rule the 2D topology canvas's own drag handler enforces
                    # (ui/topology_canvas.py's _on_press), so it's rejected
                    # here too rather than trusting the browser alone.
                    if node is None or node_id == 0:
                        self._send_bytes(json.dumps({"ok": False}).encode("utf-8"),
                                          "application/json", cache=False)
                        return
                    node.pos = (x, y)
                    self._send_bytes(json.dumps({"ok": True}).encode("utf-8"),
                                      "application/json", cache=False)
                except Exception as exc:
                    self._send_bytes(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                                      "application/json", cache=False)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._httpd.server_address[1] if self._httpd else 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
