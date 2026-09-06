#!/usr/bin/env python3
"""40-line receiver that prints what the board's cloud sync sends.

Run on the laptop while the board pushes `kpi_15m` aggregates. Unplugging the
cable mid-demo shows nothing breaks and the backlog syncs on reconnect.

    python3 tools/cloud_receiver.py [--port 8000]
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        data = json.loads(body)
        rows = data.get("kpi", [])
        print(f"received {len(rows)} kpi rows; latest bucket {rows[-1]['t_bucket'] if rows else '-'}")
        for r in rows[-3:]:
            print(f"  {r['t_bucket']} {r['key']} = {r['value']}")
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    server = HTTPServer(("0.0.0.0", a.port), Handler)
    print(f"listening on :{a.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
