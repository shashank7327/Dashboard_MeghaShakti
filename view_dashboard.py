r"""
view_dashboard.py  —  open the dashboard.

    py -3.13 view_dashboard.py

That is all. It starts a small web server on your own machine, opens your
browser at it, and stays running until you press Ctrl+C. Nothing is uploaded,
nothing is installed, and no internet connection is used.

WHY A SCRIPT RATHER THAN "JUST DOUBLE-CLICK THE HTML FILE"
  The dashboard is a modern JavaScript application, and browsers refuse to load
  that kind of application straight off the disk. Chrome, Edge and Firefox all
  block it for security reasons: a page opened from a file is not allowed to
  load its own program files alongside it.

  The page is not broken -- it needs to be SERVED rather than opened, and the
  browser will not do that for you. Double-clicking index.html gives a blank
  screen with no error, which is the most confusing possible failure, so this
  script exists to make sure that never happens.

  This uses only what comes with Python. There is nothing to install.
"""
import argparse
import http.server
import os
import pathlib
import socket
import socketserver
import sys
import threading
import webbrowser

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE / "dashboard"


def free_port(preferred=8000):
    """The preferred port, or the next free one. Something else on the machine
    is often already using 8000, and failing with 'address in use' would be a
    dead end for a non-technical reader."""
    for p in range(preferred, preferred + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true",
                    help="start the server but do not open a browser")
    a = ap.parse_args()

    if not (SITE / "index.html").exists():
        sys.exit(f"No dashboard found in {SITE}\n"
                 f"Build one with:  py -3.13 -X utf8 build_dashboard.py")

    port = free_port(a.port)
    os.chdir(SITE)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass                      # keep the terminal readable

    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print("=" * 66)
        print("  MonsoonCast dashboard")
        print("=" * 66)
        print(f"  Open:  {url}")
        print(f"  Serving: {SITE}")
        print("\n  Press Ctrl+C to stop.")
        if not a.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


if __name__ == "__main__":
    main()
