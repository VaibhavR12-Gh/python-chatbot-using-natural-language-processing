"""
app.py — HTTP Server for the NLP Chatbot

Uses Python's built-in http.server (no Flask needed).
Serves:
  GET  /          → index.html (chat UI)
  POST /chat      → JSON { "message": "..." } → JSON response
  GET  /stats     → chatbot session statistics
  POST /reset     → reset conversation history
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from chatbot import NLPChatbot

# ── Global chatbot instance ─────────────────
BOT = NLPChatbot("intents.json")
HTML_FILE = Path("index.html")


class ChatHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Custom minimal logging."""
        print(f"  [{self.command}] {self.path} → {args[1] if len(args) > 1 else ''}")

    # ── ROUTING ───────────────────────────────

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/stats":
            self._serve_json(BOT.get_stats())
        else:
            self._send_404()

    def do_POST(self):
        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/reset":
            BOT.reset()
            self._serve_json({"status": "ok", "message": "Conversation reset."})
        else:
            self._send_404()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── HANDLERS ─────────────────────────────

    def _handle_chat(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            user_message = data.get("message", "").strip()
            debug_mode = data.get("debug", False)

            if not user_message:
                self._serve_json({"error": "Empty message"}, status=400)
                return

            result = BOT.respond(user_message, debug=debug_mode)
            self._serve_json(result)

        except json.JSONDecodeError:
            self._serve_json({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            self._serve_json({"error": str(e)}, status=500)

    # ── RESPONSE HELPERS ─────────────────────

    def _serve_html(self):
        try:
            html = HTML_FILE.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except FileNotFoundError:
            self._serve_json({"error": "index.html not found"}, status=404)

    def _serve_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        self._serve_json({"error": f"Not found: {self.path}"}, status=404)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


# ── MAIN ─────────────────────────────────────

def run(host: str = "localhost", port: int = 8080):
    os.chdir(Path(__file__).parent)  # Ensure correct working dir
    server = HTTPServer((host, port), ChatHandler)
    print(f"""
╔══════════════════════════════════════════╗
║       PyBot NLP Chatbot Server           ║
╠══════════════════════════════════════════╣
║  URL:    http://{host}:{port}              ║
║  Chat:   POST /chat                      ║
║  Stats:  GET  /stats                     ║
║  Reset:  POST /reset                     ║
╚══════════════════════════════════════════╝
Press Ctrl+C to stop.
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    run()
