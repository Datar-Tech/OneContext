"""Local HTTP API server for one-click agent import from browser."""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from ..logging_config import setup_logger
from .branding import BRANDING

logger = setup_logger("realign.dashboard.local_api", "local_api.log")

ALLOWED_ORIGINS = [
    "https://realign-server.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class LocalAPIHandler(BaseHTTPRequestHandler):
    """Handle local API requests from the browser."""

    def _set_cors_headers(self) -> bool:
        """Set CORS headers. Returns True if origin is allowed."""
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Helps with browsers implementing Private Network Access.
            # Safe to include even when not requested.
            self.send_header("Access-Control-Allow-Private-Network", "true")
            return True
        return False

    def _send_json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, status: int, html: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(200, {"status": "ok"})
        elif parsed.path == "/import-agent":
            self._handle_import_agent_page(parsed.query)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/import-agent":
            self._handle_import_agent()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_import_agent(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        share_url = body.get("share_url")
        if not share_url:
            self._send_json(400, {"error": "share_url is required"})
            return

        password = body.get("password")

        logger.info(f"Import agent request: {share_url}")

        try:
            from ..commands.import_shares import import_agent_from_share

            result = import_agent_from_share(share_url, password=password)
            if result.get("success"):
                logger.info(
                    f"Agent imported: {result.get('agent_name')} "
                    f"({result.get('sessions_imported')} sessions)"
                )
                self._send_json(200, result)
            else:
                logger.warning(f"Import failed: {result.get('error')}")
                self._send_json(422, result)
        except Exception as e:
            logger.error(f"Import agent error: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})

    def _handle_import_agent_page(self, query: str) -> None:
        params = parse_qs(query or "")
        share_url = (params.get("share_url") or [None])[0]
        password = (params.get("password") or [None])[0]
        opener_origin = (params.get("opener_origin") or [None])[0]

        if not share_url:
            self._send_html(
                400,
                "<!doctype html><html><body><h3>Missing share_url</h3></body></html>",
            )
            return

        logger.info(f"Import agent request (page): {share_url}")

        try:
            from ..commands.import_shares import import_agent_from_share

            result = import_agent_from_share(share_url, password=password)
        except Exception as e:
            logger.error(f"Import agent error (page): {e}", exc_info=True)
            result = {"success": False, "error": str(e)}

        # If opener_origin is untrusted, don't postMessage back.
        target_origin = opener_origin if opener_origin in ALLOWED_ORIGINS else ""
        result_json = json.dumps(result)
        target_origin_json = json.dumps(target_origin)

        status_text = "Import Successful" if result.get("success") else "Import Failed"
        detail_text = ""
        if result.get("success"):
            agent_name = result.get("agent_name") or "Imported Agent"
            detail_text = f"<strong>{agent_name}</strong> has been imported to your local OneContext."
        else:
            error_msg = result.get("error") or "Unknown error"
            detail_text = error_msg

        success = result.get("success")
        footer_text = (
            "This window will close automatically."
            if success
            else 'Please try again, or import manually by copying the share link to the <strong>"Add Context"</strong> button in OneContext.'
        )

        html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
	    <title>{BRANDING.import_page_title}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; padding: 24px; background: #f8fafc; }}
      .card {{ max-width: 520px; margin: 40px auto; border: 1px solid #e5e7eb; border-radius: 16px; padding: 24px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
      .icon {{ font-size: 32px; margin-bottom: 8px; }}
      h2 {{ margin-top: 8px; }}
      .ok {{ color: #059669; }}
      .err {{ color: #dc2626; }}
      .footer {{ margin-top: 16px; color: #64748b; font-size: 13px; }}
      .source {{ margin-top: 12px; opacity: 0.5; font-size: 11px; word-break: break-all; }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="icon">{"✅" if success else "❌"}</div>
      <h2 class="{ 'ok' if success else 'err' }">{status_text}</h2>
      <p>{detail_text}</p>
      <p class="footer">{footer_text}</p>
      <p class="source">{share_url}</p>
    </div>
    <script>
      (function () {{
        const result = {result_json};
        const targetOrigin = {target_origin_json};
        try {{
          if (window.opener && targetOrigin) {{
            window.opener.postMessage({{ type: 'aline-import-agent-result', result }}, targetOrigin);
          }}
        }} catch {{}}
      }})();
    </script>
  </body>
</html>
"""
        self._send_html(200, html)

    def log_message(self, format: str, *args) -> None:
        """Suppress default stderr logging; use our logger instead."""
        logger.debug(f"HTTP {args[0] if args else ''}")


class LocalAPIServer:
    """Manages the local HTTP API server in a daemon thread."""

    def __init__(self, port: int = 17280):
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the server. Returns True on success."""
        try:
            self._server = HTTPServer(("127.0.0.1", self.port), LocalAPIHandler)
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
            logger.info(f"Local API server started on http://127.0.0.1:{self.port}")
            return True
        except OSError as e:
            logger.warning(f"Failed to start local API server on port {self.port}: {e}")
            return False

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("Local API server stopped")
