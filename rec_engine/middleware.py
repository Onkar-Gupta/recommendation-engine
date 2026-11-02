"""FastAPI middleware slots. The free version ships open (auth off, usage
logged to a local file). A future paid hosted tier plugs real key validation
and billing into these same two functions, so the API itself needs no rewrite.
"""

import json
import time
from pathlib import Path

from fastapi.responses import JSONResponse


def make_auth_middleware(auth_config):
    async def auth_middleware(request, call_next):
        # Paid tier: validate the key against a store here. Free tier: presence check only.
        protected = auth_config.get("enabled") and request.url.path != "/health"
        if protected and not request.headers.get(auth_config["api_key_header"]):
            return JSONResponse({"detail": "Missing API key"}, status_code=401)
        return await call_next(request)

    return auth_middleware


def make_usage_tracking_middleware(log_path):
    log_path = Path(log_path)

    async def usage_tracking_middleware(request, call_next):
        # Paid tier: send this record to billing instead of a file.
        start = time.time()
        response = await call_next(request)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start)),
            "endpoint": request.url.path,
            "user_id": request.path_params.get("user_id"),
            "response_time_ms": round((time.time() - start) * 1000, 1),
            "status": response.status_code,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:  # noqa: ASYNC230 one tiny append per request, fine until traffic is high
            f.write(json.dumps(record) + "\n")
        return response

    return usage_tracking_middleware
