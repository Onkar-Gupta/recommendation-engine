"""FastAPI middleware slots. The free version ships open; these are placeholders
so a future paid hosted tier can plug in auth and billing without an API rewrite.
"""


def auth_middleware(request, call_next):
    # Paid tier: plug in here (API key check).
    return call_next(request)


def usage_tracking_middleware(request, call_next):
    # Paid tier: plug in here (log usage to billing instead of a file).
    return call_next(request)
