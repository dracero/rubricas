"""Auth middleware: protects /api/* routes, allows /auth/* without token."""
import os
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_OPEN_PREFIXES = ("/auth/", "/docs", "/openapi", "/redoc", "/")
_EXACT_OPEN = {"/"}


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    # Allow non-API paths and auth endpoints without a token
    # Also allow public system/brand endpoints needed before login
    if not path.startswith("/api/") or any(path.startswith(p) for p in ("/auth/",)):
        return await call_next(request)

    public_api = ("/api/system/status", "/api/system/setup", "/api/brand")
    if path in public_api:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    token = auth_header.split(" ", 1)[1]
    secret = os.getenv("SECRET_KEY", "change-me-in-production")
    try:
        jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError:
        return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)

    return await call_next(request)
