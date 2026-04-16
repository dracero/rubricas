"""Auth router: mode, local login, OAuth2 (Google, Microsoft, UChile), /me."""
import os
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth

from app.auth.models import LoginRequest, TokenResponse, UserOut
from app.auth.db import get_user_by_email, upsert_oauth_user, create_local_user
from app.auth.service import hash_password, verify_password, create_access_token, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ── OAuth registry ────────────────────────────────────────────────────────────
oauth = OAuth()

_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if _GOOGLE_CLIENT_ID:
    oauth.register(
        name="google",
        client_id=_GOOGLE_CLIENT_ID,
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

_MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
if _MICROSOFT_CLIENT_ID:
    _tenant = os.getenv("MICROSOFT_TENANT_ID", "common")
    oauth.register(
        name="microsoft",
        client_id=_MICROSOFT_CLIENT_ID,
        client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
        server_metadata_url=(
            f"https://login.microsoftonline.com/{_tenant}/v2.0/.well-known/openid-configuration"
        ),
        client_kwargs={"scope": "openid email profile"},
    )

_UCHILE_CLIENT_ID = os.getenv("UCHILE_CLIENT_ID")
if _UCHILE_CLIENT_ID and os.getenv("UCHILE_AUTH_URL"):
    oauth.register(
        name="uchile",
        client_id=_UCHILE_CLIENT_ID,
        client_secret=os.getenv("UCHILE_CLIENT_SECRET"),
        authorize_url=os.getenv("UCHILE_AUTH_URL"),
        access_token_url=os.getenv("UCHILE_TOKEN_URL"),
        client_kwargs={"scope": "openid email profile"},
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/mode")
def get_auth_mode():
    """Returns configured AUTH_MODE. Empty string means show all providers."""
    mode = os.getenv("AUTH_MODE", "").strip().upper()
    return {"mode": mode}


@router.post("/login/local", response_model=TokenResponse)
async def login_local(body: LoginRequest):
    user = await get_user_by_email(body.email)
    if not user or not user.get("hashed_password"):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    if not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = create_access_token(user["email"])
    return TokenResponse(access_token=token, user=UserOut(**user))


@router.get("/login/{provider}")
async def login_oauth(provider: str, request: Request):
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' not configured")
    redirect_uri = str(request.url_for("callback_oauth", provider=provider))
    return await client.authorize_redirect(request, redirect_uri)


@router.get("/callback/{provider}", name="callback_oauth")
async def callback_oauth(provider: str, request: Request):
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' not configured")
    token_data = await client.authorize_access_token(request)
    user_info = token_data.get("userinfo") or {}
    if not user_info:
        try:
            user_info = await client.userinfo(token=token_data)
        except Exception:
            pass
    email = user_info.get("email")
    name = user_info.get("name") or user_info.get("preferred_username") or email
    if not email:
        raise HTTPException(status_code=400, detail="Provider did not return email")
    db_user = await upsert_oauth_user(email=email, name=name, provider=provider)
    jwt_token = create_access_token(db_user["email"])
    return RedirectResponse(url=f"{FRONTEND_URL}/?token={jwt_token}")


@router.get("/me", response_model=UserOut)
async def get_me(current_user=Depends(get_current_user)):
    return UserOut(**current_user)
