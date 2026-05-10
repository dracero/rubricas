"""Pydantic models for authentication."""
from typing import Optional, Literal
from pydantic import BaseModel


class UserOut(BaseModel):
    email: str
    name: str
    provider: str
    is_active: bool
    role: str = "pending"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class LoginRequest(BaseModel):
    email: str
    password: str
