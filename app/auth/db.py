"""
DB connection for authentication and app settings — MongoDB Atlas backend.
Uses motor (async driver). Configure via MONGODB_URI env var.
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_client = None
_db = None


def _get_db():
    """Return the motor database, creating the client on first call."""
    global _client, _db
    if _db is not None:
        return _db

    from motor.motor_asyncio import AsyncIOMotorClient

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "rubricai_auth")

    # AsyncIOMotorClient must be created inside the running event loop.
    # Calling this from a coroutine (init_db) guarantees that.
    _client = AsyncIOMotorClient(uri)
    _db = _client[db_name]
    logger.info("MongoDB client created (db=%s)", db_name)
    return _db


async def _ensure_indexes():
    """Create unique indexes if they don't exist."""
    db = _get_db()
    # Use create_index on the async motor collection
    await db["users"].create_index("email", unique=True)
    await db["app_settings"].create_index("key", unique=True)
    logger.info("MongoDB indexes ensured")


async def init_db():
    """Warm up connection and ensure indexes. Call once at startup."""
    await _ensure_indexes()


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    db = _get_db()
    doc = await db["users"].find_one({"email": email}, {"_id": 0})
    return doc or None


async def upsert_oauth_user(email: str, name: str, provider: str) -> Dict[str, Any]:
    db = _get_db()
    await db["users"].update_one(
        {"email": email},
        {
            "$set": {"name": name, "provider": provider},
            "$setOnInsert": {
                "email": email,
                "hashed_password": None,
                "is_active": True,
                "role": "rubricador",
            },
        },
        upsert=True,
    )
    doc = await db["users"].find_one({"email": email}, {"_id": 0})
    return doc


async def create_local_user(
    email: str, name: str, hashed_password: str, role: str = "rubricador"
) -> Dict[str, Any]:
    db = _get_db()
    existing = await db["users"].find_one({"email": email})
    if existing:
        raise ValueError(f"User {email} already exists")
    doc = {
        "email": email,
        "name": name,
        "provider": "local",
        "hashed_password": hashed_password,
        "is_active": True,
        "role": role,
    }
    await db["users"].insert_one(doc)
    doc.pop("_id", None)
    return doc


async def list_users() -> list:
    db = _get_db()
    cursor = db["users"].find({}, {"_id": 0, "hashed_password": 0}).sort("email", 1)
    return await cursor.to_list(length=None)


async def update_user(email: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = _get_db()
    result = await db["users"].find_one_and_update(
        {"email": email},
        {"$set": updates},
        projection={"_id": 0, "hashed_password": 0},
        return_document=True,
    )
    return result


async def delete_user(email: str) -> bool:
    db = _get_db()
    result = await db["users"].delete_one({"email": email})
    return result.deleted_count > 0


async def has_any_admin() -> bool:
    db = _get_db()
    doc = await db["users"].find_one({"role": "admin"})
    return doc is not None


# ---------------------------------------------------------------------------
# App Settings helpers
# ---------------------------------------------------------------------------

async def get_setting(key: str) -> Optional[str]:
    db = _get_db()
    doc = await db["app_settings"].find_one({"key": key}, {"_id": 0})
    return doc["value"] if doc else None


async def get_all_settings() -> Dict[str, str]:
    db = _get_db()
    cursor = db["app_settings"].find({}, {"_id": 0})
    docs = await cursor.to_list(length=None)
    return {d["key"]: d["value"] for d in docs}


async def upsert_setting(key: str, value: str) -> None:
    db = _get_db()
    await db["app_settings"].update_one(
        {"key": key},
        {"$set": {"key": key, "value": value}},
        upsert=True,
    )


async def upsert_settings(data: Dict[str, str]) -> None:
    for k, v in data.items():
        await upsert_setting(k, v)


# ---------------------------------------------------------------------------
# Backward-compat stubs
# ---------------------------------------------------------------------------
async def get_pool():
    return _get_db()


async def get_session():
    raise NotImplementedError("Use MongoDB helpers directly, not get_session()")
