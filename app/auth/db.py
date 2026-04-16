"""
DB connection for authentication and app settings.
Supports Cloud SQL (Google), local PostgreSQL, and SQLite.
DB_TYPE env var: "cloudsql" | "local" | "sqlite"
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, Text, Boolean, DateTime, MetaData, Table, func, text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("email", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("provider", Text, nullable=False, server_default="local"),
    Column("hashed_password", Text),
    Column("is_active", Boolean, nullable=False, server_default="true"),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("role", Text, nullable=False, server_default="user"),
)

app_settings = Table(
    "app_settings", metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False, server_default=""),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------
_engine = None
_session_factory = None


def _build_database_url() -> str:
    db_type = os.getenv("DB_TYPE", "local").lower()

    if db_type == "sqlite":
        db_path = os.getenv("SQLITE_PATH", "data/rubricai.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"

    if db_type == "cloudsql":
        user = os.getenv("CLOUDSQL_DB_USER", "")
        password = os.getenv("CLOUDSQL_DB_PASS", "")
        db = os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth")
        instance = os.getenv("CLOUDSQL_INSTANCE", "")
        # Cloud SQL Connector handles the actual socket; we use asyncpg creator
        # Return a placeholder — _get_engine overrides with `creator`
        return f"postgresql+asyncpg://{user}:{password}@/{db}"

    # local postgres
    host = os.getenv("LOCAL_DB_HOST", "localhost")
    port = os.getenv("LOCAL_DB_PORT", "5432")
    user = os.getenv("LOCAL_DB_USER", "postgres")
    password = os.getenv("LOCAL_DB_PASS", "")
    db = os.getenv("LOCAL_DB_NAME", "rubricai_auth")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


async def _get_engine():
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    db_type = os.getenv("DB_TYPE", "local").lower()
    url = _build_database_url()

    if db_type == "cloudsql":
        from google.cloud.sql.connector import Connector, IPTypes
        import asyncio

        key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        loop = asyncio.get_running_loop()
        if key_file:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                key_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            connector = Connector(credentials=creds, loop=loop)
        else:
            connector = Connector(loop=loop)

        instance = os.getenv("CLOUDSQL_INSTANCE")
        use_iam = os.getenv("CLOUDSQL_USE_IAM", "false").lower() == "true"

        async def getconn():
            return await connector.connect_async(
                instance, "asyncpg",
                user=os.getenv("CLOUDSQL_DB_USER"),
                password=None if use_iam else os.getenv("CLOUDSQL_DB_PASS"),
                db=os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth"),
                enable_iam_auth=use_iam,
                ip_type=IPTypes.PUBLIC,
            )

        _engine = create_async_engine(url, async_creator=getconn, pool_size=5, max_overflow=5)
        logger.info("Cloud SQL engine created (%s)", instance)
    else:
        _engine = create_async_engine(url, pool_size=5 if "sqlite" not in url else 1)
        logger.info("DB engine created (%s)", db_type)

    _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    logger.info("Auth + settings schema ready")

    return _engine


async def get_session() -> AsyncSession:
    """Get a new async session. Caller must use `async with`."""
    await _get_engine()
    return _session_factory()


# ---------------------------------------------------------------------------
# Backward-compat: get_pool returns engine (for make_admin.py etc.)
# ---------------------------------------------------------------------------
async def get_pool():
    return await _get_engine()


# ---------------------------------------------------------------------------
# User CRUD (same signatures as before)
# ---------------------------------------------------------------------------
async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    async with await get_session() as session:
        result = await session.execute(
            users.select().where(users.c.email == email)
        )
        row = result.mappings().first()
    return dict(row) if row else None


async def upsert_oauth_user(email: str, name: str, provider: str) -> Dict[str, Any]:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sq_insert

    db_type = os.getenv("DB_TYPE", "local").lower()
    async with await get_session() as session:
        async with session.begin():
            if db_type == "sqlite":
                stmt = sq_insert(users).values(email=email, name=name, provider=provider)
                stmt = stmt.on_conflict_do_update(index_elements=["email"], set_={"name": name})
            else:
                stmt = pg_insert(users).values(email=email, name=name, provider=provider)
                stmt = stmt.on_conflict_do_update(index_elements=["email"], set_={"name": name})
            stmt = stmt.returning(users)
            result = await session.execute(stmt)
            row = result.mappings().first()
    return dict(row)


async def create_local_user(email: str, name: str, hashed_password: str, role: str = "user") -> Dict[str, Any]:
    async with await get_session() as session:
        async with session.begin():
            stmt = users.insert().values(
                email=email, name=name, provider="local",
                hashed_password=hashed_password, role=role,
            ).returning(users)
            result = await session.execute(stmt)
            row = result.mappings().first()
    return dict(row)


# ---------------------------------------------------------------------------
# App Settings helpers
# ---------------------------------------------------------------------------
async def get_setting(key: str) -> Optional[str]:
    async with await get_session() as session:
        result = await session.execute(
            app_settings.select().where(app_settings.c.key == key)
        )
        row = result.mappings().first()
    return row["value"] if row else None


async def get_all_settings() -> Dict[str, str]:
    async with await get_session() as session:
        result = await session.execute(app_settings.select())
        rows = result.mappings().all()
    return {r["key"]: r["value"] for r in rows}


async def upsert_setting(key: str, value: str) -> None:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sq_insert

    db_type = os.getenv("DB_TYPE", "local").lower()
    async with await get_session() as session:
        async with session.begin():
            if db_type == "sqlite":
                stmt = sq_insert(app_settings).values(key=key, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
            else:
                stmt = pg_insert(app_settings).values(key=key, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
            await session.execute(stmt)


async def upsert_settings(data: Dict[str, str]) -> None:
    for k, v in data.items():
        await upsert_setting(k, v)


async def has_any_admin() -> bool:
    async with await get_session() as session:
        result = await session.execute(
            users.select().where(users.c.role == "admin").limit(1)
        )
        return result.first() is not None
