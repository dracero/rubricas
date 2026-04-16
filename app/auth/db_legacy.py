"""
DB connection for authentication.
Supports Cloud SQL (Google) via cloud-sql-python-connector and local PostgreSQL.
DB_TYPE env var: "cloudsql" | "local"
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_pool = None


async def get_pool():
    global _pool
    if _pool is not None:
        return _pool
    db_type = os.getenv("DB_TYPE", "local").lower()
    if db_type == "cloudsql":
        _pool = await _create_cloudsql_pool()
    else:
        _pool = await _create_local_pool()
    await _init_schema(_pool)
    return _pool


def _build_connector():
    """Crea un Connector usando service account JSON si está configurado, o ADC si no."""
    import asyncio
    from google.cloud.sql.connector import Connector

    loop = asyncio.get_running_loop()
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if key_file:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            key_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        logger.info("Cloud SQL: usando service account desde %s", key_file)
        return Connector(credentials=creds, loop=loop)
    logger.info("Cloud SQL: usando Application Default Credentials (ADC)")
    return Connector(loop=loop)


async def _create_cloudsql_pool():
    import asyncpg
    from google.cloud.sql.connector import IPTypes

    connector = _build_connector()
    instance = os.getenv("CLOUDSQL_INSTANCE")
    user = os.getenv("CLOUDSQL_DB_USER")
    password = os.getenv("CLOUDSQL_DB_PASS")
    db = os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth")
    use_iam = os.getenv("CLOUDSQL_USE_IAM", "false").lower() == "true"

    async def getconn(*args, **kwargs):
        return await connector.connect_async(
            instance,
            "asyncpg",
            user=user,
            password=None if use_iam else password,
            db=db,
            enable_iam_auth=use_iam,
            ip_type=IPTypes.PUBLIC,
        )

    pool = await asyncpg.create_pool(connect=getconn, min_size=2, max_size=10)
    logger.info("Cloud SQL pool created (%s)", instance)
    return pool


async def _create_local_pool():
    import asyncpg

    pool = await asyncpg.create_pool(
        host=os.getenv("LOCAL_DB_HOST", "localhost"),
        port=int(os.getenv("LOCAL_DB_PORT", "5432")),
        user=os.getenv("LOCAL_DB_USER"),
        password=os.getenv("LOCAL_DB_PASS"),
        database=os.getenv("LOCAL_DB_NAME", "rubricai_auth"),
        min_size=2,
        max_size=10,
    )
    logger.info("Local PostgreSQL pool created")
    return pool


async def _init_schema(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email           TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                provider        TEXT NOT NULL DEFAULT 'local',
                hashed_password TEXT,
                is_active       BOOLEAN NOT NULL DEFAULT TRUE,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                role            TEXT NOT NULL DEFAULT 'user'
            )
            """
        )
        await conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'"
        )
    logger.info("Auth schema ready")

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT email, name, provider, hashed_password, is_active, role "
            "FROM users WHERE email = $1",
            email,
        )
    return dict(row) if row else None

async def upsert_oauth_user(email: str, name: str, provider: str) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (email, name, provider)
            VALUES ($1, $2, $3)
            ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
            RETURNING email, name, provider, hashed_password, is_active, role
            """,
            email,
            name,
            provider,
        )
    return dict(row)

async def create_local_user(
    email: str, name: str, hashed_password: str, role: str = 'user'
) -> Dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (email, name, provider, hashed_password, role)
            VALUES ($1, $2, 'local', $3, $4)
            RETURNING email, name, provider, hashed_password, is_active, role
            """,
            email,
            name,
            hashed_password,
            role,
        )
    return dict(row)
