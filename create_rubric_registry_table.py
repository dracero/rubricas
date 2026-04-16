#!/usr/bin/env python
"""
create_rubric_registry_table.py — Crea la tabla de registro de rubricas.

Modelo cubierto:
- Un usuario genera una o mas rubricas (owner_email).
- Cada rubrica guarda su conjunto de documentos en JSONB.
- La rubrica referencia su ubicacion en Qdrant.
- Vigencia habilitable/deshabilitable (is_enabled).
- Actualizaciones por versionado (version + updated_at).
- Eliminacion logica (is_deleted + deleted_at).

Uso:
    python create_rubric_registry_table.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


CREATE_RUBRIC_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS rubric_registry (
    rubric_id            TEXT PRIMARY KEY,
    owner_email          TEXT NOT NULL REFERENCES users(email) ON UPDATE CASCADE ON DELETE RESTRICT,
    qdrant_collection    TEXT NOT NULL DEFAULT 'rubricas_repositorio',
    qdrant_point_id      TEXT NOT NULL UNIQUE,
    title                TEXT NOT NULL,
    rubric_text          TEXT,
    source_documents     JSONB NOT NULL DEFAULT '[]'::jsonb,
    version              INTEGER NOT NULL DEFAULT 1,
    is_enabled           BOOLEAN NOT NULL DEFAULT TRUE,
    is_deleted           BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at           TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_version_positive CHECK (version >= 1),
    CONSTRAINT chk_deleted_consistency CHECK (
        (is_deleted = FALSE AND deleted_at IS NULL)
        OR (is_deleted = TRUE)
    )
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_rubric_registry_owner_email
    ON rubric_registry (owner_email);

CREATE INDEX IF NOT EXISTS idx_rubric_registry_enabled_not_deleted
    ON rubric_registry (is_enabled, is_deleted)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_rubric_registry_created_at
    ON rubric_registry (created_at DESC);
"""

CREATE_UPDATED_AT_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION set_rubric_registry_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_rubric_registry_updated_at ON rubric_registry;

CREATE TRIGGER trg_rubric_registry_updated_at
BEFORE UPDATE ON rubric_registry
FOR EACH ROW
EXECUTE FUNCTION set_rubric_registry_updated_at();
"""


async def _connect_cloudsql():
    from google.cloud.sql.connector import Connector, IPTypes

    instance = os.getenv("CLOUDSQL_INSTANCE")
    user = os.getenv("CLOUDSQL_DB_USER")
    password = os.getenv("CLOUDSQL_DB_PASS")
    db = os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth")
    use_iam = os.getenv("CLOUDSQL_USE_IAM", "false").lower() == "true"

    if not instance:
        print("Error: CLOUDSQL_INSTANCE no esta configurada en .env", file=sys.stderr)
        sys.exit(1)

    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    loop = asyncio.get_running_loop()
    if key_file:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            key_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        connector = Connector(credentials=creds, loop=loop)
    else:
        connector = Connector(loop=loop)

    try:
        conn = await connector.connect_async(
            instance,
            "asyncpg",
            user=user,
            password=None if use_iam else password,
            db=db,
            enable_iam_auth=use_iam,
            ip_type=IPTypes.PUBLIC,
        )
        return conn, connector
    except Exception:
        await connector.close_async()
        raise


async def _connect_local():
    import asyncpg

    conn = await asyncpg.connect(
        host=os.getenv("LOCAL_DB_HOST", "localhost"),
        port=int(os.getenv("LOCAL_DB_PORT", "5432")),
        user=os.getenv("LOCAL_DB_USER"),
        password=os.getenv("LOCAL_DB_PASS"),
        database=os.getenv("LOCAL_DB_NAME", "rubricai_auth"),
    )
    return conn, None


async def main() -> None:
    db_type = os.getenv("DB_TYPE", "local").lower()
    print(f"DB_TYPE = {db_type}")

    conn = None
    connector = None

    try:
        if db_type == "cloudsql":
            conn, connector = await _connect_cloudsql()
        else:
            conn, connector = await _connect_local()

        await conn.execute(CREATE_RUBRIC_REGISTRY_SQL)
        await conn.execute(CREATE_INDEXES_SQL)
        await conn.execute(CREATE_UPDATED_AT_TRIGGER_SQL)

        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'rubric_registry'
            )
            """
        )

        if not exists:
            print("Error: no se pudo crear la tabla rubric_registry", file=sys.stderr)
            sys.exit(1)

        print("Tabla 'rubric_registry' creada/verificada correctamente.")
        print("Indices y trigger de updated_at creados/verificados.")

    finally:
        if conn is not None:
            await conn.close()
        if connector is not None:
            await connector.close_async()


if __name__ == "__main__":
    asyncio.run(main())
