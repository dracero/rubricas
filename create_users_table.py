#!/usr/bin/env python
"""
create_users_table.py — Crea las tablas 'users' y 'authorization' en la base de datos.

Uso:
    python create_users_table.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    email           TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT 'local',
    hashed_password TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_AUTHORIZATION_SQL = """
CREATE TABLE IF NOT EXISTS "authorization" (
    email       TEXT PRIMARY KEY REFERENCES users(email) ON DELETE CASCADE,
    is_valid    BOOLEAN NOT NULL DEFAULT TRUE,
    role        TEXT NOT NULL DEFAULT 'usuario' CHECK (role IN ('administracion', 'usuario')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def main():
    db_type = os.getenv("DB_TYPE", "local").lower()
    print(f"DB_TYPE = {db_type}")

    conn = None
    connector = None

    if db_type == "cloudsql":
        conn, connector = await _connect_cloudsql()
    else:
        conn, connector = await _connect_local()

    try:
        # Eliminar tablas existentes (authorization primero por FK)
        await conn.execute('DROP TABLE IF EXISTS "authorization" CASCADE;')
        await conn.execute("DROP TABLE IF EXISTS users CASCADE;")
        print("Tablas anteriores eliminadas (si existían).")

        # Crear tabla users
        await conn.execute(CREATE_USERS_SQL)
        print("Tabla 'users' creada correctamente (o ya existía).")

        # Crear tabla authorization
        await conn.execute(CREATE_AUTHORIZATION_SQL)
        print("Tabla 'authorization' creada correctamente (o ya existía).")

        # Verificar que existen
        for table in ("users", "authorization"):
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                table,
            )
            if exists:
                print(f"Verificación: tabla '{table}' existe en schema 'public'.")
            else:
                print(f"Error: tabla '{table}' no encontrada.", file=sys.stderr)
                sys.exit(1)
    finally:
        if conn is not None:
            await conn.close()
        if connector is not None:
            await connector.close_async()


async def _connect_cloudsql():
    from google.cloud.sql.connector import Connector, IPTypes

    instance = os.getenv("CLOUDSQL_INSTANCE")
    user = os.getenv("CLOUDSQL_DB_USER")
    password = os.getenv("CLOUDSQL_DB_PASS")
    db = os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth")
    use_iam = os.getenv("CLOUDSQL_USE_IAM", "false").lower() == "true"

    if not instance:
        print("Error: CLOUDSQL_INSTANCE no está configurada en .env", file=sys.stderr)
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


if __name__ == "__main__":
    asyncio.run(main())
