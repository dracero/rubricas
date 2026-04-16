#!/usr/bin/env python
"""
healthcheck_db.py — Prueba de conectividad a la base de datos de AuthDB.

Uso:
    python healthcheck_db.py                  # usa DB_TYPE del .env
    DB_TYPE=cloudsql python healthcheck_db.py
    DB_TYPE=local    python healthcheck_db.py

Requiere que el .env esté en la raíz del proyecto.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


# ── helpers de colores ───────────────────────────────────────────────────────
def _ok(msg: str) -> str:
    return f"\033[92m✓  {msg}\033[0m"


def _fail(msg: str) -> str:
    return f"\033[91m✗  {msg}\033[0m"


def _info(msg: str) -> str:
    return f"\033[94m   {msg}\033[0m"


def _warn(msg: str) -> str:
    return f"\033[93m⚠  {msg}\033[0m"


# ── Cloud SQL ────────────────────────────────────────────────────────────────
async def _connect_cloudsql():
    from google.cloud.sql.connector import Connector, IPTypes

    instance = os.getenv("CLOUDSQL_INSTANCE")
    user = os.getenv("CLOUDSQL_DB_USER")
    password = os.getenv("CLOUDSQL_DB_PASS")
    db = os.getenv("CLOUDSQL_DB_NAME", "rubricai_auth")
    use_iam = os.getenv("CLOUDSQL_USE_IAM", "false").lower() == "true"

    if not instance:
        raise ValueError("CLOUDSQL_INSTANCE no está configurada en .env")

    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if key_file:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            key_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        print(_info(f"Credenciales: service account → {key_file}"))
        loop = asyncio.get_running_loop()
        connector = Connector(credentials=creds, loop=loop)
    else:
        print(_info("Credenciales: Application Default Credentials (ADC)"))
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)

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


# ── PostgreSQL local ─────────────────────────────────────────────────────────
async def _connect_local():
    import asyncpg

    conn = await asyncpg.connect(
        host=os.getenv("LOCAL_DB_HOST", "localhost"),
        port=int(os.getenv("LOCAL_DB_PORT", "5432")),
        user=os.getenv("LOCAL_DB_USER"),
        password=os.getenv("LOCAL_DB_PASS"),
        database=os.getenv("LOCAL_DB_NAME", "rubricai_auth"),
    )
    return conn, None  # None = no connector object


# ── Checks ───────────────────────────────────────────────────────────────────
async def check_connection(db_type: str) -> tuple[dict, object, object]:
    """Establece conexión y devuelve (info, conn, connector)."""
    t0 = time.perf_counter()
    if db_type == "cloudsql":
        conn, connector = await _connect_cloudsql()
    else:
        conn, connector = await _connect_local()
    conn_ms = (time.perf_counter() - t0) * 1000

    version = await conn.fetchval("SELECT version()")
    db_name = await conn.fetchval("SELECT current_database()")
    db_user = await conn.fetchval("SELECT current_user")

    info = dict(version=version, db=db_name, user=db_user, connect_ms=conn_ms)
    return info, conn, connector


async def check_select_one(conn) -> None:
    """Ejecuta SELECT 1 para validar lectura básica."""
    result = await conn.fetchval("SELECT 1")
    assert result == 1, f"SELECT 1 devolvió {result!r}"


async def check_schema(conn) -> bool:
    """Retorna True si la tabla 'users' existe en el schema público."""
    exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'users'
        )
        """
    )
    return bool(exists)


# ── Punto de entrada ─────────────────────────────────────────────────────────
async def main() -> None:
    db_type = os.getenv("DB_TYPE", "local").lower()
    use_json = "--json" in sys.argv

    report = {
        "status": "ok",
        "db_type": db_type,
        "connection": None,
        "select_one": None,
        "schema_users": None,
        "error": None,
    }

    if not use_json:
        print()
        print("=" * 60)
        print("  RubricAI — Healthcheck de Base de Datos")
        print(f"  DB_TYPE = {db_type.upper()}")
        print("=" * 60)

    conn = None
    connector = None

    try:
        # 1. Conectar
        if not use_json:
            print("\n[1/3] Conectando...")
        try:
            info, conn, connector = await check_connection(db_type)
            report["connection"] = {
                "ok": True,
                "database": info["db"],
                "user": info["user"],
                "latency_ms": round(info["connect_ms"], 1),
                "version": info["version"],
            }
            if not use_json:
                if db_type == "cloudsql":
                    instance = os.getenv("CLOUDSQL_INSTANCE", "—")
                    print(_ok(f"Cloud SQL  →  {instance}"))
                else:
                    host = os.getenv("LOCAL_DB_HOST", "localhost")
                    port = os.getenv("LOCAL_DB_PORT", "5432")
                    print(_ok(f"PostgreSQL local  →  {host}:{port}"))
                print(_info(f"Base de datos : {info['db']}"))
                print(_info(f"Usuario       : {info['user']}"))
                print(_info(f"Latencia conn : {info['connect_ms']:.1f} ms"))
                print(_info(f"Versión       : {info['version'][:70]}"))
        except Exception as exc:
            report["status"] = "error"
            report["connection"] = {"ok": False, "error": str(exc)}
            report["error"] = str(exc)
            if use_json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(_fail(f"No se pudo conectar: {exc}"))
            sys.exit(1)

        # 2. SELECT 1
        if not use_json:
            print("\n[2/3] Verificando lectura (SELECT 1)...")
        try:
            await check_select_one(conn)
            report["select_one"] = {"ok": True}
            if not use_json:
                print(_ok("SELECT 1 = 1  (lectura OK)"))
        except Exception as exc:
            report["status"] = "error"
            report["select_one"] = {"ok": False, "error": str(exc)}
            report["error"] = str(exc)
            if use_json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(_fail(f"Error en SELECT 1: {exc}"))
            sys.exit(1)

        # 3. Schema
        if not use_json:
            print("\n[3/3] Verificando tabla 'users'...")
        try:
            exists = await check_schema(conn)
            report["schema_users"] = {"ok": True, "exists": exists}
            if not use_json:
                if exists:
                    print(_ok("Tabla 'users' encontrada en schema 'public'"))
                else:
                    print(_warn("Tabla 'users' no existe aún (se creará al iniciar el servidor)"))
        except Exception as exc:
            report["schema_users"] = {"ok": False, "error": str(exc)}
            if not use_json:
                print(_warn(f"No se pudo verificar schema: {exc}"))

    finally:
        if conn:
            await conn.close()
        if connector:
            await connector.close_async()

    if use_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print()
        print("=" * 60)
        print(_ok("Healthcheck completado — la base de datos responde"))
        print("=" * 60)
        print()


if __name__ == "__main__":
    asyncio.run(main())
