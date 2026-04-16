import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("PASSLIB_BUILTIN_BCRYPT", "enabled")

from app.auth.service import hash_password

EMAIL = "rodrigo.diaz.o@uchile.cl"
PASSWORD = EMAIL


async def main() -> None:
    print(f"DB_TYPE={os.getenv('DB_TYPE')}")
    conn = None
    connector = None
    try:
        if os.getenv("DB_TYPE", "local").lower() == "cloudsql":
            conn, connector = await _connect_cloudsql()
        else:
            conn, connector = await _connect_local()

        row = await conn.fetchrow(
            "SELECT email FROM users WHERE email = $1",
            EMAIL,
        )
        if row:
            print(f"Usuario ya existe: {row['email']}")
            return

        created = await conn.fetchrow(
            """
            INSERT INTO users (email, name, provider, hashed_password)
            VALUES ($1, $2, 'local', $3)
            RETURNING email
            """,
            EMAIL,
            "Rodrigo Diaz",
            hash_password(PASSWORD),
        )
        print(f"Usuario creado: {created['email']}")
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
        print("Error: CLOUDSQL_INSTANCE no esta configurada", file=sys.stderr)
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
