import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from google.cloud.sql.connector import Connector, IPTypes
import pg8000

import sqlalchemy


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    instance_connection_name = os.environ.get(
        "INSTANCE_CONNECTION_NAME") or os.environ["CLOUDSQL_INSTANCE"]
    db_user = os.environ.get("DB_USER") or os.environ["CLOUDSQL_DB_USER"]
    db_pass = os.environ.get("DB_PASS") or os.environ.get("CLOUDSQL_DB_PASS", "")
    db_name = os.environ.get("DB_NAME") or os.environ.get("CLOUDSQL_DB_NAME", "rubricai_auth")

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_file:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            key_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        connector = Connector(credentials=creds, refresh_strategy="LAZY")
    else:
        connector = Connector(refresh_strategy="LAZY")

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    # The Cloud SQL Python Connector can be used with SQLAlchemy
    # using the 'creator' argument to 'create_engine'
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # ...
    )

    # Verify the connection is successful
    try:
        with pool.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        print("Conexión a la base de datos exitosa.")
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")

    return pool


if __name__ == "__main__":
    connect_with_connector()
