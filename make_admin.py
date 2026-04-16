import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from app.auth.db import get_session, users

async def make_admin(email: str):
    async with await get_session() as session:
        async with session.begin():
            result = await session.execute(
                users.update().where(users.c.email == email).values(role="admin")
            )
            if result.rowcount == 0:
                print(f"Usuario {email} no encontrado.")
            else:
                print(f"Usuario {email} ahora tiene el rol 'admin'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python make_admin.py <email>")
        sys.exit(1)
    asyncio.run(make_admin(sys.argv[1]))