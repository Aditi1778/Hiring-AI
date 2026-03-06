import logging

import certifi
import redis.asyncio as redis
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi

from core.config import settings

load_dotenv()

logger = logging.getLogger("OXHIRE AI")

ca = certifi.where()


class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_mongo(self):
        """Connect to MongoDB on Startup"""
        try:
            self.client = AsyncIOMotorClient(
                settings.DB_URL, server_api=ServerApi("1"), tlsCAFile=ca
            )
            self.db = self.client[settings.DB_NAME]
            await self.client.admin.command("ping")
            logger.info("✅ Connected to MongoDB!")

        except Exception as e:
            print(f"⚔️ Error connecting to MongoDB: {e}")

    async def close_mongo_connection(self):
        """Close MongoDB Connection on Shutdown"""
        self.client.close()
        logger.info("MongoDB connection closed.")


db_client = MongoDB()
