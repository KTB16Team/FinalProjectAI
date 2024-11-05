import json

from fastapi import APIRouter
from sqlalchemy import text

from app.db.database import engine
from app.models.info import info, voice_info
from app.services.situation_summary import situation_summary_GPT,stt_model,generate_response
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn")

@router.post("/infos/voice")
async def get_voice(link: str = "") -> voice_info:
    logger.info("get_infos start")
    logger.info(f"message : {link}")

    entities = json.loads(stt_model(link))
    logger.info(f"entities : {entities}")

    response: voice_info = voice_info(**json.loads(generate_response(entities)))
    logger.info(f"response : {response}")

    return response

@router.post("/infos")
async def get_infos(message: str = "") -> info:
    logger.info("get_infos start")
    logger.info(f"message : {message}")

    entities = json.loads(situation_summary_GPT(message))
    logger.info(f"entities : {entities}")

    response: info = info(**json.loads(generate_response(entities)))
    logger.info(f"response : {response}")

    return response

async def read_user_info():
    query = text("SELECT user_info FROM user")
    async with engine.connect() as conn:
        result = await conn.execute(query)
        user_info = [row[0] for row in result.fetchall()]
    return user_info
