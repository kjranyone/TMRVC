"""GET/POST /characters endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from tmrvc_core.dialogue_types import CharacterProfile
from tmrvc_serve.schemas import CharacterCreateRequest, CharacterInfo

router = APIRouter()


@router.get("/characters", response_model=list[CharacterInfo])
async def list_characters() -> list[CharacterInfo]:
    from tmrvc_serve.app import _characters

    return [
        CharacterInfo(
            id=cid,
            name=c.name,
            personality=c.personality,
            voice_description=c.voice_description,
            language=c.language,
        )
        for cid, c in _characters.items()
    ]


@router.post("/characters", response_model=CharacterInfo)
async def create_character(req: CharacterCreateRequest) -> CharacterInfo:
    from tmrvc_serve.app import _characters

    if req.id in _characters:
        raise HTTPException(status_code=409, detail=f"Character '{req.id}' already exists.")

    speaker_file = None
    if req.speaker_file:
        sp = Path(req.speaker_file).resolve()
        # Validate: must end with .tmrvc_speaker and not traverse outside models/
        if not sp.name.endswith(".tmrvc_speaker"):
            raise HTTPException(status_code=400, detail="Speaker file must be .tmrvc_speaker")
        if not sp.exists():
            raise HTTPException(status_code=400, detail=f"Speaker file not found: {sp.name}")
        speaker_file = sp

    profile = CharacterProfile(
        name=req.name,
        personality=req.personality,
        voice_description=req.voice_description,
        language=req.language,
        speaker_file=speaker_file,
    )
    _characters[req.id] = profile

    return CharacterInfo(
        id=req.id,
        name=req.name,
        personality=req.personality,
        voice_description=req.voice_description,
        language=req.language,
    )
