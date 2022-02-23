from fastapi import APIRouter

router = APIRouter()

@router.get('/')
async def root():
    return {"message": "Topic Modelling API is healthy"}
