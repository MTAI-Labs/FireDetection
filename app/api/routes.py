from fastapi import APIRouter
from app.api.endpoints import users, items, fire_detection

api_router = APIRouter()

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(fire_detection.router, prefix="/fire-detection", tags=["fire-detection"])
