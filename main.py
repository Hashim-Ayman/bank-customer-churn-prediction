from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from utils.inference import predict_new
from utils.config import (APP_NAME,VERSION,preprocessor,forest_model,xgboost_model,SECRET_KEY_TOKEN,)
from fastapi.middleware.cors import CORSMiddleware
from utils.CustomerData import CustomerData

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/", tags=["General"])
async def home():
    return {"message": f"Welcome to {APP_NAME} API v{VERSION}"}


api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403, detail="You are not authorized to use API")
    return api_key


@app.post("/predict/forest", tags=["Models"])
async def predict_forest(
    data: CustomerData, api_key: str = Depends(verify_api_key)
) -> dict:

    try:
        result = predict_new(data=data, preprocessor=preprocessor, model=forest_model)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/xgboost", tags=["Models"])
async def predict_xgboost(
    data: CustomerData,
) -> dict:

    try:
        result = predict_new(data=data, preprocessor=preprocessor, model=xgboost_model)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
