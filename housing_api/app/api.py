import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from housing_model import __version__ as model_version
from housing_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input ={
     "inputs": [
         {'Id': 226,
     'MSSubClass': 160,
     'MSZoning': 'RM',
     'LotFrontage': 21.0,
     'LotArea': 1680,
     'Street': 'Pave',
     'Alley': "nan",
     'LotShape': 'Reg',
     'LandContour': 'Lvl',
     'Utilities': 'AllPub',
     'LotConfig': 'Inside',
     'LandSlope': 'Gtl',
     'Neighborhood': 'BrDale',
     'Condition1': 'Norm',
     'Condition2': 'Norm',
     'BldgType': 'Twnhs',
     'HouseStyle': '2Story',
     'OverallQual': 5,
     'OverallCond': 5,
     'YearBuilt': 1971,
     'YearRemodAdd': 1971,
     'RoofStyle': 'Gable',
     'RoofMatl': 'CompShg',
     'Exterior1st': 'HdBoard',
     'Exterior2nd': 'HdBoard',
     'MasVnrType': 'BrkFace',
     'MasVnrArea': 142.0,
     'ExterQual': 'TA',
     'ExterCond': 'TA',
     'Foundation': 'CBlock',
     'BsmtQual': 'TA',
     'BsmtCond': 'TA',
     'BsmtExposure': 'No',
     'BsmtFinType1': 'Unf',
     'BsmtFinSF1': 0,
     'BsmtFinType2': 'Unf',
     'BsmtFinSF2': 0,
     'BsmtUnfSF': 630,
     'TotalBsmtSF': 630,
     'Heating': 'GasA',
     'HeatingQC': 'TA',
     'CentralAir': 'Y',
     'Electrical': 'SBrkr',
     'firstFlrSF': 630,
     'secondFlrSF': 672,
     'LowQualFinSF': 0,
     'GrLivArea': 1302,
     'BsmtFullBath': 0,
     'BsmtHalfBath': 0,
     'FullBath': 2,
     'HalfBath': 1,
     'BedroomAbvGr': 3,
     'KitchenAbvGr': 1,
     'KitchenQual': 'TA',
     'TotRmsAbvGrd': 6,
     'Functional': 'Typ',
     'Fireplaces': 0,
     'FireplaceQu': "nan",
     'GarageType': 'Detchd',
     'GarageYrBlt': 1991.0,
     'GarageFinish': 'Unf',
     'GarageCars': 1,
     'GarageArea': 280,
     'GarageQual': 'TA',
     'GarageCond': 'TA',
     'PavedDrive': 'Y',
     'WoodDeckSF': 0,
     'OpenPorchSF': 0,
     'EnclosedPorch': 0,
     'thirdSsnPorch': 0,
     'ScreenPorch': 0,
     'PoolArea': 0,
     'PoolQC': "nan",
     'Fence': "nan",
     'MiscFeature': "nan",
     'MiscVal': 0,
     'MoSold': 5,
     'YrSold': 2009,
     'SaleType': 'COD',
     'SaleCondition': 'Abnorml',
    }
    ]}



@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    House price prediction with the housing_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
