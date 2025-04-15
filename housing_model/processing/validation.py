import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from housing_model.config.core import config

def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    #pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = input_df[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    Id: Optional[int]
    MSSubClass: Optional[int]
    MSZoning: Optional[str]
    LotFrontage: Optional[float]
    LotArea: Optional[int]
    Street: Optional[str]
    Alley: Optional[str]
    LotShape: Optional[str]
    LandContour: Optional[str]
    Utilities: Optional[str]
    LotConfig: Optional[str]
    LandSlope: Optional[str]
    Neighborhood: Optional[str]
    Condition1: Optional[str]
    Condition2: Optional[str]
    BldgType: Optional[str]
    HouseStyle: Optional[str]
    OverallQual: Optional[int]
    OverallCond: Optional[int]
    YearBuilt: Optional[int]
    YearRemodAdd: Optional[int]
    RoofStyle: Optional[str]
    RoofMatl: Optional[str]
    Exterior1st: Optional[str]
    Exterior2nd: Optional[str]
    MasVnrType: Optional[str]
    MasVnrArea: Optional[float]
    ExterQual: Optional[str]
    ExterCond: Optional[str]
    Foundation: Optional[str]
    BsmtQual: Optional[str]
    BsmtCond: Optional[str]
    BsmtExposure: Optional[str]
    BsmtFinType1: Optional[str]
    BsmtFinSF1: Optional[int]
    BsmtFinType2: Optional[str]
    BsmtFinSF2: Optional[int]
    BsmtUnfSF: Optional[int]
    TotalBsmtSF: Optional[int]
    Heating: Optional[str]
    HeatingQC: Optional[str]
    CentralAir: Optional[str]
    Electrical: Optional[str]
    LowQualFinSF: Optional[int]
    GrLivArea: Optional[int]
    BsmtFullBath: Optional[int]
    BsmtHalfBath: Optional[int]
    FullBath: Optional[int]
    HalfBath: Optional[int]
    BedroomAbvGr: Optional[int]
    KitchenAbvGr: Optional[int]
    KitchenQual: Optional[str]
    TotRmsAbvGrd : Optional[int]
    Functional: Optional[str]
    Fireplaces : Optional[int]
    FireplaceQu: Optional[str]
    GarageType: Optional[str]
    GarageYrBlt: Optional[int]
    GarageFinish: Optional[str]
    GarageCars: Optional[int]
    GarageArea: Optional[int]
    GarageQual: Optional[str]
    GarageCond: Optional[str]
    PavedDrive: Optional[str]
    WoodDeckSF: Optional[int]
    OpenPorchSF: Optional[int]
    EnclosedPorch: Optional[int]
    ScreenPorch: Optional[int]
    PoolArea: Optional[int]
    PoolQC: Optional[str]
    Fence: Optional[str]
    MiscFeature: Optional[str]
    MiscVal: Optional[int]
    MoSold: Optional[int]
    YrSold: Optional[int]
    SaleType: Optional[str]
    SaleCondition: Optional[str]
    firstFlrSF: Optional[int]
    secondFlrSF: Optional[int]
    thirdSsnPorch: Optional[int]
    # SalePrice: Optional[float]



class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]