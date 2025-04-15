import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from housing_model import __version__ as _version
from housing_model.config.core import config
from housing_model.processing.data_manager import load_pipeline
#from housing_model.processing.data_manager import pre_pipeline_preparation
from housing_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = bikeshare_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {'Id': 226,
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
     '1stFlrSF': 630,
     '2ndFlrSF': 672,
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
     '3SsnPorch': 0,
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

    make_prediction(input_data = data_in)