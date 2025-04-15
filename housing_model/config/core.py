import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load
import housing_model

PACKAGE_ROOT = Path(housing_model.__file__).parent.resolve()
ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / 'config.yml'

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODELS_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application configuration class.
    """
    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    """
    Model configuration class.
    """
    target: str
    features: List[str]
    var_cat: List[str]
    var_num: List[str]
    
    Id: str
    MSSubClass: str
    MSZoning:   str    
    LotFrontage: str
    LotArea: str
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: str
    OverallCond: str
    YearBuilt: str
    YearRemodAdd: str
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: str
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: str
    BsmtFinType2: str
    BsmtFinSF2: str
    BsmtUnfSF: str
    TotalBsmtSF: str
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    firstFlrSF: str
    secondFlrSF: str
    LowQualFinSF: str
    GrLivArea: str
    BsmtFullBath: str
    BsmtHalfBath: str
    FullBath: str
    HalfBath: str
    BedroomAbvGr: str
    KitchenAbvGr: str
    KitchenQual: str
    TotRmsAbvGrd: str
    Functional: str
    Fireplaces: str
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: str
    GarageFinish: str
    GarageCars: str
    GarageArea: str
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: str
    OpenPorchSF: str
    EnclosedPorch: str
    thirdSsnPorch: str
    ScreenPorch: str
    PoolArea: str
    PoolQC: str
    Fence: str
    MiscFeature: str
    MiscVal: str
    MoSold: str
    YrSold: str
    SaleType: str
    SaleCondition: str
    #SalePrice: str


    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int

class Config(BaseModel):
    """
    Main configuration class that loads and validates the configuration from a YAML file.
    """
    app_config_: AppConfig
    model_config_: ModelConfig

def find_cofnig_file() -> Path:
    """
    Locate the configuration file in the package directory.
    """
    if CONFIG_PATH.is_file():
        return CONFIG_PATH
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}.")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """
    Fetch the configuration from the YAML file.
    """
    if not cfg_path:
        cfg_path = find_cofnig_file()
    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Configuration file not found at {cfg_path}.")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    Create and validate the configuration object.
    """
    if not parsed_config:
        parsed_config = fetch_config_from_yaml()
    try:
        #print(parsed_config.data)
        _config = Config(
            app_config_=AppConfig(**parsed_config.data),
            model_config_=ModelConfig(**parsed_config.data)
        )
        return _config
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}") from e
    
config = create_and_validate_config()