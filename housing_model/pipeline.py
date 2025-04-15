import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from housing_model.config.core import config
from housing_model.processing.features import TemporalVariableTransformer

from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer

from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder

from feature_engine.selection import DropFeatures
import xgboost as xgb

price_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute numerical variables with the ArbitraryNumberImputer
    ('ArbitraryNumber_imputation', ArbitraryNumberImputer( arbitrary_number=-1, variables='LotFrontage' )),

     # impute numerical variables with the mostfrequent
    ('frequentNumber_imputation', CategoricalImputer(imputation_method='frequent', variables=config.model_config_.var_num, ignore_format=True)),

    # impute categorical variables with string missing
    ('missing_imputation', CategoricalImputer(imputation_method='missing', variables=config.model_config_.var_cat, ignore_format=True)),

    # == TEMPORAL VARIABLES ====
    ('elapsed_time', TemporalVariableTransformer(
        variables=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], reference_variable='YrSold')),

    ('drop_features', DropFeatures(features_to_drop=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'])),

      # == CATEGORICAL ENCODING
    ('rare_label_encoder', RareLabelEncoder(tol=0.01, n_categories=5, variables=config.model_config_.var_cat)),

    # encode categorical and discrete variables using the target mean
    ('categorical_encoder', OrdinalEncoder(encoding_method='ordered', variables=config.model_config_.var_cat)), #
    
    ('model_rf', xgb.XGBRegressor(n_estimators=100,max_depth=7,eta=0.1,subsample=0.7,colsample_bytree=0.8,objective='reg:squarederror',random_state=0))
])