# include module
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# the function of data cleaning 
def data_cleaning(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # 將 training data 與 testing data 合併
    total = pd.concat([train, test], axis = 0)

    # some features which will drop nan
    drop_list = ['PoolQC', 'Fence']
    total.drop(columns = drop_list, inplace = True)

    # some features which will fill with 'Na'
    fill_list = ['MiscFeature', 'Alley', 'FireplaceQu', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
    for feature in fill_list:
        total[feature] = total[feature].fillna('Na')

    # some features which will fill with mean
    mean_list = ['LotFrontage', 'MasVnrArea', 'MasVnrArea', 'GarageYrBlt']
    for feature in mean_list:
        total[feature] = total[feature].fillna(train[feature].mean())

    # GarageType
    feature = 'GarageType'
    mode = train[(train[feature] != 'BuiltIn') & (train[feature] != 'Attchd') & (train[feature] != 'Detchd')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # GarageQual
    feature = 'GarageQual'
    mode = train[(train[feature] != 'TA')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # GarageCond
    feature = 'GarageCond'
    mode = train[(train[feature] != 'TA')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # BsmtFinType2
    feature = 'BsmtFinType2'
    mode = train[(train[feature] != 'Unf')][feature].mode()[0]
    total[feature] = total[feature].fillna(mode)

    # MasVnrType
    feature = 'MasVnrType'
    total[feature] = total[feature].fillna('BrkCmn')

    # Electrical
    feature = 'Electrical'
    total[feature] = total[feature].fillna('FuseF')

    return total