def check_df(dataframe, head=8):
  print("##### Shape #####")
  print(dataframe.shape)
  print("##### Types #####")
  print(dataframe.dtypes)
  print("##### Tail #####")
  print(dataframe.tail(head))
  print("##### Head #####")
  print(dataframe.head(head))
  print("##### Null Analysis #####")
  print(dataframe.isnull().sum())
  print("##### Quantiles #####")
  print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframedir.
    cat_th : int, float
        numerik fakat kategorik değişkenler için sınıf sayısı eşik değeri
    car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf sayısı eşik değeri

    Returns
    -------
    cat_cols : liste
        kategorik değişkenler listesi
    num_cols : liste
        numerik değişkenler listesi
    cat_but_car : liste
        kategorik görünümlü kardinal değişkenler listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat değişkenler cat_cols un içerisindedir.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes != "O") and (dataframe[col].nunique() < cat_th)]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def cat_summary(dataframe, col_name, plot=False):
    import pandas as pd
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": dataframe[col_name].value_counts() * (100 / len(dataframe))}))
    print("###################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary(dataframe, numerical_col, plot=False):
    import matplotlib.pyplot as plt
    quantiles = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 0.8, 0.9]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("###########")
    if plot == True:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_mean": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe,col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else :
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
def check_outlier(dataframe, col_name, q1=0.1, q3=0.9):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else :
        return False
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
def replace_with_threholds(dataframe, col_name, q1=0.1, q3=0.9):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
def missing_values_table(dataframe, na_name=False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().any() == True]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis =1, keys = ["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
def missing_vs_target(dataframe, target, na_columns) :
    temp_df = dataframe.copy()
    for col in na_columns :
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags :
        print(pd.DataFrame({"Target_Mean" : temp_df.groupby(col)[target].mean(),
                            "Count" : temp_df.groupby(col)[target].count()}), end = "\n\n\n")
def label_encoder(dataframe, binary_col):
        le = LabelEncoder()
        dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
        return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def rare_analyzer(dataframe, target, cat_cols) :
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()), "unique values")
        print(pd.DataFrame({"Count" : dataframe[col].value_counts(),
                            "Ratio" : dataframe[col].value_counts() / len(dataframe),
                            "Target_Mean" : dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe,rare_perc) :
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels= tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = ".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size =10)
    plt.show()
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    from sklearn.model_selection import validation_curve
    train_score, test_score = validation_curve(model, X=X, y=y,param_name=param_name,
                                               param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color="b")

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color="g")

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from Guide.MiuulGuide import *

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("datasets/train.csv")
df_test = pd.read_csv("datasets/test.csv")

df.head()
df.shape
df_test.head()
df_test.shape

df.describe().T

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 15, car_th = 30)

for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)


############
###### Missing Values #####
###########

miss_cols = missing_values_table(df, na_name=True)
missing_vs_target(df, "SalePrice", miss_cols)

#### Missing value oranı yüksek featurelar ####
df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1, inplace=True)

### Garage ile alakalı missing values #####
garage = [col for col in df.columns if "Garage" in col]
df.loc[df["GarageType"].isnull(), garage]

garage_num = [col for col in garage if df[col].dtype in ["float", "int"]]
garage_cat = [col for col in garage if df[col].dtype not in ["float", "int"]]

df.loc[df["GarageType"].isnull(), garage]

for col in garage_cat:
    df[col].fillna("no_garage", inplace=True)

for col in garage_num :
    df[col].fillna(0, inplace=True)

missing_values_table(df)

### Bsmt ile alakalı missing values #####
# No_basement
bsmt = [col for col in df.columns if "Bsmt" in col]
df[bsmt]

bsmt_num = [col for col in bsmt if df[col].dtype in ["float", "int"]]
bsmt_cat = [col for col in bsmt if df[col].dtype not in ["float", "int"]]

for col in bsmt_cat :
    df[col].fillna("no_basement", inplace=True)

for col in bsmt_num :
    df[col].fillna(0, inplace=True)

### Fireplace olmayan gözlemler
df.loc[df["FireplaceQu"].isnull(), "Fireplaces"]
df["FireplaceQu"].fillna("no_fireplace", inplace=True)

### MasVnrType ve MasVnrArea aynı gözlemlerde eksik veri var.
# MasVnrType(categoric) mode a göre dolduruldu, MasVnrArea(numeric), MasVnrType kırılımında mean ile dolduruldu.
df.loc[df["MasVnrType"].isnull(), "MasVnrArea"]
df["MasVnrType"].fillna(df["MasVnrType"].mode()[0], inplace=True)
df["MasVnrArea"].fillna(df.groupby("MasVnrType")["MasVnrArea"].transform("mean"), inplace=True)

### Electrical 1 missing value var
df["Electrical"].fillna(df["Electrical"].mode()[0], inplace=True)

### LotFrontage
df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)

missing_values_table(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 15, car_th = 30)

###############
### Outlier ###
##############

for col in num_cols:
    print(col, check_outlier(df, col))

outlier_nums = [col for col in num_cols if col != "SalePrice"]

for col in outlier_nums:
    replace_with_threholds(df, col)

df.head()

#################
##### Rare Analyze ###
###################

rare_analyzer(df, "SalePrice", cat_cols)

df = rare_encoder(df, 0.01)

################
## Feature Ext ####
#################

df["New_Pool_Flag"] = [0 if i == 0 else 1 for i in df["PoolArea"]]


df[["YearBuilt", "YearRemodAdd"]]
remodel_index = [i for i in df.index if df.loc[i, "YearBuilt"] != df.loc[i, "YearRemodAdd"]]
df["New_Remodel_Flag"] = [0 if i not in remodel_index else 1 for i in df.index]
df[["YearBuilt", "YearRemodAdd","New_Remodel_Flag"]]

######## Encoding ########3

ohe_cols = [col for col in cat_cols if df[col].dtype == "O"]
df[ohe_cols]

df = one_hot_encoder(df, ohe_cols)
df.shape

########## Scaling ##########
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

scale_cols = [col for col in num_cols if "SalePrice" not in col]
for col in X[scale_cols] :
    X[col] = scaler.fit_transform(X[[col]])

#################
### Model
#################

### Catboost
catboost_model = CatBoostRegressor(random_state=17, verbose=False).fit(X, y)

cv_results = cross_validate(catboost_model, X, y, cv=5,
                            scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                     'neg_root_mean_squared_error', "r2"])

cv_results["test_neg_mean_absolute_error"].mean() #-14475
cv_results["test_neg_mean_squared_error"].mean() #650461574
cv_results["test_neg_root_mean_squared_error"].mean() #25166
cv_results["test_r2"].mean()# 0.89


catboost_params = catboost_params = {"iterations" : [200, 500, 700],
                   "learning_rate" : [0.1, 0.01],
                   "depth" : [3, 6, 9]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv = 5, n_jobs=-1, verbose=True).fit(X, y)

catboost_model.get_params()
catboost_best_grid.best_params_

catboost_final = CatBoostRegressor(iterations=700, learning_rate=0.1, depth=6, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5,
                            scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                     'neg_root_mean_squared_error', "r2"])

cv_results["test_neg_mean_absolute_error"].mean() #-14475  --- 14853
cv_results["test_neg_mean_squared_error"].mean() #650461574 ---- 716022337
cv_results["test_neg_root_mean_squared_error"].mean() #25166 --- 26416
cv_results["test_r2"].mean()# 0.89 -- 0.88

## Linear Regression

linreg_model = LinearRegression().fit(X, y)

cv_results = cross_validate(linreg_model, X, y, cv=5,
                            scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                     'neg_root_mean_squared_error', "r2"])

cv_results["test_neg_mean_absolute_error"].mean() #-19333
cv_results["test_neg_mean_squared_error"].mean() #1.138.888.144
cv_results["test_neg_root_mean_squared_error"].mean() #33175
cv_results["test_r2"].mean()# 0.82

####### LightGBM


lgbm_model = LGBMRegressor(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=5,
                            scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                     'neg_root_mean_squared_error', "r2"])

cv_results["test_neg_mean_absolute_error"].mean() #-16724
cv_results["test_neg_mean_squared_error"].mean() #841.989.069
cv_results["test_neg_root_mean_squared_error"].mean() #28.700
cv_results["test_r2"].mean()# 0.867

lgbm_params = {"learning_rate" : [0.01, 0.1],
              "n_estimators" : [100, 300, 500, 1000],
              "colsample_bytree" : [1, 0.5, 0.7]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5,
                            scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                     'neg_root_mean_squared_error', "r2"])

cv_results["test_neg_mean_absolute_error"].mean() #-16724  & 15739
cv_results["test_neg_mean_squared_error"].mean() #841.989.069 & 792.194.633
cv_results["test_neg_root_mean_squared_error"].mean() #28.700 & 27762
cv_results["test_r2"].mean()# 0.867 % 0.876

