import matplotlib
matplotlib.use("Agg")  # ekranda açma, dosya olarak kaydet
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

# 1) Veri setini oku
df = pd.read_csv("data/diabetic_data.csv")

# 2) ? işaretlerini NaN yap
df = df.replace("?", np.nan)

# 3) 30 gün içinde tekrar yatış hedef değişkeni
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

print(df.head())

# Gereksiz kolonları at
cols_to_drop = [
    "encounter_id","patient_nbr","weight",
    "payer_code","medical_specialty"
]

df = df.drop(columns=cols_to_drop)

# İlaç sütunları
med_cols = [
    "metformin","repaglinide","nateglinide","chlorpropamide",
    "glimepiride","acetohexamide","glipizide","glyburide",
    "tolbutamide","pioglitazone","rosiglitazone","acarbose",
    "miglitol","troglitazone","tolazamide","examide",
    "citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"
]

# Kategorik kolonlar
cat_cols = [
    "race","gender","age",
    "admission_type_id","discharge_disposition_id","admission_source_id",
    "max_glu_serum","A1Cresult","change","diabetesMed"
] + med_cols

# Bazı ID kolonlarını kategorik yap
for col in ["admission_type_id","discharge_disposition_id","admission_source_id"]:
    df[col] = df[col].astype(str)

# Numerik kolonlar
num_cols = [
    "time_in_hospital","num_lab_procedures","num_procedures",
    "num_medications","number_outpatient","number_emergency",
    "number_inpatient","number_diagnoses"
]

# Feature ve target ayır
feature_cols = cat_cols + num_cols
X = df[feature_cols]
y = df["readmit_30"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Numerik işlemler
numeric_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Kategorik işlemler
categorical_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Transform
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# XGBoost modeli
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=5,   # dengesiz veriyi dengelemek için
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


# Pipeline (preprocess + SMOTE + model)
clf = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb_model)
])

# Modeli eğit
clf.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

y_proba = clf.predict_proba(X_test)[:, 1]

# Threshold = 0.3 (pozitifleri daha fazla yakala)
y_pred = (y_proba >= 0.3).astype(int)


print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC :", average_precision_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

import shap

# Modelin içindeki XGBoost
xgb_model_only = clf.named_steps["model"]

# SHAP örnek verisi (hız için ilk 500 satır)
sample_data = X_test[:500]

# Önişleme uygula
processed_sample = clf.named_steps["preprocess"].transform(sample_data)

explainer = shap.TreeExplainer(xgb_model_only)
shap_values = explainer.shap_values(processed_sample)

# Şekil oluştur
plt.figure()
shap.summary_plot(shap_values, processed_sample, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
print("SHAP grafiği kaydedildi: shap_summary.png")

import joblib
joblib.dump(clf, "models/readmission_xgb.pkl")
print("Model kaydedildi: models/readmission_xgb.pkl")
