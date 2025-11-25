# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

#Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

#Categorical and numerical columns
categorical = ["gender", "smoking_history"]

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
#Identify categorical feature indice for Catboost
cat_features = [X.columns.get_loc(col) for col in categorical]

#Train catboost model
clf = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=42)
clf.fit(X_train, y_train, cat_features=cat_features)

#Save model
joblib.dump(clf,"diabetes_catboost_model.pkl")

#Predict
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

#Metrics
print("Accuracy:", accuracy_score(y_test,y_pred))
print("ROC AUC:", roc_auc_score(y_test,y_prob))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# Fairness check by gender
metric_frame = MetricFrame(metrics={
    "accuracy": accuracy_score,
    "selection_rate": selection_rate
}, y_true=y_test, y_pred=y_pred, sensitive_features=X_test["gender"])

print("Fairness metrics by gender:\n", metric_frame.by_group)

# SHAP analysis using TreeExplainer (works great with CatBoost)
print("\nRunning SHAP analysis with CatBoost + TreeExplainer...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

#SHAP Summary plot
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.clf()

#Shap Dependence Plot
shap.dependence_plot("age", shap_values, X_test, show=False)
plt.clf()

