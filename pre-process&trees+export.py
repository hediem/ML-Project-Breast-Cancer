import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from openpyxl import Workbook

# بارگذاری داده‌ها از فایل 'data.csv'
data = pd.read_csv("data.csv")

# تبدیل مقادیر '?' به NaN در ستون 'Bare Nuclei'
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan)
data['Bare Nuclei'] = data['Bare Nuclei'].astype(float)
data['Bare Nuclei'] = data['Bare Nuclei'].fillna(data['Bare Nuclei'].mean())

# # مثال دیگر: جایگزینی مقادیر گمشده در 'Bare Nuclei' با میانگین هر کلاس
# data['Bare Nuclei'] = data.groupby('Class')['Bare Nuclei'].transform(lambda x: x.fillna(x.mean()))

# استفاده از KNN برای پر کردن مقادیر خالی
# انتخاب ویژگی‌ها بدون ستون‌های غیرمرتبط مانند 'Sample code number' و 'Class'
# from sklearn.impute import KNNImputer
# features = data.drop(columns=['Sample code number', 'Class'])
# # ایجاد imputer با استفاده از KNN با تعداد k=5
# imputer = KNNImputer(n_neighbors=5)
# # پر کردن مقادیر خالی و جایگزینی داده‌های پرشده در دیتافریم اصلی
# data_filled = imputer.fit_transform(features)
# data.update(pd.DataFrame(data_filled, columns=features.columns))

# نرمال‌سازی ویژگی‌ها
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Sample code number', 'Class']))

# جداسازی داده‌ها به آموزش و آزمون
X = data.drop(columns=['Sample code number', 'Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# مدل درخت تصمیم
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# نمایش و ذخیره قوانین درخت تصمیم
rules = export_text(dt, feature_names=list(X.columns))
with open("decision_tree_rules.txt", "w") as f:
    f.write(rules)

# ارزیابی درخت تصمیم
y_pred = dt.predict(X_test)
dt_results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred, pos_label=2),
    "Precision": precision_score(y_test, y_pred, pos_label=2),
    "F1 Score": f1_score(y_test, y_pred, pos_label=2),
    "Classification Report": classification_report(y_test, y_pred, output_dict=True)
}

# مدل Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ارزیابی Random Forest
rf_results = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf, pos_label=2),
    "Precision": precision_score(y_test, y_pred_rf, pos_label=2),
    "F1 Score": f1_score(y_test, y_pred_rf, pos_label=2),
    "Classification Report": classification_report(y_test, y_pred_rf, output_dict=True)
}

# ذخیره خروجی‌ها در یک فایل JSON
output_data = {
    "Decision Tree": dt_results,
    "Random Forest": rf_results
}
with open("model_results.json", "w") as f:
    json.dump(output_data, f, indent=4)

# ایجاد و ذخیره جدول مقایسه‌ای در یک فایل اکسل
results_df = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_results["Accuracy"], rf_results["Accuracy"]],
    "Recall": [dt_results["Recall"], rf_results["Recall"]],
    "Precision": [dt_results["Precision"], rf_results["Precision"]],
    "F1 Score": [dt_results["F1 Score"], rf_results["F1 Score"]]
})
results_df.to_excel("comparison_results.xlsx", index=False)

# رسم و ذخیره نمودار درخت تصمیم
# رسم و ذخیره نمودار درخت تصمیم به فرمت SVG یا PDF برای کیفیت بهتر
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("decision_tree_visualization.svg", format="svg")
plt.show()
