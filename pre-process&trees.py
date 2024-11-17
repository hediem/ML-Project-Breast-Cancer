import pandas as pd
import numpy as np

# بارگذاری داده‌ها از فایل 'data.csv'
data = pd.read_csv("data.csv")

# تبدیل مقادیر '?' به NaN در ستون 'Bare Nuclei' 
# این کار برای شناسایی مقادیر نامعتبر به عنوان داده‌های گمشده ضروری است
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan)

# تبدیل نوع داده‌ای ستون 'Bare Nuclei' به عددی
# این تبدیل برای انجام محاسبات عددی روی این ستون لازم است
data['Bare Nuclei'] = data['Bare Nuclei'].astype(float)

# # جایگزینی مقادیر گمشده در 'Bare Nuclei' با میانگین آن ستون
# # این روش، جایگزین مناسبی برای مقادیر گمشده است تا خطاهای تحلیل را کاهش دهد
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

from sklearn.preprocessing import MinMaxScaler
# نرمال‌سازی ویژگی‌ها: استفاده از MinMaxScaler برای تبدیل داده‌ها به مقیاس 0 تا 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Sample code number', 'Class']))

# جدا کردن داده‌ها به بخش‌های آموزش و آزمون
# این جداسازی با نسبت 70% برای آموزش و 30% برای آزمون انجام می‌شود
from sklearn.model_selection import train_test_split
X = data.drop(columns=['Sample code number', 'Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# اجرای مدل درخت تصمیم
from sklearn.tree import DecisionTreeClassifier, export_text
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# نمایش قوانین ایجاد شده توسط درخت تصمیم به عنوان راهنما برای فهم مدل
rules = export_text(dt, feature_names=list(X.columns))
print("Decision Tree Rules:\n", rules)

# ارزیابی مدل درخت تصمیم با معیارهایی مانند دقت و F1
from sklearn.metrics import accuracy_score, classification_report,recall_score,precision_score,f1_score
y_pred = dt.predict(X_test)
print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred))
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred))

# اجرای مدل Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ارزیابی مدل Random Forest با معیارهایی مانند دقت و F1
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# ایجاد جدول مقایسه‌ای نتایج بین مدل‌ها با استفاده از معیارهایی مانند دقت و F1
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rf)],
    "Recall": [recall_score(y_test, y_pred, pos_label=2), recall_score(y_test, y_pred_rf, pos_label=2)],
    "Precision": [precision_score(y_test, y_pred, pos_label=2), precision_score(y_test, y_pred_rf, pos_label=2)],
    "F1 score": [f1_score(y_test, y_pred, pos_label=2), f1_score(y_test, y_pred_rf, pos_label=2)],
})

# نمایش جدول مقایسه‌ای نتایج بین مدل‌ها
print("Comparison Table:\n", results)
