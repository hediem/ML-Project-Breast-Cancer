import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# بارگذاری داده‌ها از فایل data.csv بدون در نظر گرفتن هدر اصلی
data = pd.read_csv('data.csv', header=None, names=[
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'
])

# تبدیل تمامی ستون‌ها (بجز Sample code number) به عددی و جایگزینی مقادیر غیرقابل تبدیل با NaN
for column in data.columns[1:]:  # بجز ستون اول که شناسه است
    data[column] = pd.to_numeric(data[column], errors='coerce')
    if data[column].isnull().sum() > 0:
        print(f"مقادیر غیرقابل تبدیل در ستون '{column}' به NaN تبدیل شدند: {data[column].isnull().sum()} مورد")
    if column != 'Class':
        # جایگزینی مقادیر NaN با میانگین در ستون‌های ویژگی
        data[column] = data[column].fillna(data[column].mean())
    else:
        # در ستون Class، جایگزینی مقادیر NaN با مد (بیشترین مقدار تکراری)
        data[column] = data[column].fillna(data[column].mode()[0])

# حذف ستون 'Sample code number' از داده‌ها برای محاسبه همبستگی
data = data.drop(columns=['Sample code number'])

# بررسی نهایی انواع داده‌ها پس از پاک‌سازی
print(data.dtypes)

# نمودار توزیع ویژگی‌ها
sns.countplot(x='Class', data=data)
plt.title('Tumor type distribution (Class)')
plt.xlabel('Class')
plt.ylabel('Number')
plt.show()

# نقشه حرارتی همبستگی
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of the correlation coefficient of features.')
plt.show()
