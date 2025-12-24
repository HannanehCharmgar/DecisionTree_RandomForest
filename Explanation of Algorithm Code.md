# اضافه کردن کتابخانه ها
NumPy / Pandas برای پردازش و مدیریت داده‌ها

Matplotlib / Seaborn برای ترسیم نمودارها و تحلیل بصری

Scikit-learn برای:
تقسیم داده‌ها

ساخت مدل‌های یادگیری ماشین

ارزیابی عملکرد مدل‌ها(precision,recall,accuracy,f1score...)

تنظیمات ظاهری نمودارها و حذف هشدارهای غیرضروری

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
```
## بارگذاری دیتا
در این بخش دیتاست بیماری قلبی بارگذاری و بررسی اولیه انجام می‌شود.

بررسی:

خواندن فایل CSV

تعیین نام ستون‌ها

نمایش:

تعداد نمونه‌ها

تعداد ویژگی‌ها

نمونه‌ای از داده‌ها

آمار توصیفی (میانگین، انحراف معیار، مین و ماکس)
```
filename = "heart_disease.csv"

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(filename, names=column_names)

print("\n Dataset Information:")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"\nFeature names:")
for i, col in enumerate(column_names, 1):
    print(f"{i:2}. {col}")

print("\n Data Sample:")
print(df.head())
print("\n Descriptive Statistics:")
print(df.describe())
```
## پاکسازی دیتا
شناسایی مقادیر گمشده 

تبدیل داده‌ها به نوع عددی

حذف سطرهای ناقص

تبدیل متغیر هدف به فرم باینری:

0 → سالم

1 → بیمار
```
print("\n Data Cleaning:")

missing_values = df.isin(['?']).sum()
print("Missing values (?):")
print(missing_values[missing_values > 0])

df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')

initial_rows = len(df)
df = df.dropna()
final_rows = len(df)
print(f"\nRows removed: {initial_rows - final_rows}")
print(f"Remaining rows: {final_rows}")

df['target_binary'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

print(f"\n Class Distribution:")
print(df['target_binary'].value_counts())
print(f"\nPercentage with disease: {(df['target_binary'].sum() / len(df) * 100):.1f}%")
```
## تحلیل داده ( بصورت بصری)
تحلیل‌های انجام‌شده:

توزیع ویژگی‌های عددی بین افراد سالم و بیمار

بررسی ویژگی‌های دسته‌ای (مانند جنسیت و نوع درد قفسه سینه)

تحلیل همبستگی ویژگی‌ها با بیماری قلبی
```
print("\n Exploratory Data Analysis (EDA):")

def create_eda_plots(df):

    plt.rcParams.update({
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Numeric Histograms
    for i, feature in enumerate(numeric_features):
        ax = axes.flat[i]
        ax.hist(df[df['target_binary'] == 0][feature],
                bins=20, alpha=0.6, label='Healthy')
        ax.hist(df[df['target_binary'] == 1][feature],
                bins=20, alpha=0.6, label='Diseased')
        ax.set_title(f'{feature}')
        ax.set_ylabel('Count')
        if i == 0:
            ax.legend()

    # Categorical Features
    categorical_features = ['sex', 'cp', 'fbs', 'exang']
    for i, feature in enumerate(categorical_features, start=len(numeric_features)):
        ax = axes.flat[i]
        cross_tab = pd.crosstab(df[feature], df['target_binary'])
        cross_tab.plot(kind='bar', ax=ax, alpha=0.85, legend=False)
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('Count')

    # Correlation Heatmap
    ax = axes.flat[-1]
    corr = df[numeric_features + ['target_binary']].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'shrink': 0.75},
        ax=ax
    )
    ax.set_title('Correlation Matrix')

    fig.suptitle(
        'Heart Disease – Exploratory Data Analysis',
        fontsize=16,
        fontweight='bold'
    )

    plt.show()

create_eda_plots(df)
```
## آماده سازی داده برای آموزش مدل
جدا کردن ویژگی‌ها (X) و برچسب هدف (y)

تقسیم داده به:

داده آموزش (80٪)

داده تست (20٪)

حفظ نسبت کلاس‌ها (Stratified Split)
```
print("\n Preparing Data for Modeling:")

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features]
y = df['target_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data size: {X_train.shape[0]}")
print(f"Test data size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"\nClass distribution in training: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Class distribution in test: {pd.Series(y_test).value_counts().to_dict()}")
```
## ساخت و ارزیابی درخت تصمیم( به همراه رسم) 
تنظیم پارامترهای کنترل پیچیدگی درخت

آموزش مدل روی داده آموزش

پیش‌بینی داده تست

محاسبه معیارهای ارزیابی:

Accuracy

Precision

Recall

F1-score

رسم ماتریس درهم‌ریختگی
```
print("\n Decision Tree Implementation:")

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

print("\n Decision Tree Evaluation:")
print("-" * 40)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-Score: {f1_dt:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred_dt, 
                           target_names=['Healthy', 'Diseased']))

print(" Confusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Healthy', 'Predicted Diseased'],
            yticklabels=['Actual Healthy', 'Actual Diseased'])
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('Actual Values')
plt.xlabel('Model Predictions')
plt.show()
```
## ایجاد و ارزیابی جنگل تصادفی 
ترکیب چندین درخت تصمیم

کاهش بیش‌برازش( overfitting)

افزایش دقت پیش‌بینی
```
print("\n Random Forest Implementation:")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n Random Forest Evaluation:")
print("-" * 40)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred_rf, 
                           target_names=['Healthy', 'Diseased']))

print(" Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted Healthy', 'Predicted Diseased'],
            yticklabels=['Actual Healthy', 'Actual Diseased'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual Values')
plt.xlabel('Model Predictions')
plt.show()
```
## مقایسه عملکرد دو مدل
ساخت جدول مقایسه‌ای معیارها

ترسیم نمودار میله‌ای برای مقایسه بصری
در این بخش معیار هایی مانند precision,recall,accuracy,f1-score برای ارزیابی و مقایسه دو مدل حجنگل تصادفی و درخت تصمیم مورد استفاده قرار میگیرد.
```
print("\n Model Comparison:")

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Decision Tree': [accuracy_dt, precision_dt, recall_dt, f1_dt],
    'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf]
})

print(" Performance Comparison Table:")
print(comparison_df.to_string(index=False))

plt.figure(figsize=(10, 6))
x = range(len(comparison_df))
width = 0.35

plt.bar([i - width/2 for i in x], comparison_df['Decision Tree'], 
        width, label='Decision Tree', alpha=0.8)
plt.bar([i + width/2 for i in x], comparison_df['Random Forest'], 
        width, label='Random Forest', alpha=0.8)

plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('Decision Tree vs Random Forest Performance')
plt.xticks(x, comparison_df['Metric'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```
## بررسی میزان اثرگذاری ویژگی ها (feature)
استخراج اهمیت ویژگی‌ها از هر دو مدل

نمایش ۱۰ ویژگی مهم‌تر

تحلیل نقش پزشکی این ویژگی‌ها( که بین صفر و یک هست)

```
print("\n Feature Importance Analysis:")

feature_importance_rf = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(" Top 5 Important Features (Random Forest):")
print(feature_importance_rf.head(5).to_string(index=False))

feature_importance_dt = pd.DataFrame({
    'Feature': features,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n Top 5 Important Features (Decision Tree):")
print(feature_importance_dt.head(5).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(feature_importance_rf['Feature'][:10], 
            feature_importance_rf['Importance'][:10])
axes[0].set_title('Top 10 Features - Random Forest', fontsize=14)
axes[0].set_xlabel('Importance')

axes[1].barh(feature_importance_dt['Feature'][:10], 
            feature_importance_dt['Importance'][:10])
axes[1].set_title('Top 10 Features - Decision Tree', fontsize=14)
axes[1].set_xlabel('Importance')

plt.suptitle('Feature Importance in Heart Disease Diagnosis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```
## نمایش درخت تصمیم 
```
plt.figure(figsize=(16, 9))

plot_tree(
    dt_model,
    feature_names=features,
    class_names=['Healthy', 'Diseased'],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=11,
    impurity=False,
    proportion=True,
    precision=2
)

plt.title(
    'Key Decision Rules for Heart Disease',
    fontsize=16,
    pad=20
)

plt.tight_layout()
plt.show()
```
## تحلیل بر اساس نمودار ROC
نمودار ROC ابزاری برای ارزیابی عملکرد مدل‌های دسته‌بندی دودویی است و نشان می‌دهد مدل تا چه حد می‌تواند بین دو کلاس سالم و بیمار تمایز قائل شود.

در این نمودار:

محور افقی (False Positive Rate – FPR) نشان‌دهنده نسبت نمونه‌های سالمی است که به اشتباه بیمار تشخیص داده شده‌اند.

محور عمودی (True Positive Rate – TPR یا Recall) نشان‌دهنده نسبت بیماران واقعی است که به‌درستی شناسایی شده‌اند.

نقطه چین هایی که در نمودار هستند نشان دهنده تصادفی بودن عملکرد مدل است و هرچقدر از این نقطه چین مدل فاصله داشته باشد یعنی از عملکرد تصادفی به دور است.

معیار AUC مساحت زیر منحنی ROC است و معیاری خلاصه برای سنجش کیفیت مدل محسوب می‌شود.

تفسیر AUC:

0.5 → عملکرد تصادفی

0.7 – 0.8 → عملکرد قابل قبول

0.8 – 0.9 → عملکرد خوب

> 0.9 → عملکرد بسیار عالی

```
print("\n ROC Curve Analysis:")


fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(10, 8))
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2,
         label=f'Decision Tree (AUC = {roc_auc_dt:.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2,
         label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Heart Disease Diagnosis', fontsize=16)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
```
منحنی ROC مدل Random Forest بالاتر از Decision Tree قرار دارد.

مقدار AUC بزرگ‌تر نشان می‌دهد Random Forest توانایی بهتری در تفکیک بیماران مبتلا از افراد سالم