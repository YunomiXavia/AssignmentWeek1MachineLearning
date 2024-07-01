import pandas as pd

# Thư viện tiền xử lý dữ liệu
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Thư viện train model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Thư viện đánh giá model
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Thư viện vẽ biểu đồ, trực quan hoá dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm đọc file txt và chuyển nó sang DataFrame
def read_txt_to_df(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    df = pd.DataFrame(lines, columns=['text'])
    df['label'] = label
    return df

# Đọc file
fake_news = read_txt_to_df('./clean_fake.txt', 0)  # 0 fake news
real_news = read_txt_to_df('./clean_real.txt', 1)  # 1 real news

# Kết hợp 2 tập dữ liệu lại thành 1 tập dữ liệu duy nhất
# Thiết lập lại chỉ mục: reset_index(drop=True) sẽ tạo một chỉ mục mới, liên tục bắt đầu từ 0 cho đến số lượng bản ghi trong DataFrame kết hợp. Tham số drop=True đảm bảo rằng chỉ mục cũ sẽ không được thêm vào như một cột mới trong DataFrame.
data = pd.concat([fake_news, real_news]).reset_index(drop=True)

# Hàm tiền xử lý text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

data['text'] = data['text'].apply(clean_text)

# Chia bộ dữ liệu thành bộ train và bộ test
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Tạo Pipeline cho việc train
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# Thiết lập các tham số cho Grid search
param_grid_lr = {
    'tfidf__max_features': [1000, 5000, 10000],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']
}

param_grid_nb = {
    'tfidf__max_features': [1000, 5000, 10000],
    'clf__alpha': [0.1, 0.5, 1, 5, 10]
}

param_grid_rf = {
    'tfidf__max_features': [1000, 5000, 10000],
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [4, 6, 8],
    'clf__criterion': ['gini', 'entropy']
}

# Grid search
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

grid_search_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=5, scoring='accuracy')
grid_search_nb.fit(X_train, y_train)

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Best models và parameters
best_lr = grid_search_lr.best_estimator_
best_nb = grid_search_nb.best_estimator_
best_rf = grid_search_rf.best_estimator_

print("Best Logistic Regression parameters:", grid_search_lr.best_params_)
print("Best Naive Bayes parameters:", grid_search_nb.best_params_)
print("Best Random Forest parameters:", grid_search_rf.best_params_)
print("----------------------------------------------------")

# Đánh giá model
lr_best_pred = best_lr.predict(X_test)
nb_best_pred = best_nb.predict(X_test)
rf_best_pred = best_rf.predict(X_test)

lr_best_accuracy = accuracy_score(y_test, lr_best_pred)
lr_best_recall = recall_score(y_test, lr_best_pred)
lr_best_f1 = f1_score(y_test, lr_best_pred)

nb_best_accuracy = accuracy_score(y_test, nb_best_pred)
nb_best_recall = recall_score(y_test, nb_best_pred)
nb_best_f1 = f1_score(y_test, nb_best_pred)

rf_best_accuracy = accuracy_score(y_test, rf_best_pred)
rf_best_recall = recall_score(y_test, rf_best_pred)
rf_best_f1 = f1_score(y_test, rf_best_pred)

print("Best Logistic Regression Accuracy:", lr_best_accuracy)
print("Best Logistic Regression Recall:", lr_best_recall)
print("Best Logistic Regression F1-Score:", lr_best_f1)
print("----------------------------------------------------")

print("Best Naive Bayes Accuracy:", nb_best_accuracy)
print("Best Naive Bayes Recall:", nb_best_recall)
print("Best Naive Bayes F1-Score:", nb_best_f1)
print("----------------------------------------------------")


print("Best Random Forest Accuracy:", rf_best_accuracy)
print("Best Random Forest Recall:", rf_best_recall)
print("Best Random Forest F1-Score:", rf_best_f1)
print("----------------------------------------------------")


# Tìm model tốt nhất
best_model_name = None
best_model_accuracy = 0

if lr_best_accuracy > best_model_accuracy:
    best_model_name = "Logistic Regression"
    best_model_accuracy = lr_best_accuracy

if nb_best_accuracy > best_model_accuracy:
    best_model_name = "Naive Bayes"
    best_model_accuracy = nb_best_accuracy

if rf_best_accuracy > best_model_accuracy:
    best_model_name = "Random Forest"
    best_model_accuracy = rf_best_accuracy

print(f"The best model is: {best_model_name} with accuracy: {best_model_accuracy}")

# Trực quan hoá
results = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest'],
    'Accuracy': [lr_best_accuracy, nb_best_accuracy, rf_best_accuracy],
    'Recall': [lr_best_recall, nb_best_recall, rf_best_recall],
    'F1 Score': [lr_best_f1, nb_best_f1, rf_best_f1]
}

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy')

plt.subplot(1, 3, 2)
sns.barplot(x='Model', y='Recall', data=results_df)
plt.title('Model Recall')

plt.subplot(1, 3, 3)
sns.barplot(x='Model', y='F1 Score', data=results_df)
plt.title('Model F1 Score')

plt.tight_layout()
plt.show()
