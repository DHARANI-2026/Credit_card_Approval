import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()

# Generate Synthetic Dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_redundant=2, random_state=42, class_sep=1.5)
columns = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['Approval_Status'] = y

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(df.drop('Approval_Status', axis=1), df['Approval_Status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results[name] = {"Accuracy": accuracy, "AUC": auc, "Confusion Matrix": confusion_matrix(y_test, y_pred)}
    print(f"{name}: Accuracy = {accuracy:.2f}, AUC = {auc:.2f}")

# Bar Chart for Accuracy and AUC
plt.figure(figsize=(12, 6))
accuracy_scores = [result["Accuracy"] for result in results.values()]
auc_scores = [result["AUC"] for result in results.values()]

x_labels = list(results.keys())
x = np.arange(len(x_labels))
width = 0.35

plt.bar(x - width/2, accuracy_scores, width, color='blue', label='Accuracy', alpha=0.7)
plt.bar(x + width/2, auc_scores, width, color='orange', label='AUC', alpha=0.7)
plt.xticks(x, x_labels, rotation=45)
plt.title('Model Performance (Accuracy & AUC)')
plt.ylabel('Score')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Confusion Matrix for Best Model (Random Forest)
best_model_name = max(results, key=lambda k: results[k]['AUC'])
plot_confusion_matrix(results[best_model_name]["Confusion Matrix"], best_model_name)

# Feature Importance (Random Forest)
best_model = models[best_model_name]
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.Series(best_model.feature_importances_, index=columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.show()

# ROC Curve
plot_roc_curve(models, X_test, y_test)
