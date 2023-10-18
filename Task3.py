import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load and inspect data
df = pd.read_csv('Loan_Data.csv')
print('----Data Inspection----')
print(df.head())
print(df.shape)
print(df.tail())

# Data checks
print('----Data Check----')
print(df.isnull().sum())
print(df.dtypes.unique())
print(df.duplicated().sum())

# Visualization settings
sns.set(style='whitegrid')

# Bar chart for default distribution
plt.figure(figsize=(12, 8))
sns.countplot(x='default', data=df)
plt.title('Default Distribution')
plt.savefig('default_distribution.png')

# Boxplots for outlier detection
plt.figure(figsize=(12, 8))
for idx, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, idx)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('boxplots.png')

# Histograms for feature distributions
features_df = df.drop(columns=['customer_id', 'default'])
plt.figure(figsize=(12, 8))
for idx, feature in enumerate(features_df, 1):
    plt.subplot(3, 2, idx)
    sns.histplot(features_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('histograms.png')

# Box-Cox transformation
df['total_debt_outstanding'], _ = boxcox(features_df['total_debt_outstanding'] + 1)
df['loan_amt_outstanding'], _ = boxcox(features_df['loan_amt_outstanding'] + 1)

# Plot transformed histograms
plt.figure(figsize=(12, 4))
for idx, feature in enumerate(['total_debt_outstanding', 'loan_amt_outstanding'], 1):
    plt.subplot(1, 2, idx)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('transformed_histograms.png')

# Correlation heatmaps
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
mask = np.triu(df.drop('customer_id', axis=1).corr())
sns.heatmap(df.drop('customer_id', axis=1).corr(), mask=mask, annot=True, cmap='rocket', ax=ax[0])
ax[0].set_title('Triangle Correlation Heatmap')
sns.heatmap(df.drop('customer_id', axis=1).corr()[['default']].sort_values(by='default', ascending=False), annot=True, cmap='rocket', ax=ax[1])
ax[1].set_title('Variables Correlating with Default')
plt.tight_layout()
plt.savefig('correlation_heatmaps.png')

# Constructing potential models
# Creating the target and feature variables
X = df.drop(columns=['customer_id', 'default'])
y = df['default']
X_normalized = preprocessing.StandardScaler().fit_transform(X)

# Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
log_model = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
y_pred = log_model.predict(X_test)

# Logistic Regression Model testing
print('Jaccard Score (Logistic Regression) =', jaccard_score(y_test, y_pred, pos_label=0))
print('---Logistic Regression Model---')
print(classification_report(y_test, y_pred))

# Confusion Matrix for Logistic Regression Model
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix_logistic_regression.png')

# Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print('Jaccard Score (Random Forest) =', jaccard_score(y_test, y_pred_rf, pos_label=0))
print('---Random Forest Model---')
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest Model
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix_random_forest.png')

# Constructing potential models
# Creating the target and feature variables
X = df.drop(columns=['customer_id', 'default'])
y = df['default']
X_normalized = preprocessing.StandardScaler().fit_transform(X)

# Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
log_model = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
y_pred = log_model.predict(X_test)

def expected_loss(loan_properties, model=log_model, recovery_rate=0.10):
    """
    The function returns the expected loss of a given loan.

    Parameters:
    - loan_properties (DataFrame): DataFrame containing all the following features:
                                    - Credit Lines Outstanding
                                    - Loan amt Outstanding 
                                    - Total Debt Outstanding 
                                    - Income
                                    - Years Employed 
                                    - Fico Score
    - model (Model): The trained machine learning model used to predict the default probability.
    - recovery_rate (float): The rate at which the loan amount can be recovered if a default occurs. Default is 10%.

    Returns:
    float: Expected loss value.
    """

    # Probability of default (PD)
    pd = model.predict_proba(loan_properties)[:, 1]

    # Loss given default (LGD)
    lgd = (1- recovery_rate)

    # Exposure at default (EAD)
    ead = loan_properties['loan_amt_outstanding'].values

    # Expected loss (EL)
    el = pd * lgd * ead

    return max(el[0], 0)

plt.show()