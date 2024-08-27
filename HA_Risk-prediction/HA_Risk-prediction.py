# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import logging
import warnings
import pickle

warnings.filterwarnings('ignore')

# Initialize tqdm for progress bars
tqdm.pandas()  # For pandas progress_apply

# Load the dataset
data = pd.read_csv('HA_Risk-prediction/heart_attack_prediction_dataset.csv')

# Clean column names by stripping any leading or trailing whitespace
data.columns = data.columns.str.strip()

# Save Patient ID column for later use
patient_ids = data[['Patient ID', 'Heart Attack Risk']] if 'Patient ID' in data.columns else None

# Drop unnecessary columns if they exist
columns_to_drop = ['Country', 'Continent', 'Hemisphere']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

# Ensure that the 'Patient ID' column is retained for later use
data = data[~data['Patient ID'].isna()]  # Drop rows where 'Patient ID' is NaN

# Convert all columns to numeric where possible, coerce errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Define feature columns and target column
features = ['Age', 'Cholesterol', 'Heart Rate', 'BMI', 'Triglycerides', 
            'Exercise Hours Per Week', 'Physical Activity Days Per Week', 'Stress Level', 
            'Sedentary Hours Per Day']
target_column = 'Heart Attack Risk'

# Ensure that the target column is in the DataFrame
if target_column in data.columns:
    X = data[features]
    y = data[target_column]

    # Drop columns with all NaN values
    X = X.dropna(axis=1, how='all')

    # Handle missing values using SimpleImputer
    X = X.copy()  # Create a copy to avoid SettingWithCopyWarning
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    # Check if any remaining NaN values
    if np.any(np.isnan(X_imputed)):
        print("Warning: Data still contains NaN values.")
        print(X_imputed.isnull().sum())

    # Apply SMOTE to balance the classes in the training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_imputed, y)

    # Train-test split and feature scaling
    X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.6, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Function to evaluate and print model performance
    def evaluate_model(model, X_test, y_test, threshold=0.5):
        preds_prob = model.predict_proba(X_test)[:, 1]
        preds = (preds_prob > threshold).astype(int)
        print(f'{model.__class__.__name__} Accuracy: {accuracy_score(y_test, preds)}')
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

    # SVM
    print("Tuning SVM...")
    param_dist_svm = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.1, 10)
    }
    random_search_svm = RandomizedSearchCV(estimator=SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
                                            param_distributions=param_dist_svm,
                                            n_iter=10,  # Number of parameter settings to sample
                                            cv=5,
                                            scoring='accuracy',
                                            verbose=1,
                                            n_jobs=-1,
                                            random_state=42)
    with tqdm(total=10) as pbar:  # Number of iterations in RandomizedSearchCV
        def update(*args, **kwargs):
            pbar.update(1)
        random_search_svm.fit(X_train, y_train)
        update()
    
    svm_best_model = random_search_svm.best_estimator_

    # Gradient Boosting
    print("Tuning Gradient Boosting...")
    param_dist_gb = {
        'n_estimators': randint(100, 200),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 7)
    }
    random_search_gb = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                                           param_distributions=param_dist_gb,
                                           n_iter=10,  # Number of parameter settings to sample
                                           cv=5,
                                           scoring='accuracy',
                                           verbose=1,
                                           n_jobs=-1,
                                           random_state=42)
    with tqdm(total=10) as pbar:  # Number of iterations in RandomizedSearchCV
        def update(*args, **kwargs):
            pbar.update(1)
        random_search_gb.fit(X_train, y_train)
        update()
    
    gb_best_model = random_search_gb.best_estimator_

    # Naive Bayes
    print("Tuning Naive Bayes...")
    nb_model = GaussianNB()  # No hyperparameter tuning needed for GaussianNB
    nb_model.fit(X_train, y_train)
    
    # AdaBoost
    print("Tuning AdaBoost...")
    param_dist_ada = {
        'n_estimators': randint(50, 150),
        'learning_rate': uniform(0.01, 1.0)
    }
    random_search_ada = RandomizedSearchCV(
    estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42),
    param_distributions=param_dist_ada,
                                            n_iter=10,
                                            cv=5,
                                            scoring='accuracy',
                                            verbose=1,
                                            n_jobs=-1,
                                            random_state=42)
    with tqdm(total=10) as pbar:
        def update(*args, **kwargs):
            pbar.update(1)
        random_search_ada.fit(X_train, y_train)
        update()
    
    ada_best_model = random_search_ada.best_estimator_

    # LightGBM
    print("Tuning LightGBM...")
    param_dist_lgb = {
        'n_estimators': randint(100, 200),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 7)
    }
    random_search_lgb = RandomizedSearchCV(estimator=LGBMClassifier(class_weight='balanced', random_state=42),
                                           param_distributions=param_dist_lgb,
                                           n_iter=10,
                                           cv=5,
                                           scoring='accuracy',
                                           verbose=1,
                                           n_jobs=-1,
                                           random_state=42)
    with tqdm(total=10) as pbar:
        def update(*args, **kwargs):
            pbar.update(1)
        random_search_lgb.fit(X_train, y_train)
        update()

    lgb_best_model = random_search_lgb.best_estimator_

    # Evaluate models
    models = {
        'SVM': svm_best_model,
        'Gradient Boosting': gb_best_model,
        'Naive Bayes': nb_model,
        'AdaBoost': ada_best_model,
        'LightGBM': lgb_best_model
    }

    for model_name, model in models.items():
        print(f"{model_name} Model Evaluation:")
        evaluate_model(model, X_test, y_test, threshold=0.3)

    # Model selection based on accuracy
    best_model_name, best_model = max(models.items(), key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))
    print(f'Best model: {best_model_name} with accuracy: {accuracy_score(y_test, best_model.predict(X_test))}')

# Save the model and scaler
pickle.dump(lgb_best_model, open('lgbm_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and scaler saved successfully!")

def plot_learning_curve(model, X, y, title='Learning Curve'):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='g', alpha=0.1)

    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    
def plot_precision_recall_curve(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]  # Get probability estimates for the positive class
    precision, recall, _ = precision_recall_curve(y, y_prob)
    average_precision = average_precision_score(y, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={average_precision:.2f})')
    plt.grid()
    plt.show()
    
def plot_roc_curve(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.2f})')
    plt.grid()
    plt.show()

def plot_confusion_matrix(model, X, y, title='Confusion Matrix'):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Heart Attack', 'Heart Attack'],
                yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Plot feature importances for LightGBM
if 'LightGBM' in models:
    lgb_best_model = models['LightGBM']
    feature_importances = lgb_best_model.feature_importances_
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=X.columns)
    plt.title('Feature Importances for LightGBM')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Plot learning curves for LightGBM
if 'LightGBM' in models:
    lgb_best_model = models['LightGBM']
    plot_learning_curve(lgb_best_model, X_train, y_train)

# Plot Precision-Recall Curve for LightGBM
if 'LightGBM' in models:
    lgb_best_model = models['LightGBM']
    plot_precision_recall_curve(lgb_best_model, X_test, y_test)

# Plot ROC Curve for LightGBM    
if 'LightGBM' in models:
    lgb_best_model = models['LightGBM']
    plot_roc_curve(lgb_best_model, X_test, y_test)

# Plot Confusion Matrix for LightGBM
if 'LightGBM' in models:
    lgb_best_model = models['LightGBM']
    plot_confusion_matrix(lgb_best_model, X_test, y_test)
    
    # Print number of patients at risk and their Patient IDs
    if patient_ids is not None:
        at_risk_ids = patient_ids.loc[patient_ids[target_column] == 1, 'Patient ID']
        print(f'Number of patients at risk: {len(at_risk_ids)}')
        # Uncomment the following lines if you want to display the IDs
        # print('Patient IDs of those at risk:')
        # print(at_risk_ids.tolist())

# Check for zero variance columns
zero_var_columns = data.loc[:, data.std() == 0].columns
print(f"Zero variance columns: {zero_var_columns}")

# Drop zero variance columns if any
data = data.drop(columns=zero_var_columns)

# Visualize heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Try a simple pairplot with fewer features
sns.pairplot(data[features[:5] + [target_column]].dropna(), hue=target_column)
plt.show()

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Convert back to DataFrame to match original column names
scaled_data = pd.DataFrame(scaled_features, columns=features)

# Combine with the target column
scaled_data[target_column] = data[target_column].values

# Use a smaller subset of the data
subset_data = scaled_data.sample(n=500, random_state=42)

# Plot pairplot on the scaled subset data
sns.pairplot(subset_data[features[:5] + [target_column]], hue=target_column)
plt.show()

# Distribution graphs for each feature
for column in features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column].dropna(), kde=True)  # Dropping NA values for histplot
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
