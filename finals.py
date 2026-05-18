import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the dataset
df = pd.read_csv('ai4i2020.csv')

# 2. Drop irrelevant identifiers
df_cleaned = df.drop(columns=['UDI', 'Product ID'])

# 3. Encode the categorical 'Type' column
le = LabelEncoder()
df_cleaned['Type'] = le.fit_transform(df_cleaned['Type'])

# ==========================================
# NEW: CREATE THE MAINTENANCE DECISION LOGIC
# ==========================================
def determine_action(row):
    # Catastrophic failures = Replace (Class 2)
    if row['PWF'] == 1 or row['OSF'] == 1 or row['RNF'] == 1:
        return 2 
    # Wear and tear = Maintain (Class 1)
    elif row['TWF'] == 1 or row['HDF'] == 1:
        return 1    
    # Healthy machine = Do nothing (Class 0)
    else:
        return 0 

# Apply the rule to create our new target column
df_cleaned['Maintenance_Action'] = df_cleaned.apply(determine_action, axis=1)
# ==========================================


# 4. Define Features (X) and Target (y)
# X contains all operational parameters
X = df_cleaned[['Type', 'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]

# y now contains the 3 Maintenance Actions instead of the binary failure
y = df_cleaned['Maintenance_Action']

# 5. Split the dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")


# 6. Initialize the model with class_weight='balanced'
rf_balanced = RandomForestClassifier(
    n_estimators=100, 
    criterion='gini', 
    class_weight='balanced', # Still perfectly handles our 3 imbalanced classes!
    random_state=42
)

# 7. Train the new model
print("Training Balanced Random Forest for Maintenance Decisions")
rf_balanced.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred_bal = rf_balanced.predict(X_test)

# Map the numerical classes back to readable text for our report
action_labels = ['No Maintenance (0)', 'Maintenance Required (1)', 'Replacement Required (2)']

print("\n--- Final Multi-Class F1-Scores ---")
print(classification_report(y_test, y_pred_bal, target_names=action_labels))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_bal))


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -------------------------------------------------------------
# PLOT 1: Confusion Matrix Heatmap
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_bal)

# Plot heatmap 
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=False,
    xticklabels=action_labels, 
    yticklabels=action_labels,
    ax=axes[0],
    annot_kws={"size": 13, "weight": "bold"}
)

axes[0].set_title('Confusion Matrix: Predictive Performance', fontsize=14, pad=15, weight='bold')
axes[0].set_xlabel('Predicted Action', fontsize=12, labelpad=10)
axes[0].set_ylabel('True Action', fontsize=12, labelpad=10)
axes[0].tick_params(axis='both', which='major', labelsize=10)

# -------------------------------------------------------------
# PLOT 2: Feature Importance Bar Chart
# -------------------------------------------------------------

importances = rf_balanced.feature_importances_
feature_names = X.columns


indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]


sns.barplot(
    x=sorted_importances, 
    y=sorted_features, 
    palette='viridis', 
    ax=axes[1]
)

axes[1].set_title('Feature Importance: What Drives the Decisions?', fontsize=14, pad=15, weight='bold')
axes[1].set_xlabel('Relative Importance Score', fontsize=12, labelpad=10)
axes[1].set_ylabel('Operational Parameters', fontsize=12, labelpad=10)
axes[1].tick_params(axis='both', which='major', labelsize=10)


for i, v in enumerate(sorted_importances):
    axes[1].text(v + 0.01, i, f"{v:.2%}", va='center', fontsize=10, weight='semibold')

plt.tight_layout()
plt.show()

