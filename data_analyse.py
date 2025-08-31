
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, 
                           precision_score, recall_score, f1_score, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Download the dataset from Kaggle
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("devendra416/ddos-datasets")
print(f"Dataset downloaded to: {path}")

# Use the imbalanced dataset first to ensure we have both classes
TRAINING_CSV_PATH = os.path.join(path, "ddos_imbalanced", "unbalaced_20_80_dataset.csv")
print(f"Loading dataset from: {TRAINING_CSV_PATH}")

print("\n" + "="*60)
print("STEP 1: DATA EXPLORATION")
print("="*60)

# First, let's understand the dataset structure
print("Checking dataset structure...")
print("Reading first few rows to understand columns...")
sample_df = pd.read_csv(TRAINING_CSV_PATH, nrows=1000)
print(f"Sample shape: {sample_df.shape}")
print(f"Columns: {list(sample_df.columns)}")
print(f"Data types:")
print(sample_df.dtypes)

print(f"\nLabel distribution in sample:")
print(sample_df['Label'].value_counts())

print(f"\nFirst 3 rows:")
print(sample_df.head(3))

# Check the label distribution across the file
print("\nChecking label distribution across different parts of the file...")
total_lines = 7616509  # From wc -l command (minus 1 for header)

# Sample from different positions to understand data distribution
positions_to_check = [0, total_lines//4, total_lines//2, 3*total_lines//4, total_lines-1000]
label_distributions = []

for i, pos in enumerate(positions_to_check):
    if pos == 0:
        chunk = pd.read_csv(TRAINING_CSV_PATH, nrows=1000)
    else:
        chunk = pd.read_csv(TRAINING_CSV_PATH, skiprows=range(1, pos+1), nrows=1000, header=0)
        chunk.columns = sample_df.columns
    
    dist = chunk['Label'].value_counts().to_dict()
    label_distributions.append(dist)
    print(f"Position {pos:>8}: {dist}")

# Now let's create a balanced sample for analysis
print(f"\nCreating balanced sample for analysis...")

# Since we know the approximate positions, let's sample strategically
ddos_samples = []
benign_samples = []

# Get DDoS samples from the beginning
ddos_chunk = pd.read_csv(TRAINING_CSV_PATH, nrows=25000)
ddos_data = ddos_chunk[ddos_chunk['Label'].str.lower() == 'ddos']
if len(ddos_data) > 0:
    ddos_samples.append(ddos_data.sample(min(10000, len(ddos_data))))

# Get Benign samples from the end
benign_chunk = pd.read_csv(TRAINING_CSV_PATH, skiprows=range(1, 6000000), nrows=50000, header=0)
benign_chunk.columns = sample_df.columns
benign_data = benign_chunk[benign_chunk['Label'].str.lower() == 'benign']
if len(benign_data) > 0:
    benign_samples.append(benign_data.sample(min(40000, len(benign_data))))

# Combine samples
all_samples = []
if ddos_samples:
    all_samples.extend(ddos_samples)
if benign_samples:
    all_samples.extend(benign_samples)

if all_samples:
    ddos_original = pd.concat(all_samples, ignore_index=True)
    print(f"Final sample shape: {ddos_original.shape}")
    print(f"Final label distribution: {ddos_original['Label'].value_counts().to_dict()}")
else:
    print("ERROR: Could not create balanced sample!")
    exit(1)

print("\n" + "="*60)
print("STEP 2: DETAILED DATA EXPLORATION")
print("="*60)

# Analyze the dataset structure in detail
print("Dataset Information:")
print(f"- Shape: {ddos_original.shape}")
print(f"- Columns: {ddos_original.shape[1]}")
print(f"- Memory usage: {ddos_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check data types and missing values
print(f"\nMissing values analysis:")
missing_values = ddos_original.isnull().sum()
missing_percent = (missing_values / len(ddos_original)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing %': missing_percent.values
}).sort_values('Missing Count', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0].head(10))

# Analyze features that might cause data leakage
print(f"\nAnalyzing potential problematic features:")
problematic_features = ['Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
feature_analysis = {}

for feature in problematic_features:
    if feature in ddos_original.columns:
        unique_count = ddos_original[feature].nunique()
        unique_ratio = unique_count / len(ddos_original)
        
        print(f"\n{feature}:")
        print(f"  - Unique values: {unique_count}")
        print(f"  - Uniqueness ratio: {unique_ratio:.4f}")
        
        if unique_ratio > 0.5:
            print(f"  - WARNING: High uniqueness ratio - potential ID column!")
            feature_analysis[feature] = 'high_cardinality'
        elif unique_ratio < 0.01:
            print(f"  - WARNING: Very low uniqueness - might be constant!")
            feature_analysis[feature] = 'low_cardinality'
        else:
            feature_analysis[feature] = 'normal'
            
        # Show sample values
        print(f"  - Sample values: {list(ddos_original[feature].unique()[:5])}")

# Analyze label distribution by different groups
print(f"\nLabel distribution analysis:")
print(f"Overall distribution:")
print(ddos_original['Label'].value_counts(normalize=True))

# Check if IP addresses or timestamps correlate too strongly with labels
if 'Src IP' in ddos_original.columns:
    print(f"\nTop 10 Source IPs by label:")
    ip_label_dist = ddos_original.groupby(['Src IP', 'Label']).size().unstack(fill_value=0)
    print(ip_label_dist.head(10))

if 'Timestamp' in ddos_original.columns:
    print(f"\nSample timestamps by label:")
    for label in ddos_original['Label'].unique():
        sample_timestamps = ddos_original[ddos_original['Label'] == label]['Timestamp'].head(3).tolist()
        print(f"  {label}: {sample_timestamps}")

print(f"\n" + "="*60)
print("STEP 3: DATA PREPROCESSING")
print("="*60)

print(f"\n" + "="*60)
print("STEP 3: DATA PREPROCESSING")
print("="*60)

# Create a copy for processing
ddos_processed = ddos_original.copy()

# Step 1: Remove problematic features that could cause data leakage
print("Step 1: Removing potentially problematic features...")

# Features to definitely remove (IDs, timestamps, etc.)
features_to_remove = []

# Remove row index if present
if 'Unnamed: 0' in ddos_processed.columns:
    features_to_remove.append('Unnamed: 0')
    print(f"  - Removing 'Unnamed: 0' (row index)")

# Handle Flow ID - too unique, likely causes overfitting
if 'Flow ID' in ddos_processed.columns:
    unique_ratio = ddos_processed['Flow ID'].nunique() / len(ddos_processed)
    if unique_ratio > 0.5:
        features_to_remove.append('Flow ID')
        print(f"  - Removing 'Flow ID' (uniqueness ratio: {unique_ratio:.4f})")

# Handle IP addresses - encode them differently to avoid overfitting
ip_columns = ['Src IP', 'Dst IP']
ip_features_to_add = []

for ip_col in ip_columns:
    if ip_col in ddos_processed.columns:
        # Instead of using raw IPs, create network-based features
        print(f"  - Processing {ip_col}...")
        
        # Count of unique IPs per label (this is less specific than the IP itself)
        ip_counts = ddos_processed[ip_col].value_counts()
        ddos_processed[f'{ip_col}_frequency'] = ddos_processed[ip_col].map(ip_counts)
        ip_features_to_add.append(f'{ip_col}_frequency')
        
        # Is it a private IP? (basic network feature)
        private_ip_mask = ddos_processed[ip_col].str.startswith(('192.168.', '10.', '172.'))
        ddos_processed[f'{ip_col}_is_private'] = private_ip_mask.astype(int)
        ip_features_to_add.append(f'{ip_col}_is_private')
        
        # Remove the original IP column
        features_to_remove.append(ip_col)

print(f"  - Added network features: {ip_features_to_add}")

# Handle Timestamp - convert to useful temporal features
if 'Timestamp' in ddos_processed.columns:
    print(f"  - Processing Timestamp...")
    
    # Try to parse timestamps
    try:
        ddos_processed['Timestamp_parsed'] = pd.to_datetime(ddos_processed['Timestamp'])
        
        # Extract useful temporal features (but not too specific)
        ddos_processed['Hour'] = ddos_processed['Timestamp_parsed'].dt.hour
        ddos_processed['DayOfWeek'] = ddos_processed['Timestamp_parsed'].dt.dayofweek
        
        # Remove parsed timestamp and original
        features_to_remove.extend(['Timestamp', 'Timestamp_parsed'])
        print(f"  - Added temporal features: Hour, DayOfWeek")
        
    except:
        print(f"  - Could not parse timestamps, removing column")
        features_to_remove.append('Timestamp')

# Remove identified problematic features
ddos_processed = ddos_processed.drop(columns=features_to_remove, errors='ignore')
print(f"  - Removed features: {features_to_remove}")

print(f"Shape after feature removal: {ddos_processed.shape}")

# Step 2: Handle missing values
print(f"\nStep 2: Handling missing values...")
missing_counts = ddos_processed.isnull().sum()
columns_with_missing = missing_counts[missing_counts > 0]

if len(columns_with_missing) > 0:
    print(f"Columns with missing values:")
    for col, count in columns_with_missing.items():
        percent = (count / len(ddos_processed)) * 100
        print(f"  - {col}: {count} ({percent:.2f}%)")
    
    # Fill missing values
    numeric_cols = ddos_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = ddos_processed.select_dtypes(include=['object']).columns
    categorical_cols = categorical_cols.drop('Label', errors='ignore')
    
    # Fill numeric columns with median
    for col in numeric_cols:
        if ddos_processed[col].isnull().sum() > 0:
            median_val = ddos_processed[col].median()
            ddos_processed[col] = ddos_processed[col].fillna(median_val)
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        if ddos_processed[col].isnull().sum() > 0:
            mode_val = ddos_processed[col].mode()[0] if not ddos_processed[col].mode().empty else 'Unknown'
            ddos_processed[col] = ddos_processed[col].fillna(mode_val)
            
    print(f"  - Filled missing values")
else:
    print(f"  - No missing values found")

# Step 3: Process target variable
print(f"\nStep 3: Processing target variable...")
print(f"Original label distribution:")
original_dist = ddos_processed['Label'].value_counts()
print(original_dist)

# Standardize labels to binary
def standardize_label(label):
    if isinstance(label, str):
        label_lower = label.lower().strip()
        if 'benign' in label_lower or 'normal' in label_lower:
            return 'Benign'
        else:
            return 'DDoS'
    else:
        return 'Benign' if label == 0 else 'DDoS'

ddos_processed['Label'] = ddos_processed['Label'].apply(standardize_label)

print(f"\nStandardized label distribution:")
new_dist = ddos_processed['Label'].value_counts()
print(new_dist)

# Step 4: Prepare features and target
print(f"\nStep 4: Preparing features and target...")
y = ddos_processed['Label'].copy()
X = ddos_processed.drop(columns=['Label']).copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Step 5: Handle categorical features
print(f"\nStep 5: Encoding categorical features...")
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

if len(categorical_features) > 0:
    print(f"Categorical features found: {categorical_features}")
    
    for col in categorical_features:
        unique_count = X[col].nunique()
        print(f"  - {col}: {unique_count} unique values")
        
        # Use label encoding for categorical features
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    print(f"  - Encoded all categorical features")
else:
    print(f"  - No categorical features found")

# Step 6: Handle infinite and extreme values
print(f"\nStep 6: Handling infinite and extreme values...")
inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
if inf_counts > 0:
    print(f"  - Found {inf_counts} infinite values, replacing with NaN")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

# Step 7: Feature scaling
print(f"\nStep 7: Applying feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Encode target variable
print(f"\nStep 8: Encoding target variable...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
print(f"Label mapping: {label_mapping}")

print(f"\nFinal processed data:")
print(f"  - Features shape: {X_scaled.shape}")
print(f"  - Target shape: {y_encoded.shape}")
print(f"  - Feature names: {list(X.columns)}")

print(f"\n" + "="*60)
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("="*60)

print(f"\n" + "="*60)
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("="*60)

# Train-test split
print("Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Check class distribution
train_dist = np.bincount(y_train)
test_dist = np.bincount(y_test)

print(f"\nClass distribution:")
for i, label in enumerate(label_encoder.classes_):
    train_pct = (train_dist[i] / len(y_train)) * 100
    test_pct = (test_dist[i] / len(y_test)) * 100
    print(f"  {label}: Train {train_dist[i]} ({train_pct:.1f}%), Test {test_dist[i]} ({test_pct:.1f}%)")

# Model evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{model_name} Results:")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Benign → Benign):   {tn}")
    print(f"  False Positives (Benign → DDoS):    {fp}")
    print(f"  False Negatives (DDoS → Benign):    {fn}")  
    print(f"  True Positives (DDoS → DDoS):       {tp}")
    
    # Calculate additional metrics
    if tp + fp > 0:
        precision_manual = tp / (tp + fp)
    else:
        precision_manual = 0
        
    if tp + fn > 0:
        recall_manual = tp / (tp + fn)
    else:
        recall_manual = 0
    
    print(f"\nAdditional Analysis:")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    # ROC AUC if probabilities available
    roc_auc = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"  ROC AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'roc_auc': roc_auc, 'confusion_matrix': cm, 'predictions': y_pred, 
        'probabilities': y_prob
    }

# Model 1: Decision Tree with regularization
print(f"\nTraining Decision Tree with regularization...")
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,           # Limit tree depth
    min_samples_split=200, # Need more samples to split
    min_samples_leaf=100,  # Need more samples in leaf
    max_features='sqrt',   # Use subset of features
    class_weight='balanced'
)

dt_model.fit(X_train, y_train)
dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

# Model 2: Random Forest with regularization  
print(f"\nTraining Random Forest with regularization...")
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=50,       # Fewer trees
    max_depth=10,          # Limit depth
    min_samples_split=200,
    min_samples_leaf=100,
    max_features='sqrt',   # Use subset of features
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Model 3: XGBoost with strong regularization
print(f"\nTraining XGBoost with strong regularization...")
xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=100,      # Moderate number of estimators
    max_depth=6,           # Limit depth
    learning_rate=0.05,    # Lower learning rate
    subsample=0.7,         # Use 70% of samples
    colsample_bytree=0.7,  # Use 70% of features
    reg_alpha=1.0,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    scale_pos_weight=1,    # Handle imbalance
    eval_metric='logloss',
    verbosity=0
)

xgb_model.fit(X_train, y_train)
xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

# Results comparison
print(f"\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [dt_results['accuracy'], rf_results['accuracy'], xgb_results['accuracy']],
    'Precision': [dt_results['precision'], rf_results['precision'], xgb_results['precision']],
    'Recall': [dt_results['recall'], rf_results['recall'], xgb_results['recall']],
    'F1-Score': [dt_results['f1'], rf_results['f1'], xgb_results['f1']],
    'ROC AUC': [dt_results['roc_auc'], rf_results['roc_auc'], xgb_results['roc_auc']]
})

print(results_df.round(4))

# Find best model
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_f1_score = results_df.loc[best_model_idx, 'F1-Score']

print(f"\nBest Model: {best_model_name} (F1-Score: {best_f1_score:.4f})")

# Feature importance analysis
print(f"\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if hasattr(xgb_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 15 Most Important Features (XGBoost):")
    print(feature_importance.head(15).to_string(index=False))
    
    # Check for suspicious features
    print(f"\nFeature Analysis:")
    suspicious_features = []
    
    for idx, row in feature_importance.head(10).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        if importance > 0.15:  # High importance threshold
            suspicious_features.append(feature)
            print(f"  ⚠️  {feature}: {importance:.4f} (HIGH - check for data leakage)")
        else:
            print(f"  ✓  {feature}: {importance:.4f}")
    
    if suspicious_features:
        print(f"\n⚠️  WARNING: Features with suspiciously high importance detected:")
        print(f"   {suspicious_features}")
        print(f"   These might indicate data leakage or overfitting.")

# Final warnings and recommendations
print(f"\n" + "="*60)
print("ANALYSIS SUMMARY & RECOMMENDATIONS")
print("="*60)

max_accuracy = max([dt_results['accuracy'], rf_results['accuracy'], xgb_results['accuracy']])

if max_accuracy > 0.99:
    print("🔴 CRITICAL WARNING: Accuracy > 99%")
    print("   This strongly suggests data leakage or overfitting!")
    print("   Recommendations:")
    print("   1. Remove more specific features (IPs, timestamps)")
    print("   2. Use time-based train/test split")
    print("   3. Test on completely different data")
    print("   4. Check feature correlation with target")
    
elif max_accuracy > 0.95:
    print("🟡 WARNING: Very high accuracy (>95%)")
    print("   This might indicate mild overfitting.")
    print("   Recommendations:")
    print("   1. Increase regularization")
    print("   2. Use cross-validation") 
    print("   3. Test with different feature sets")
    
else:
    print("✅ Model performance appears reasonable")
    print("   The models show good but not suspiciously perfect performance.")

print(f"\nDataset characteristics:")
print(f"  - Original samples: {len(ddos_original)}")
print(f"  - Features used: {X_scaled.shape[1]}")
print(f"  - Class balance: {(y_encoded.sum() / len(y_encoded)):.2%} DDoS")

print(f"\nNext steps for production:")
print(f"  1. Validate on different network environments")
print(f"  2. Test against new attack types")
print(f"  3. Monitor for concept drift")
print(f"  4. Implement real-time feature computation")
