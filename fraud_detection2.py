import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc

# Step 1: Load the Kaggle Credit Card Fraud Dataset
def load_data():
    try:
        df = pd.read_csv("creditcard.csv.gz", compression='gzip')
        df.dropna(inplace=True)  # Handle missing values
        required_columns = {'Amount', 'Time', 'Class'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

# Step 2: Data Preprocessing
def preprocess_data(df):
    try:
        scaler = RobustScaler()
        df[['NormalizedAmount', 'NormalizedTime']] = scaler.fit_transform(df[['Amount', 'Time']])
        df.drop(columns=['Time', 'Amount'], inplace=True)  # Keep normalized versions
        X = df.drop(columns=['Class'])  # Features
        y = df['Class']  # Target
        return X, y, scaler
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        return None, None, None

X, y, scaler = preprocess_data(data)

# Step 3: Handle Imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.6, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 5: Train the XGBoost Model
scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])  # Adjust for class imbalance
xgb = XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.05, 
    subsample=0.9, 
    colsample_bytree=0.9, 
    scale_pos_weight=scale_pos_weight, 
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# Step 6: Evaluate Model
predictions = xgb.predict(X_test)
roc_score = roc_auc_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

print("Classification Report:")
print(classification_report(y_test, predictions))
print("ROC AUC Score:", roc_score)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Save Model
joblib.dump(xgb, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Step 7: Streamlit UI Deployment
def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.sidebar.header("🔍 User Input")
    
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV for Prediction", type=["csv"])
    if uploaded_file:
        try:
            user_data = pd.read_csv(uploaded_file)
            required_columns = {'Amount', 'Time'}
            if not required_columns.issubset(user_data.columns):
                raise ValueError(f"Missing required columns: {required_columns - set(user_data.columns)}")
            
            user_data[['NormalizedAmount', 'NormalizedTime']] = scaler.transform(user_data[['Amount', 'Time']])
            user_data.drop(columns=['Time', 'Amount'], inplace=True)
            st.write("📊 Sample of Uploaded Data:", user_data.head())
            
            if st.sidebar.button("🚀 Detect Fraud in File"):
                model = joblib.load("fraud_model.pkl")
                predictions = model.predict(user_data)
                user_data['Prediction'] = predictions
                st.write(user_data)
                st.success("✅ Fraud Detection Completed")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    user_input = []
    for col in X.columns:
        user_input.append(st.sidebar.number_input(f"{col}", min_value=-5.0, max_value=5.0, step=0.01))
    
    if st.sidebar.button("🔍 Detect Fraud Manually"):
        try:
            model = joblib.load("fraud_model.pkl")
            input_data = np.array(user_input).reshape(1, -1)
            prediction = model.predict(input_data)
            
            if prediction[0] == 1:
                st.error("🚨 Fraudulent Transaction Detected! High Risk!")
            else:
                st.success("✅ Legitimate Transaction")
        except Exception as e:
            st.error(f"Error in manual fraud detection: {e}")
    
    if st.checkbox("📜 Show Dataset Summary"):
        st.write(data.describe())
    
    if st.checkbox("📌 Show Sample Data"):
        st.write(data.head())
    
    if st.checkbox("📊 Show Fraud Distribution"):
        fig, ax = plt.subplots()
        sns.countplot(x='Class', data=data, ax=ax)
        st.pyplot(fig)
    
    if st.checkbox("📉 Show Confusion Matrix"):
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
    
    if st.checkbox("📈 Show ROC Curve"):
        fpr, tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
    
if __name__ == "__main__":
    main()