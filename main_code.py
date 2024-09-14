from base64 import standard_b64decode
from numpy.random import rand
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath, columns):
    # Load the dataset from the file and return it as a DataFrame
    train_data = pd.read_csv(filepath, header=None)  # No header in CSV
    train_data.columns = columns
    return train_data

def preprocess_data(df):
    # Perform preprocessing like scaling features
    data_without_label = df.drop(columns="target")
    y = df["target"]
    
    # Scale features
    z_scalar = StandardScaler()
    data_scaled = z_scalar.fit_transform(data_without_label)
    data_scaled = pd.DataFrame(data_scaled, columns=data_without_label.columns)
    
    return data_scaled, y, z_scalar

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    rfc = RandomForestClassifier(n_estimators=350,criterion="gini", max_depth=15, min_samples_split=5,min_samples_leaf=2,max_features='sqrt',bootstrap=True,random_state=1234 )
    rfc.fit(X_train, y_train)
    return rfc

def evaluate_model(model, X_test, y_test, scaling_model):
    # Evaluate the trained model and return the accuracy and classification report
    scaled_test = scaling_model.transform(X_test)
    scaled_test = pd.DataFrame(scaled_test, columns=X_test.columns)
    
    predicted_labels = model.predict(scaled_test)
    acc = accuracy_score(y_test, predicted_labels)
    class_rep = classification_report(y_test, predicted_labels, zero_division=1)
    
    return acc, class_rep

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"
    columns = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
        "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
        "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
        "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
        "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
        "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
        "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
        "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
        "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
        "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
        "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
        "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
        "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
        "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
        "capital_run_length_total", "target"
    ]
    
    # Load and preprocess the data
    df = load_data(data_path, columns)
    df_scaled, y, scaling_model = preprocess_data(df)
    
    # Split data into features and target
    X = df_scaled
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=43)
    print(f"the number of data in each label class = :\n{y.value_counts()}")
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test, scaling_model)
    
    print(f"Train Model Accuracy: {accuracy:.2f}")
    print(f"Train Model's Classification Report:\n{report}")
