import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sqlalchemy import true
from main_code import load_data, preprocess_data, train_model, evaluate_model

class TestSpambaseChallenge(unittest.TestCase):
    def setUp(self):
        self.data_path = "data/spambase.csv"
        self.df = load_data(self.data_path,columns)
        self.df,self.train_targets,self.z_scaler = preprocess_data(self.df)
        self.X = self.df#.drop("target", axis=1)
        self.y = self.train_targets #self.df["target"]

    def test_load_data(self):
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)  # Check that data is loaded

    def test_preprocess_data(self):
        print("checking for missing valuess",self.assertFalse(self.df.isnull().values.any()))  # Check for missing values

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2,shuffle=True, random_state=42)

        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)
        accuracy, test_classification = evaluate_model(model, X_test, y_test,self.z_scaler)
        print(f"Test Model's Accuracy: {accuracy:.2f}")
        print("-------------------------------------------------------------------------------------------------------")
        print(f"test classification report:\n {test_classification}")
        self.assertGreater(accuracy, 0.8)  # Expecting at least 80% accuracy

if __name__ == "__main__":
        columns = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
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
        "capital_run_length_total", "target"]
        unittest.main()
    

