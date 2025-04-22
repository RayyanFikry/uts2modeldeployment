import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import pickle

class LoanPredictionModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        self.scaler = StandardScaler()

    def preprocess_data(self):
        X = self.data.drop(columns=['loan_status'])
        y = self.data['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Random Forest Accuracy: {accuracy}")

        # Simpan model pakai joblib (dengan kompresi)
        dump(self.model, 'best_rf_model.pkl', compress=3)

        # Simpan scaler seperti biasa (atau bisa pakai joblib juga jika diinginkan)
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)

# Example usage
data_path = 'Dataset_A_loan.csv'
model = LoanPredictionModel(data_path)
model.preprocess_data()
model.train_model()
model.evaluate_model()
