import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("Preprocessed_Airwriting_v2.csv")

#Typo
df['label'] = df['label'].replace({'FF': 'F'})

X = df.drop(columns=["label"]).values
y = df["label"].values

# Encode A-Z jadi angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Bagi data train dan test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model KNN
k = 5 
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi KNN (Preprocessing V2): {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Simpan model dan label encoder
joblib.dump(model, "knn_model_v2.pkl")
joblib.dump(le, "label_encoder_v2.pkl")
print("\n✅ Model dan encoder tersimpan sebagai knn_model_v2.pkl & label_encoder_v2.pkl")
