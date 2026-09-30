from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

print("Samples:", X.shape[0], "| Features:", X.shape[1], "| Classes:", len(iris.target_names))
print("Missing values:", int(X.isna().sum().sum()))
print(y.map(dict(enumerate(iris.target_names))).value_counts().to_string(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

# fit the scaler on training data only so test data doesn't leak into it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Weighted F1:", round(f1_score(y_test, predictions, average="weighted"), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\n" + classification_report(y_test, predictions, target_names=iris.target_names))

print("Error rate for different K:")
for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
    error = 1 - accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"K={k:<2} error={error:.4f}")
