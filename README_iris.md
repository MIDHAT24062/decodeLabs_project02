# Iris Classification (KNN)

DecodeLabs AI Project 2. Classifies iris flowers into 3 species using K-Nearest Neighbors.

## Run

```
pip install scikit-learn pandas
python iris_classifier.py
```

## Steps

1. Load the Iris dataset (150 samples, 4 features, 3 balanced classes)
2. Split 80/20 with shuffling and stratification
3. Scale features with `StandardScaler` (fit on training data only)
4. Train `KNeighborsClassifier` with K=5
5. Evaluate with accuracy, confusion matrix and F1 score
6. Print the error rate for K = 1 to 15 to compare values of K

## Result

Accuracy is 93.3% and weighted F1 is 0.93 on the 30 test samples. Setosa is classified perfectly, and the only mistakes are 2 virginica flowers predicted as versicolor. With a test set this small, one wrong sample moves accuracy by about 3%.
