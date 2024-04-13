from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
paths = ['data/reviews_0-250_processed.csv',
         'data/reviews_250-500_processed.csv',
         'data/reviews_500-750_processed.csv',
         'data/reviews_750-1250_processed.csv',
         'data/reviews_1250-end_processed.csv']


df = pd.read_csv(paths[1], low_memory=False).sample(frac=0.1)


# Define a dictionary of classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000)
}

print(df.columns)

X = df[['skin_tone', 'eye_color', 'skin_type', 'hair_color']]
y = df['product_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: Accuracy = {accuracy}")
