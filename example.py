# The code is from https://automl.github.io/auto-sklearn/master/index.html#example
# Added time log to check the time taken to fit the model.
import datetime
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier()

print(datetime.datetime.now(),'Start of fit')
automl.fit(X_train, y_train)
print(datetime.datetime.now(),'End of fit')

y_hat = automl.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
