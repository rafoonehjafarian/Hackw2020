
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score

column = None


def split_data(df,split_pct):
    """
    splitting dataframe into train and
    :param df: encoded and preprocessed csv dataset
    :param split_pct: training set percentage
    :return: training data, test data and associated labels
    """
    global column
    column = df.columns
    df_copy = df.copy()
    X_train= df_copy.sample(frac=split_pct, random_state=0)
    X_test = df_copy.drop(X_train.index)
    Y_train = X_train.pop(column[-1])
    Y_test = X_test.pop(column[-1])

    return X_train, Y_train, X_test, Y_test

def over_sample(X_train, Y_train):

    sm = SMOTE(random_state=2)
    X_train, Y_train= sm.fit_sample(X_train, Y_train.ravel())

    return X_train, Y_train

# def classify(X_train,Y_train,X_test,Y_test):
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(X_train,Y_train )
#     pred = clf.predict(X_test)
#     acc = accuracy_score(Y_test, pred)
#     return pred





