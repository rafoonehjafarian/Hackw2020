from model import *
import pandas as pd
import numpy as np
import sys


def main(data_file, split_pc, test):

    data = pd.read_csv(data_file)
    X_train, Y_train, X_test, Y_test = split_data(data, split_pc)
    X_train, Y_train = over_sample(X_train, Y_train)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train )
    pred = clf.predict(test)
    if pred[0] == 1:
        print("Your symptoms are consistent with Covid19, however, we encourage you to contact public-Sante at: \n"
              "1-877-644-4545 \n  for more information and instructions")
    else:
        print("your symptoms are not consistent with Covid19, however we encourage you to monitor your symptoms.\n"
              "If you noticed a new symptoms or you were in contact with someone who is a confirmed case of covid,"
              "contact  1-877-644-4545  for more information and instructions")



if __name__ == "__main__":
    print("Hello there!")
    print("I am going to ask to see what symptoms you are experiencing. Please answer by saying:\"yes\" or \"no\" ")
    test = []
    list_symptoms = ['loss of appetite', 'fever', 'cough', 'fatigue',
          'body aches/muscle pain', 'sore throat', 'diarrhoea', 'conjunctivities',
          'headache', 'loss of taste or smell', 'difficulty breathing',
          'chest pain or pressure', 'runny nose', 'new confusion', 'vomiting',
          'Sneezing', 'chills', ' watery eyes', 'Hoarsness']

    for symptom in list_symptoms:
        answer = input(f'are you experiencing {symptom} ?')
        if answer == "yes":
            test.append(1)
        else:
            test.append(0)

    test = np.array(test).reshape(1, -1)

    data_file = 'data/data_RuleOut - Sheet1.csv'
    split_pc = 0.8
    main(data_file, split_pc, test)