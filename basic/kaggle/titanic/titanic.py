import random
import numpy as np
import pandas as pd
from sklearn import preprocessing

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
labels_df = pd.read_csv('gender_submission.csv')

train_df['Fare'] = preprocessing.scale(train_df['Fare'])
train_df['Fare'] += 1

train_df.loc[train_df['Sex'] == 'female', 'Sex'] = 1
train_df.loc[train_df['Sex'] == 'male', 'Sex'] = 2

train_df.loc[train_df['Age'].isna() == True, 'Age'] = train_df[train_df['Age'].isna() == True]['Age'].apply(lambda x: random.randint(20, 38))
train_df['Age'] = preprocessing.scale(train_df['Age'])
train_df['Age'] += 3

train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 1
train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 2
train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 3
train_df = train_df.loc[train_df['Embarked'].notna()]

train_df.loc[train_df['SibSp'] == 0, 'SibSp'] = -1
train_df.loc[train_df['Parch'] == 0, 'Parch'] = -1

x_train = np.array(train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
y_train = np.array(train_df['Survived'])


test_df = test_df.join(labels_df, lsuffix='_l', rsuffix='_r')
test_df.isna().sum()/train_df.shape[0]*100

test_df.loc[test_df['Age'].isna() == True, 'Age'] = test_df[test_df['Age'].isna() == True]['Age'].apply(lambda x: random.randint(20, 38))
test_df['Age'] = preprocessing.scale(test_df['Age'])
test_df['Age'] += 3

test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Fare'] = preprocessing.scale(test_df['Fare'])
test_df['Fare'] += 1

test_df.loc[test_df['Embarked'] == 'C', 'Embarked'] = 1
test_df.loc[test_df['Embarked'] == 'Q', 'Embarked'] = 2
test_df.loc[test_df['Embarked'] == 'S', 'Embarked'] = 3
test_df = test_df.loc[test_df['Embarked'].notna()]

test_df.loc[test_df['SibSp'] == 0, 'SibSp'] = -1
test_df.loc[test_df['Parch'] == 0, 'Parch'] = -1

test_df.loc[test_df['Sex'] == 'female', 'Sex'] = 1
test_df.loc[test_df['Sex'] == 'male', 'Sex'] = 2

print(test_df.isna().sum()/test_df.shape[0]*100)

x_test = np.array(test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
y_test = np.array(test_df['Survived'])

# Функции активации
def tanh(x):
    return np.tanh(x)
def tanh2deriv(x):
    return 1 - (x ** 2)
def softmax(x):
    return x >= 0.5

# Дополнительные переменные
alpha, iterations, hidden_size = 0.01, 50, 10

weights_0_1 = 0.2 * np.random.random((x_train.shape[1], hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

for i in range(iterations):
    print('Iteration:', i)
    error_for_all = 0
    right_answers = 0
    for inps, goal_pred in zip(x_train, y_train):
        layer_0 = np.array([inps], dtype=np.float)
        layer_1 = tanh(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        
        goal_pred = np.array([[goal_pred]])
        
        layer_2_delta = (layer_2 - goal_pred)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * tanh2deriv(layer_1)
        
        weigths_0_1_delta = np.dot(layer_0.T, layer_1_delta) * alpha
        weights_1_2_delta = np.dot(layer_1.T, layer_2_delta) * alpha
        
        weights_0_1 -= weigths_0_1_delta
        weights_1_2 -= weights_1_2_delta

        layer_2 = softmax(layer_2)
        right_answers += layer_2[0][0] == goal_pred[0][0]
        error_for_all += np.sum((layer_2 - goal_pred) ** 2)
        
        # print(f'| Prediction: {layer_2[0][0]: >15} | Goal prediction: {goal_pred[0][0]} |')
        
    print(f'Accuracy: {right_answers/x_train.shape[0]*100: .2f}%')
    print(f'Error for all: {error_for_all:.8f}')
    print('-' * 70)

answers = []
for inps in x_test:
    layer_0 = np.array([inps], dtype=np.float)
    layer_1 = tanh(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)
    
    layer_2 = softmax(layer_2)
    answers.append(int(layer_2[0][0]))
    
answers_df = pd.DataFrame(data=zip(np.arange(892, 1310), answers), columns=['PassengerId','Survived'])
answers_df.to_csv('./my_submission.csv', index=False)