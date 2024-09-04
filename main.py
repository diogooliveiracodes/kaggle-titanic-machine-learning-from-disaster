import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize

# %% abrir o database de treino e teste
dataset_treino: DataFrame = pd.read_csv('data/train.csv')
dataset_teste: DataFrame = pd.read_csv('data/test.csv')

# %% Tratamento dos dados

# Substituindo dados nulos no campo idade (Age) dos passageiros com o valor da média.
treino_idade_media = dataset_treino['Age'].median()
dataset_treino['Age'] = dataset_treino['Age'].fillna(treino_idade_media)

# Substituindo dados nulos no campo idade (Age) dos passageiros com o valor da média.
teste_idade_media = dataset_teste['Age'].median()
dataset_teste['Age'] = dataset_teste['Age'].fillna(teste_idade_media)

# Verificando dados nulos na coluna Cabin
lista_passageiros_cabine = dataset_treino.groupby('Cabin').size()

def determine_cabin_by_pclass(row):
    if pd.notna(row['Cabin']):
        return row['Cabin'][0]  # Retorna a primeira letra de Cabin se não for nulo
    else:
        # Define as opções com base no valor de Pclass
        if row['Pclass'] == 1:
            return np.random.choice(['A', 'B', 'C'])
        elif row['Pclass'] == 2:
            return np.random.choice(['D', 'E', 'F'])
        elif row['Pclass'] == 3:
            return np.random.choice(['E', 'F', 'G'])


# Aplicando a função ao DataFrame
dataset_treino['CabinByPclass'] = dataset_treino.apply(determine_cabin_by_pclass, axis=1)
dataset_teste['CabinByPclass'] = dataset_teste.apply(determine_cabin_by_pclass, axis=1)

# transformando idade em integer
dataset_treino['Age'] = dataset_treino['Age'].astype(int)
dataset_teste['Age'] = dataset_teste['Age'].astype(int)

# Nova Feature para verificar se o usuário é mulher
dataset_treino['Woman'] = dataset_treino['Sex'].apply(lambda x: 1 if x == 'female' else 0)
dataset_teste['Woman'] = dataset_teste['Sex'].apply(lambda x: 1 if x == 'female' else 0)

# Nova Feature para verificar se é criança
dataset_treino['Child'] = dataset_treino['Age'].apply(lambda x: 1 if x <= 12 else 0)
dataset_teste['Child'] = dataset_teste['Age'].apply(lambda x: 1 if x <= 12 else 0)

# Nova Feature para verificar se é idoso
dataset_treino['OldAge'] = dataset_treino['Age'].apply(lambda x: 1 if x >= 60 else 0)
dataset_teste['OldAge'] = dataset_teste['Age'].apply(lambda x: 1 if x >= 60 else 0)

# %% Limpeza dos dados
# Substituindo os valores de letras por números no CabinByPclass
subs = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
dataset_treino['CabinByPclass'] = dataset_treino['CabinByPclass'].replace(subs)
dataset_teste['CabinByPclass'] = dataset_teste['CabinByPclass'].replace(subs)

# %% pre-processamento dos dados

# mapear as colunas
col = pd.Series(list(dataset_treino.columns))

X_train = dataset_treino.drop(['PassengerId', 'Survived'], axis=1)
X_test = dataset_teste.drop(['PassengerId'], axis=1)

# %% Selecionar as Features

features = ['Pclass', 'Age', 'CabinByPclass', 'Woman', 'Child', 'OldAge']
X_train = X_train[features]
X_test = X_test[features]

y_train = dataset_treino['Survived']

# %% Padronização das variáveis

# Transforma os dados de treinamento em dados escalonados (média 0 e desvio padrão 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# o de teste não pode rodar o Fit
X_test = scaler.transform(X_test)

# %% logistic regression (algorítimo de classificação)
model_lr = LogisticRegression(max_iter=10000, random_state=0)
score = cross_val_score(model_lr, X_train, y_train, cv=10)
print(np.mean(score))

# %% Naive Bayes para Classificação
model_nb = GaussianNB()
score = cross_val_score(model_nb, X_train, y_train, cv=10)
print(np.mean(score))

# %% KNN para classificação
model_knn = KNeighborsClassifier(n_neighbors=5, p=2)
score = cross_val_score(model_knn, X_train, y_train, cv=10)
print(np.mean(score))

# %% SVM para classificação
model_svm = SVC(C=3, kernel='rbf', degree=2, gamma=0.1)
score = cross_val_score(model_svm, X_train, y_train, cv=10)
print(np.mean(score))

# %% Decision Tree
model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=2, min_samples_leaf=1,
                                  random_state=0)
score = cross_val_score(model_dt, X_train, y_train, cv=10)
print(np.mean(score))

# %% Random Forest

model_rf = RandomForestClassifier(criterion='entropy', max_depth=3, min_samples_split=2, min_samples_leaf=1,
                                  random_state=0, n_estimators=100)
score = cross_val_score(model_rf, X_train, y_train, cv=10)
print(np.mean(score))


# %% Otimizando dos Hiperparâmetros

def treinar_modelo(hiperparametros):
    model_rf_new = RandomForestClassifier(
        criterion=hiperparametros[0],
        n_estimators=hiperparametros[1],
        max_depth=hiperparametros[2],
        min_samples_split=hiperparametros[3],
        min_samples_leaf=hiperparametros[4],
        random_state=0,
        n_jobs=-1
    )
    score = cross_val_score(model_rf_new, X_train, y_train, cv=10)
    print(np.mean(score))

    return -np.mean(score)


hiperparametros = [('entropy', 'gini'), (100, 1000), (3, 20), (2, 10), (1, 10)]

otimos = gp_minimize(
    treinar_modelo,
    hiperparametros,
    random_state=0,
    verbose=1,
    n_calls=30,
    n_random_starts=10
)

# %% Utilizando os valores Ótimos para os hiperparametros
print(otimos.fun, otimos.x)

model_rf_otimo = RandomForestClassifier(
    criterion=otimos.x[0],
    n_estimators=otimos.x[1],
    max_depth=otimos.x[2],
    min_samples_split=otimos.x[3],
    min_samples_leaf=otimos.x[4],
    random_state=0,
    n_jobs=-1
)

# %% Ensanble model (Voting)
from sklearn.ensemble import VotingClassifier

model_voting = VotingClassifier(estimators=[
    ('LR', model_lr),
    ('KNN', model_knn),
    ('SVM', model_svm),
    ('RF', model_rf_otimo)
], voting='hard')
model_voting.fit(X_train, y_train)

score = cross_val_score(model_voting, X_train, y_train, cv=10)
print(np.mean(score))

# %% predição nos dados de teste

y_pred = model_voting.predict(X_test)
submission = pd.DataFrame(dataset_teste['PassengerId'])
submission['Survived'] = y_pred
submission.to_csv('submission6.csv', index=False)
