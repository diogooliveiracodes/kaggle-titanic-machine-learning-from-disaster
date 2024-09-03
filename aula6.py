import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# %% abrir o database de treino e teste
dataset_treino: DataFrame = pd.read_csv('data/train.csv')
dataset_teste: DataFrame = pd.read_csv('data/test.csv')

# %% Análise dos campos do Dataset

# Dataset de Treino
# dataset_treino_descricao_estatistica = dataset_treino.describe()
# dataset_treino_tipos_de_dados = dataset_treino.dtypes
# dataset_treino_valores_nulos = dataset_treino.isnull().sum()

# Dataset de Testes
# dataset_teste_descricao_estatistica = dataset_teste.describe()
# dataset_teste_tipos_de_dados = dataset_teste.dtypes
# dataset_teste_valores_nulos = dataset_teste.isnull().sum()

# %% Plotagem dos dados

for i in dataset_treino.columns:
    plt.hist(dataset_treino[i])
    plt.title(i)
    plt.show()

# %% Groupby
# agrupados_por_sobreviventes = dataset_treino.groupby('Survived').count()
media_por_sobreviventes = dataset_treino.groupby(['Survived']).mean()

# %% pivot_table
sobreviventes_por_classe = pd.pivot_table(dataset_treino, index=['Survived'], columns=['Pclass'], values='PassengerId',
                                          aggfunc='count')
sobreviventes_por_embarque = pd.pivot_table(dataset_treino, index=['Survived'], columns=['Embarked'],
                                            values='PassengerId',
                                            aggfunc='count')

# %% Tratamento dos dados

# Substituindo dados nulos no campo idade (Age) dos passageiros com o valor da média.
treino_idade_media = dataset_treino['Age'].median()
dataset_treino['Age'] = dataset_treino['Age'].fillna(treino_idade_media)

# Substituindo dados nulos no campo idade (Age) dos passageiros com o valor da média.
teste_idade_media = dataset_teste['Age'].median()
dataset_teste['Age'] = dataset_teste['Age'].fillna(teste_idade_media)

# Verificando dados nulos na coluna Cabin
lista_passageiros_cabine = dataset_treino.groupby('Cabin').size()


# Criando nova coluna chamada "CabinByPclass"
# - Verifica se o campo Cabin está preenchido, se estiver, pega apenas a primeira letra da cabine.
# - Se estiver nulo, verifica qual a classe do passageiro e preenche aleatoriamente com a seguinte divisão:
# [ 3 classe = E, F ou G ], [ 2 classe = D, E ou F], [ 1 classe = A, B ou C]

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

# %% modelo e validação cruzada

# logistic regression (algorítimo de classificação)
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
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(criterion='entropy', max_depth=3, min_samples_split=2, min_samples_leaf=1,
                                  random_state=0, n_estimators=100)
score = cross_val_score(model_rf, X_train, y_train, cv=10)
print(np.mean(score))

# %% Modelo Final
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_train)
mc = confusion_matrix(y_train, y_pred)  # matriz de confusão
print(mc)

# OUTPUT:
# [[468  81] 468 certos, 81 errados
# [109 233]] 109 errados, 233 certos

score = model_rf.score(X_train, y_train)
print(score)

# %% predição nos dados de teste

y_pred = model_rf.predict(X_test)

submission = pd.DataFrame(dataset_teste['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('submission4.csv', index=False)
