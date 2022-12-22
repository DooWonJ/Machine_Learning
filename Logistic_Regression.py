from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, recall_score, precision_score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family = 'Malgun Gothic')

breast_cancer = load_breast_cancer()
print(breast_cancer.DESCR)

print(breast_cancer.feature_names)
print(breast_cancer.target_names)

breast_cancer.target # 0 :malignant(O), 1: benign(X)

# P(Y=1) 유방암일 확률, P(Y=0), 유방암이 아닐 확률
breast_cancer.target = np.where(breast_cancer.target == 0,1,0)
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size = 0.3, random_state = 2021)
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

column_name = ['const'] + breast_cancer.feature_names.tolist()
beta = np.concatenate([model.intercept_, model.coef_.reshape(-1)]).round(2)
odds = np.exp(beta).round(2)
interpret = np.where(beta>0, 'risky', 'protective')

beta_analysis = pd.DataFrame(np.c_[beta, odds, interpret], index = column_name, columns = ['beta', 'exp(beta)', 'interpret'])
beta_analysis

model.predict_proba(X_test).round(3) # 아닐확률, 유방암일 확률

Xbeta = np.matmul(np.c_[np.ones(X_test.shape[0]), X_test], beta.reshape(-1,1))

P_1 = 1/ (1+np.exp(-Xbeta))

pd.DataFrame(np.concatenate([P_1, model.predict_proba(X_test)[:,1].reshape(-1,1)], axis=1), columns=['직', '패'])

Cut_off = np.linspace(0.01, 0.99, 10)
for cutoff in Cut_off:
    y_pred = np.where(P_1.reshape(-1)>=cutoff,1,0)
    acc = accuracy_score(y_true = y_test, y_pred = y_pred)
    recall = recall_score(y_true = y_test, y_pred = y_pred)
    precision = precision_score(y_true = y_test, y_pred = y_pred)
    print(f"정확도 :{acc}, 민감도:{recall}, 정밀도:{precision}, cutoff:{cutoff}", sep='cut_off')

    # X_test에 대한 예측 확률 값 - 패키지
probs = model.predict_proba(X_test)[:, 1] # 두번째 컬럼 indexing

model_fpr, model_tpr, threshold1 = roc_curve(y_test, probs)
random_fpr, random_tpr, threshold2 = roc_curve(y_test, [0 for i in range(X_test.__len__())])

plt.figure(figsize = (10, 10))
plt.plot(model_fpr, model_tpr, marker = ',', label="logi")
plt.plot(random_fpr, random_tpr, linestyle = "--", label="ran")

plt.xlabel("FaslepositiveRate", size=20)
plt.ylabel("TruePositiveRate", size=20)

plt.legend(fontsize=20)

plt.title("ROC", size = 20)

plt.show()