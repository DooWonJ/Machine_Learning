import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family = 'Malgun Gothic')

boston = load_boston()
# 간략한 설명
print(boston.DESCR)
# x변수 출력
print(boston.data)
# x변수명 확인
print(boston.feature_names)
# 데이터 규격 (로드 데이터 : boston.shape)
print(boston.data.shape)
# y변수 출력
print(boston.target)

# 데이터 프레임 형태로 변환
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['MEDV'])

# Train, Test 나누기 (testsize, randomstate(seed값))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2021)

# 모델 구축
# OLS: 결정론적 선형회귀 방법, 잔차 제곱합(RSS:Residual Sum of Squares)를 최소화하는 가중치 구하는 방법
# 모델 선언 model = sm.OLS(y데이터, x데이터)
# 모델 학습 model_trained = model.fit()

# 상수항 (b0 생성)
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train, axis = 1)
model_trained = model.fit()

# 선형회귀모델 가정
# 1. 확률 오차 정규성
model_residuals = model_trained.resid

# 음수 폰트 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1,1)
fig.set_figheight(12)
fig.set_figwidth(12)

sm.ProbPlot(model_residuals).qqplot(line='s', color='#1f77b4', ax=ax)
ax.title.set_text("QQ Plot")

# 2. 확률오차의 등분산성 확인
model_fitted_y = model_trained.fittedvalues

fig, ax = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(12)

sns.residplot(model_fitted_y, y= y_train, x =X_train, lowess=True, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax = ax)
ax.title.set_text('Residuals vs Fitted')
ax.set(xlabel='Fitted values', ylabel='Residuals')

# 통계적해석
# R-squared(결정계수, coefficient of determination):모형의 성능
# coef(회귀계수): X가 한단위 증가할 때 Y의 변화량
# P>[t] (p-value):0.05(유의수준) 이하일 때 변수가 유의미하다
print(model_trained.summary())

# 확인 결과, INDUS, Age는 P-value값이 높아, 무의미하다 판단

model = sm.OLS(y_train, X_train.drop(['INDUS', 'AGE'], axis = 1))
model_trained= model.fit()
print(model_trained.summary())

# 예측
y_train_pred = model_trained.fittedvalues

plt.figure(figsize=(8,8))
plt.title('실제값 vs. 모델 출력 값')
plt.scatter(y_train, y_train_pred)
plt.plot([-5, 55], [-5, 55], ls = "-", c = 'red')
plt.xlabel('실제값', size = 16)
plt.ylabel('예측값', size = 16)
plt.xlim(-5, 55)
plt.ylim(-5, 55)
plt.show()

X_test = sm.add_constant(X_test)
y_test_pred = model_trained.predict(X_test.drop(['INDUS', 'AGE'], axis = 1))
y_test_pred.head()

# Mean Squared error (평균 제곱 오차)
print(mean_squared_error(y_test, y_test_pred))

# 제곱근 평균 제곱오차
print(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# 평균 절대 오차
print(mean_absolute_error(y_test, y_test_pred))

# 평균 절대 백분율 오차
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

print(mean_absolute_percentage_error(y_test, y_test_pred))

# 결정계수
print(r2_score(y_test, y_test_pred))

# 결과 정리 (train vs test)
print(mean_squared_error(y_test, y_test_pred))
print(np.sqrt(mean_squared_error(y_test, y_test_pred)))
print(mean_absolute_error(y_test, y_test_pred))
print(mean_absolute_percentage_error(y_test, y_test_pred))
print(r2_score(y_test, y_test_pred))

# 결과 정리
print(mean_squared_error(y_train, y_train_pred))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
print(mean_absolute_error(y_train, y_train_pred))
print(mean_absolute_percentage_error(y_train, y_train_pred))
print(r2_score(y_train, y_train_pred))

# sklearn vs statsmodels

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lr_skl = LinearRegression(fit_intercept = False)
lr_skl.fit(X_train, y_train)
y_pred_skl = lr_skl.predict(X_test)

lr_stat = sm.OLS(y_train, X_train).fit()
y_pred_stat = lr_stat.predict(X_test)

#statmodels 결과

print(mean_squared_error(y_test, y_pred_stat))
print(np.sqrt(mean_squared_error(y_test, y_pred_stat)))
print(mean_absolute_error(y_test, y_pred_stat))
print(mean_absolute_percentage_error(y_test, y_pred_stat))
print(r2_score(y_test, y_pred_stat))

#sklearn 결과
print(mean_squared_error(y_test, y_pred_skl))
print(np.sqrt(mean_squared_error(y_test, y_pred_skl)))
print(mean_absolute_error(y_test, y_pred_skl))
print(mean_absolute_percentage_error(y_test, y_pred_skl))
print(r2_score(y_test, y_pred_skl))