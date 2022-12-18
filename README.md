# - 선형회귀 모델

 선형회귀 모델은 데이터를 모델에 학습시켜 예측가능하도록 하는 모델이다.
 
 주 목적은 입력 변수와 출력변수의 평균이 어떤 관계를 설명하는 선형식을 찾는 것이다.
 
 선형회귀 모델을 구축하기 위해 가정이 존재한다.
 

# - 1. 선형회귀 모델의 가정
 * 모든 관측치에 대해서 오차는 평균이 0, 분산이 시그마 제곱인 정규분포를 따른다.
 
   * $\epsilon_i \backsim N(0, \sigma^2), i = 1,2,...,n$
 
 * 출력값은 평균이 $\beta_0 + \beta_1 X_i$, 분산이 시그마 제곱인 정규분포를 따른다.

   * $Y_i = \beta_0 + \beta_1 X_i + \epsilon$ (선형회귀 식)
 
   * $Y_i \backsim N(\beta_0 + \beta_1 X_i, \sigma^2), i=1,2,...,n $

# - 2. 파라미터 추정
 * 모델 예측값과 실제 값과의 차이를 최소화하는 식을 통해 추정함
 
 * cost function(비용 함수)
 
   * $min \displaystyle\sum_{i=1} ^n (Y_i - ( \beta_0 + \beta_1 X_i))^2$
   
 * Least Squares Estimation Algorithm (최소제곱법)
 
   * 위 식에서 $\beta_0, \beta_1$ 에 대해 미분
      
   * $\hat{\beta_0} = \bar{Y} - \hat{\beta_1} \bar{X}$
   
   * $\hat{\beta_1} = \frac{\displaystyle\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{\displaystyle\sum_{i=1} ^n(X_i - \bar{X})^2}$
 
 * → 잔차 e는 확률 오차 $\epsilon$이 실제로 구현된 값
   
 
