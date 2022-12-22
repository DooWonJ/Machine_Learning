# - 1. 로지스틱 회귀모델

  1) 필요성
  
     * 출력변수가 범주형일 경우 로지스틱 회귀모델로 접근해야함.
     
     * 분류 문제를 해결할 때 활용
     
     * 출력 변수가 0, 1로 나뉠 경우 데이터를 찍어보면 위 아래로 나뉘는 현상이 발생. → 선형 표현 불가
  
  2) 형태
  
     * $f(X) = \frac{1}{1+e^{-(\beta_0 + \beta_1X)}}$
       * Probability that an observation x belongs to class 1
     * X를 범주별로 구분하여 
    
     * 로지스틱, 시그모이드 함수
     
     * 인풋값에 대해 단조 증가 및 감소 형태를 띔.

     * 미분 결과를 아웃풋의 함수로 표현이 가능.
       * $\phi(z) = \frac{1}{1+e^{-z}}$
       * $\frac{d \phi(z)}{dz} = \phi(z)(1-\phi(z))$
  
  3) $\beta_1$의 해석
     * 승산(Odds): 범주 0에 속할 확률 대비 범주 1에 속할 확률
       * 성공 확률이 p일 때 실패 대비 성공 확률 비
       * odd = $\frac{p}{1-p}$
     * $log(odd) = \beta_0 + \beta_1 x$ (Logit Transform)
       * 즉, $beta_1$은 x가 한단위 증가했을 때 log(odds)의 증가량

  4) 파라미터 추정
   * 최대 우도 추정법(Maimum Likelihood Estimation)
      * 베르누이 함수(L) ($f_i(y_i) = \pi(x_i)^y_i(1-\pi(x_i)^{1-y_i}$)에 로그를 취해주면
      * $ln L = \sum_i y_i(\beta_0+\beta_1 X_1 + ... + \beta_p X_p) +\sum_i ln(1+e^{\beta_0+\beta_1 X_1 + ... + \beta_p X_p})$ [Log likelihood function]
      * 식이 최대가 되는 $\beta$값을 결정
      * 단, 파라미터에 대해 비선형적이라 명시적인 해가 존재하지 않음.(No Closed from solution exist)
      * Iterative reweight least square, Conjugate gradient, newton's method 등 알고리즘 이용*    
   * Cross entropy: 두 확률 분포($p(x), q(x)$)의 차이를 측정하는 척도
      * $ -\sum p(x) log q(x)$
      * 음의 log likelihood function의 기대값
      * Cross entropy를 최소화한다는 것은 입력분포와 출력분포 차이를 최소화한다는 것을 의미
      * max log likelihood function = min Creoss entropy
      
