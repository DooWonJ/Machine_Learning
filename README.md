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
 
   * $\hat{\sigma}^2 = \frac{n-2}{1} \displaystyle\sum_{i=1}^n e_i^2$
 
      * 잔차 e는 확률 오차 $\epsilon$이 실제로 구현된 값
 
      * 최소제곱법 추정량 성질
        
        Gauss-markovTheorm: Least square estimator is the best linear unbiased estimator (BLUE)
        BLUE: 최소제곱법 추정량은 불편 추정량이고, 다른 불편 추정량에 비해 분산이 적다.
        
        (1) unbiased estimator $E(\hat{\beta_0}) = \beta_0, E(\hat{\beta_1}) = \beta_1$
        
        (2) smallest variance estimator $V(\alpha\hat{\beta_0}) <= V(\beta\hat{\theta})$ 
     
 
   # - 3. 결정계수
   
   ![image](https://user-images.githubusercontent.com/86637366/208448158-b1ef9028-e2e2-448b-90b4-0f7bcf3a8341.png)

     1) SSE: X로 설명할 수 없는 부분
     2) SSR: X로 설명할 수 있는 부분
     3) SST: SSE + SSR
     
       * SSE가 0이라는 것은, 확정적인 관계임을 의미함.(Error가 없으므롤 모든 것이 X로 설명가능함)
       * SSR이 0이라는 것은, 현재 직선을 이루는 X변수는 Y를 설명하는데 도움이 되지 않음.
     
     * $R^2$ = $\frac{SSR}{SST}$
       * 현재 사용하는 X가 Y변수의 분산을 줄이는데 도움이 되었는지
       * 단순한 Y의 평균값보다 X정보를 통해 얻는 성능향상 정도
       * 사용하는 X변수의 품질
       
     * 수정 결정계수(Adjusted $R^2$)
         * $R_adj^2 = 1- \frac{n-1}{n-p-1} \frac{SSE}{SST}$
         * 결정계수는 표본 수가 증가하고 유의하지 않은 종속 변수가 추가되어도 증가한다는 단점이 있다.
         * 수정 결정계수는 특정 계수를 곱해주어, 유의하지 않는 변수가 추가되면 증가하지 않게 한다.
   # - 4. 분산분석(Analysis of Variance)
         
         * SSR/SSE의 값을 통해 X변수와 에러에 의해 설명된 양의 차이를 알아보고자 함.
         
         ![image](https://user-images.githubusercontent.com/86637366/208461100-823210dc-2d46-4f48-88ef-975bf813f9ec.png)
         
         - SSR/SSE의 분포는 모르지만, SSR, SSE 각각 카이제곱 분포를 따르고, 각 수치를 자유도로 나눈 값, MSR, MSE를 통해 통계적 판단이 가능해짐.
         - MSR(Mean Squared Residual): SSR / DF(1)
         - MSE(Mean Squared Error): SSE / DF(n-2)
         
         - 귀무가설: 모든 파라미터가 0 ($\beta_i = 0$) vs 대립가설: 적어도 하나의 파라미터가 0 이 아닐 수 있다.
         - MSR/MSE = F*이고, 이를 통해 사용된 X변수의 설명 가능 여부를 판단 가능.
         




