# Regularization

1. Expected MSE = $\sigma^2 + bias^2 (\hat{Y}) + Var(\hat{Y})$ = Irreducible Error + Bias^2 + Variance
2. MSE가 낮으면 예측 성능이 좋은 모델이다.
3. bias, variance의 장점은 각각 다르지만 상충하는 관계임. 둘 중 하나라도 낮추면 모델을 최적화시킬 수 있음.
  - Variance가 높다면, Training Data에 대해 좋지만 Testing Data에 대해서는 부적절한 모델.(과적합)
![image](https://user-images.githubusercontent.com/86637366/209555201-0633fe2a-4fe9-4d39-b918-bcf167708dcb.png)
4. 최소제곱법을 이용하여 구한 파라미터는 분산이 가장 작은(BLUE:Best Linear Unbiased Estimator) 파라미터임
  - $\hat{\beta}^{LS} = argmin_{\beta} \displaystyle \sum_{i=1}^n (y_i-x_i \beta)^2 = (X^T X)^{-1} X^T y$
5. Subset Selection
  - 전체 p개의 설명변수(X) 중 일부 k개만을 선택하여 $\beta$를 추정하는 방법
  - 전체 변수 중 일부를 선택하여 bias가 증가할 수 있지만 variance 감소
6. Regularization
  - 파라미터에 패널티를 부여하여 모델링에 변화를 꾀함(차수를 낮춤)
  - $\min \sum_{i=1} (y_i - \hat{y_i}^2) + \displaystyle \lambda \sum_j^p  \beta_j^2$
  - if $\lambda$ = big: Underfitting, else: Overfitting
7. Least Squares Method vs Regularize method
  - $\beta$ 에 대한 제약조건이 없음, 이때 Bias가 증가할 수 있지만 Variance가 감소함.
  - Bias: How much predicted value differ from true values
    - $E(\hat{f}(x) - f(x))^2$
  - variacne : How predictions made on the same value vary on different realizations of the model
    - $E(\hat{f}(x) - E(\hat{f}(x))^2)$
8. Ridge Regression
  - $\hat{\beta}^{ridge} = \displaystyle argmin_\beta \sum_{i=1} (y_i - \hat{y_i}^2) + \displaystyle \lambda \sum_j^p  \beta_j^2$
  - * Lagrangian multiplier
9. Lasso(Least Absolute Shrinkage and Selection Operator)
  - $\hat{\beta}^{ridge} = \displaystyle argmin_\beta \sum_{i=1} (y_i - \hat{y_i}^2) + \displaystyle \lambda \sum_j^p \left|\beta_j\right| $
10. Lasso vs Ridge
  - 모두 제약이 커짐과 동시에 파라미터가 작아지지만, Lasso는 중요하지 않은 변수가 더 빠르게 감소함. (selection)
  - 단, Lasso는 변수 간 상관 관계가 높을 수록 선택 성능이 저하된다.
    - 변수간 상관관계는 모델이 training data를 통해 얻어진 파라미터의 분산을 통해 알 수 있는데, 분산이 클 경우 Robustness(강건성)과 상관관계가 높다.
11. Elastic net
  - $min MSE + \displaystyle \lambda \sum_j^p \left|\beta_j\right| + \displaystyle \lambda \sum_j^p  \beta_j^2 $
  - 
