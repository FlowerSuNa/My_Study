# Correlation





## Pearson Correlation Coefficient

$ \gamma = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum(X_i - \overline{X})(Y_i - \overline{Y})}{\sqrt{\sum(X_i - \overline{X})^2(Y_i - \overline{Y})^2}}$


## Spearman's Rank Correlation Coefficient

$ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $

- $ d_i $ : 각 데이터 포인트의 순위 차이, $ d_i = R(X_i) - R(Y_i) $
- $ n $ : 데이터 포인트 수

## Kendall's Tau Correlation Coefficient

$ \tau_a = \frac{C - D}{C + D} = \frac{C - D}{\frac{1}{2} n(n - 1)} $

$ \tau_b = \frac{C - D}{\sqrt(C + D + T)(C + D + U)} $ : 동순위 데이터 처리 시 적합

- $ C $ : 순위가 일치하는 쌍의 수
- $ D $ : 순위가 불일치 하는 쌍의 수
- $ T $ : $ X $의 순위가 같은 쌍의 수
- $ U $ : $ Y $의 순위가 같은 쌍의 수