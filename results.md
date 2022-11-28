| 次      | exp settings      | logits (valid) predictor |  LR (test) classifier | gap mean std| note|
| ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- |
| 第一次      | 原sage (epoch ~200)      | 0.86   | 0.86 | -|
|     |        |
| 第二次   | 原sage (epoch ~200)       | 0.86 | 0.76| 0.18(0.07)
|    | label sage cosine (epoch ~5.5k, coef 0.5)      | 0.92+ | 0.826| 
|    | label sage euclidean  (epoch ~7k, coef 0.5)     | ~0.85 | 0.8244|

