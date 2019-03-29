# EvoTrees

> This is the R package wrapping EvoTrees.jl, a pure Julia tree boosting library. 

# Getting started

EvoTrees.jl need first to be installed and available from main julia environment. 
Then, install package with: `devtools::install_github("Evovest/EvoTrees")`.

# Example

```
params <- set_params(loss = "linear", nrounds = 100)
data_train <- matrix(runif(10000), nrow=1000)
target_train <- runif(1000)
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
preds <- evo_predict(model = model, data = data_train)
```

# Parameters

- loss: {"linear","logistic"}
- nrounds: 10L
- lambda: 0.0
- gamma: 0.0
- eta: 0.1
- max_depth: integer, default 5L
- min_weight: float >= 0 default=1.0,
- rowsample: float [0,1] default=1.0
- colsample: float [0,1] default=1.0
