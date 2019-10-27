library(EvoTrees)
library(ggplot2)
library(xgboost)

nrows <- 200000
ncols = 100
x <- matrix(runif(nrows * ncols), nrow = nrows)
y <- runif(nrows)
target_train <- y
data_train <- x

# no regularisation
params <- list(nthreads = 4, max_depth = 5, eta = 0.05, subsample = 0.5, colsample_bytree = 0.5, min_child_weight = 1, lambda = 0, alpha = 0, tree_method = "hist", objective = "reg:linear", eval_metric = "rmse", max_bin=32)
xgb_train <- xgb.DMatrix(data = data_train, label = target_train)
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 100, verbose = 1, print_every_n = 2L, early_stopping_rounds = NULL))
pred_xgb <- predict(model, xgb_train)

params <- list(loss = "linear", nrounds = 100, eta = 0.05, lambda = 0, gamma = 0.01, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5, nbins = 32)
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params, print_every_n = 10))
pred_linear <- predict(model = model, data = data_train)

params <- list(loss = "logistic", nrounds = 100, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5)
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params))
pred_logistic <- predict(model = model, data = data_train)

