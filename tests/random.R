library(EvoTrees)
library(ggplot2)
library(xgboost)

nrows <- 1e6
ncols = 100
x <- matrix(runif(nrows * ncols), nrow = nrows)
y <- runif(nrows)
target_train <- y
data_train <- x

# no regularisation
params <- list(nthread = 16, max_depth = 5, eta = 0.05, subsample = 0.5, colsample_bytree = 0.5, min_child_weight = 1, lambda = 0, alpha = 0, tree_method = "hist", objective = "reg:squarederror", eval_metric = "rmse", max_bin = 64)
system.time(xgb_train <- xgb.DMatrix(data = data_train, label = target_train))
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 200, verbose = 1, print_every_n = 2L, early_stopping_rounds = NULL))
system.time(pred_xgb <- predict(model, xgb_train, ntreelimit = 99))

# no regularisation - GPU
params <- list(nthread = 16, max_depth = 5, eta = 0.05, subsample = 0.5, colsample_bytree = 0.5, min_child_weight = 1, lambda = 0, alpha = 0, tree_method = "gpu_hist", objective = "reg:squarederror", eval_metric = "rmse", max_bin = 64)
system.time(xgb_train <- xgb.DMatrix(data = data_train, label = target_train))
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 200, verbose = 1, print_every_n = 2L, early_stopping_rounds = NULL))
pred_xgb <- predict(model, xgb_train)

# evotrees
params <- list(loss = "linear", nrounds = 200, eta = 0.05, lambda = 0, gamma = 0.01, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5, nbins = 64, metric = "mse")
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params, print_every_n = 9999))
model <- evo_train(data_train = data_train, target_train = target_train, params = params, print_every_n = 10)
system.time(pred_linear <- predict(model = model, data = data_train))
var_names <- paste0("var_", 1:ncol(data_train))
var_importance <- importance(model = model, var_names = var_names)

#save /load
params <- list(loss = "logistic", nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5)
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params))
system.time(pred_logistic <- predict(model = model, data = data_train))

params <- list(loss = "quantile", alpha = 0.25, nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5, metric = "quantile")
system.time(model <- evo_train(data_train = data_train, target_train = target_train, X_eval = data_train, Y_eval = target_train, params = params, print_every_n = 10))
system.time(pred_quantile <- predict(model = model, data = data_train))

params <- list(loss = "gaussian", nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5)
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params))
system.time(pred_gaussian <- predict(model = model, data = data_train))


###################
# classification
target_train_int_xgb <- as.integer(target_train*3)
target_train_int <- as.integer(target_train*3) + 1
table(target_train_int_xgb)
table(target_train_int)

# no regularisation
params <- list(nthreads = 4, max_depth = 5, eta = 0.05, subsample = 0.5, colsample_bytree = 0.5, min_child_weight = 1, lambda = 0, alpha = 0, tree_method = "hist", objective = "multi:softprob", num_class = 3, eval_metric = "mlogloss", max_bin = 64)
system.time(xgb_train <- xgb.DMatrix(data = data_train, label = target_train_int_xgb))
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 100, verbose = 1, print_every_n = 2L, early_stopping_rounds = NULL))
system.time(pred_xgb <- predict(model, xgb_train, ntreelimit = 99))

params <- list(loss = "softmax", nrounds = 100, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 0.5)
system.time(model <- evo_train(data_train = data_train, target_train = target_train_int, params = params))
system.time(pred_softmax <- predict(model = model, data = data_train))

