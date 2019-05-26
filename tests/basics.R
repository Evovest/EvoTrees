library(EvoTrees)
library(ggplot2)
library(xgboost)

x <- runif(10000, -10, 10)
y <- sin(x) * 0.5 + 0.5
y <- log(y/(1-y)) + rnorm(length(y))
y <- 1 / (1 + exp(-y))
target_train <- y
data_train <- matrix(x)
# data_train <- matrix(round(x - min(x) / (max(x) - min(x)) * 255))
# mode(data_train) <- "integer"

# no regularisation
params <- list(nthreads = 4, max_depth = 5, eta = 0.05,subsample = 0.5, colsample_bytree = 1.0, min_child_weight = 1, lambda = 0, alpha = 0, gamma = 0,tree_method = "exact", objective = "reg:linear")
xgb_train <- xgb.DMatrix(data = data_train, label = target_train)
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 200, verbose = 1, print_every_n = 10L, early_stopping_rounds = NULL))
pred_xgb <- predict(model, xgb_train)

params <- list(loss = "linear", nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1, nbins=64, metric="mae")
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params))
pred_linear <- predict(model = model, data = data_train)

params <- list(loss = "logistic", nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1)
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
pred_logistic <- predict(model = model, data = data_train)

params <- list(loss = "poisson", nrounds = 200, eta = 0.05, lambda = 0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1)
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
pred_poisson <- predict(model = model, data = data_train)

data <- data.frame(x = x, y = y, pred_linear = pred_linear, pred_logistic = pred_logistic)
ggplot() + geom_point(aes(x=x, y=y), col = "gray", size = 0.5) +
  geom_line(aes(x=x, y=pred_linear, col = "linear"), size = 1) +
  geom_line(aes(x=x, y=pred_logistic, col = "logistic"), size = 1) +
  geom_line(aes(x=x, y=pred_poisson, col = "poisson"), size = 1) +
  geom_line(aes(x=x, y=pred_xgb, col = "xgb"), size = 1) +
  ggtitle("No penalty")


# l2 regularization
params <- list(max_depth = 5, eta = 0.05,subsample = 0.5, colsample_bytree = 1.0, min_child_weight = 1, lambda = 1.0, alpha = 0, gamma = 0,tree_method = "exact", objective = "reg:linear", eval_metric = "rmse")
xgb_train <- xgb.DMatrix(data = data_train, label = target_train)
system.time(model <- xgb.train(data = xgb_train, params = params, nrounds = 200, verbose = 1, print_every_n = 10L, early_stopping_rounds = NULL))
pred_xgb <- predict(model, xgb_train)

params <- list(loss = "linear", nrounds = 200, eta = 0.05, lambda = 1.0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1, metric = "mse")
system.time(model <- evo_train(data_train = data_train, target_train = target_train, params = params))
pred_linear <- predict(model = model, data = data_train)

params <- list(loss = "logistic", nrounds = 200, eta = 0.05, lambda = 1.0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1, metric = "logloss")
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
pred_logistic <- predict(model = model, data = data_train)

data <- data.frame(x = x, y = y, pred_linear = pred_linear, pred_logistic = pred_logistic)
ggplot() + geom_point(aes(x=x, y=y), col = "gray", size = 0.5) +
  geom_line(aes(x=x, y=pred_linear, col = "linear"), size = 1) +
  geom_line(aes(x=x, y=pred_logistic, col = "logistic"), size = 1) +
  geom_line(aes(x=x, y=pred_xgb, col = "xgb"), size = 1) +
  ggtitle("L2 regularization")

# gamma/pruning regularization
params <- list(max_depth = 5, eta = 0.05,subsample = 0.5, colsample_bytree = 1.0, min_child_weight = 1, lambda = 0, alpha = 0, gamma = 1.0, tree_method = "exact", objective = "reg:linear", eval_metric = "rmse")
xgb_train <- xgb.DMatrix(data = data_train, label = target_train)
model <- xgb.train(data = xgb_train, params = params, nrounds = 200, verbose = 1, print_every_n = 10L, early_stopping_rounds = NULL)
pred_xgb <- predict(model, xgb_train)

params <- list(loss = "linear", nrounds = 200, eta = 0.05, lambda = 0, gamma = 1.0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1, metric = "mse")
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
pred_linear <- predict(model = model, data = data_train)

params <- list(loss = "logistic", nrounds = 200, eta = 0.05, lambda = 0, gamma = 1.0, max_depth = 6, min_weight = 1, rowsample = 0.5, colsample = 1, metric = "logloss")
model <- evo_train(data_train = data_train, target_train = target_train, params = params)
pred_logistic <- predict(model = model, data = data_train)

data <- data.frame(x = x, y = y, pred_linear = pred_linear, pred_logistic = pred_logistic)
ggplot() + geom_point(aes(x=x, y=y), col = "gray", size = 0.5) +
  geom_line(aes(x=x, y=pred_linear, col = "linear"), size = 1) +
  geom_line(aes(x=x, y=pred_logistic, col = "logistic"), size = 1) +
  geom_line(aes(x=x, y=pred_xgb, col = "xgb"), size = 1) +
  ggtitle("Gamma pruning")
