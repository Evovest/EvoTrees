#' Set EvoTree model parameters
#' @export
set_params <- function(loss="linear",
                       nrounds=10L,
                       lambda=0.0,
                       gamma=0.0,
                       eta=0.1,
                       max_depth=5L,
                       min_weight=1.0,
                       rowsample=1.0,
                       colsample=1.0,
                       nbins=50,
                       alpha=0.0,
                       metric="none",
                       ...) {

  params <- JuliaCall::julia_call("EvoTreeRegressorR",
                                  as.symbol(loss),
                                  as.integer(nrounds),
                                  as.numeric(lambda),
                                  as.numeric(gamma),
                                  as.numeric(eta),
                                  as.integer(max_depth),
                                  as.numeric(min_weight),
                                  as.numeric(rowsample),
                                  as.numeric(colsample),
                                  as.integer(nbins),
                                  as.numeric(alpha),
                                  as.symbol(metric),
                                  need_return = "Julia")

  return(params)
}

#' Train an EvoTree model
#' @export
evo_train <- function(data_train, target_train, params=set_params(), ...) {
  params <- do.call(set_params, params)
  model <- JuliaCall::julia_call("grow_gbtree", data_train, target_train, params, ..., need_return = "Julia")
  return(model)
}

#' Get model best iter and eval metric
#' @export
get_metric <- function(model) {
  metric <- JuliaCall::field(model, "metric")
  best_iter <- JuliaCall::field(metric, "iter")
  best_score <- JuliaCall::field(metric, "metric")
  return(list(best_iter = best_iter, best_score = best_score))
}

#' Get model best iter and eval metric
#' @export
best_iter.JuliaObject <- function(model) {
  metric <- JuliaCall::field(model, "metric")
  best_iter <- JuliaCall::field(metric, "iter")
  return(best_iter)
}

#' Get model best iter and eval metric
#' @export
best_score.JuliaObject <- function(model) {
  metric <- JuliaCall::field(model, "metric")
  best_score <- JuliaCall::field(metric, "metric")
  return(best_score)
}

#' Get prediction from an EvoTree model
#' @export
predict.JuliaObject <- function(model, data) {
  pred <- JuliaCall::julia_call("predict", model, data, need_return = "R")
  return(pred)
}
