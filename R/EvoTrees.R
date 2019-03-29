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
                       colsample=1.0) {

  loss <- JuliaCall::julia_eval(paste0(":", loss))
  params <- JuliaCall::julia_call("Params",
                                  loss,
                                  as.integer(nrounds),
                                  lambda,
                                  gamma,
                                  eta,
                                  as.integer(max_depth),
                                  min_weight,
                                  rowsample,
                                  colsample,
                                  need_return = "Julia")
  return(params)
}

#' Train an EvoTree model
#' @export
evo_train <- function(data_train=1, target_train=1, params=set_params()) {
  model <- JuliaCall::julia_call("grow_gbtree", data_train, target_train, params, need_return = "Julia")
  return(model)
}


#' Get prediction from an EvoTree model
#' @export
evo_predict <- function(model, data) {
  pred <- JuliaCall::julia_call("predict", model, data, need_return = "R")
  return(pred)
}

