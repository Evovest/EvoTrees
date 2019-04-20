
.onLoad <- function(libname, pkgname) {
  library(JuliaCall)
  JuliaCall::julia_setup()
  JuliaCall::julia_library("EvoTrees")
}
