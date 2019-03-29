
.onLoad <- function(libname, pkgname) {
  JuliaCall::julia_setup()
  JuliaCall::julia_library("EvoTrees")
}
