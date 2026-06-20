# Entry point: ReTestItems discovers and runs every `@testitem` under this package's test/
# directory (see purefft_tests.jl). Test items run in isolated modules, in parallel where possible.
using ReTestItems
using PureFFT

runtests(PureFFT)
