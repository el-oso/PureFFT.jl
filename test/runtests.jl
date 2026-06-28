# Entry point: ReTestItems discovers and runs every `@testitem` under this package's test/
# directory (see purefft_tests.jl). Test items run in isolated modules, in parallel where possible.
#
# Filtered runs for fast iteration: `Pkg.test(test_args=["pat"])` runs only `@testitem`s whose NAME
# matches the (case-insensitive) regex `pat` — e.g. `["REDFT|RODFT"]` for the r2r kinds, `["N-D|NDPlan"]`
# for the N-D items. Multiple args are OR'd. No args runs the whole suite (the pre-merge gate).
using ReTestItems
using PureFFT

if isempty(ARGS)
    runtests(PureFFT)
else
    runtests(PureFFT; name = Regex(join(ARGS, "|"), "i"))
end
