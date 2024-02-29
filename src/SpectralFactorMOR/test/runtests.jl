using Test
using Pkg
Pkg.add(url="https://github.com/Algopaul/PortHamiltonianBenchmarkSystems.jl/")

@testset "spectralFactorMORdescriptor" begin
    include("./test_irka.jl")
    include("./test_lyapunov.jl")
    include("./test_prbt.jl")
    include("./test_riccati.jl")
    include("./test_sfmor.jl")
end
