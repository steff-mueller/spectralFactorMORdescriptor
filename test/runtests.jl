using Test

@testset "spectralFactorMORdescriptor" begin
    include("./test_irka.jl")
    include("./test_lyapunov.jl")
    include("./test_riccati.jl")
end
