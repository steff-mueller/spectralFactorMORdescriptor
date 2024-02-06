using DescriptorSystems

function ispr(sys::AbstractDescriptorStateSpace)
    (; E, A, B, C, D) = sys
    T(s) = Matrix(C)/(s.*E-A)*B + D
    popov(w) = Hermitian(T(im*w) + transpose(T(-im*w)))

    w = [
        -exp10.(range(1, stop=-12, length=800));
        exp10.(range(-1, stop=12, length=800))
    ]

    mineigval, idx = findmin(minimum.(eigvals.(popov.(w))))
    return (ispr = mineigval > 0, mineigval = mineigval, w = w[idx])
end

"""
Check if `sys` is asymptotically stable.
"""
function isastable(sys::AbstractDescriptorStateSpace)
    sys_poles = filter(s -> !isinf(s), gpole(sys)) # finite eigenvalues
    return maximum(real((sys_poles))) < 0
end
