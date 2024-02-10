struct SparseDescriptorStateSpace{Tv,Ti} <: AbstractSparseDescriptorStateSpace{Tv,Ti}
    E::SparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    B::SparseMatrixCSC{Tv,Ti}
    C::SparseMatrixCSC{Tv,Ti}
    D::Matrix{Tv}
    Ts::Float64
end

function +(sys1::AbstractSparseDescriptorStateSpace{Tv1,Ti}, sys2::DescriptorStateSpace{Tv2}) where {Tv1,Tv2,Ti}
    # We assume that the dense system `sys2` has a much smaller state space dimension than
    # the sparse system `sys1`. Thus, we return a `SparseDescriptorStateSpace` to maintain
    # the sparsity.
    T = promote_type(Tv1, Tv2)
    n1 = size(sys1.A, 1)
    n2 = sys2.nx
    A = [sys1.A  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.A]
    E = [sys1.E  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.E]
    B = [sys1.B ; sys2.B]
    C = [sys1.C sys2.C;]
    D = [sys1.D + sys2.D;]
    return SparseDescriptorStateSpace(E, A, B, C, D, 0.0)
end

function +(sys1::DescriptorStateSpace{Tv1}, sys2::AbstractSparseDescriptorStateSpace{Tv2,Ti}) where {Tv1,Tv2,Ti}
    return sys2 + sys1
end

function -(sys::AbstractSparseDescriptorStateSpace{Tv,Ti}) where {Tv,Ti}
    (; E, A, B, C, D) = sys
    return SparseDescriptorStateSpace(E, A, B, -C, -D, 0.0)
end

function -(sys1::AbstractSparseDescriptorStateSpace{Tv1,Ti}, sys2::DescriptorStateSpace{Tv2}) where {Tv1,Tv2,Ti}
    return sys1 + (-sys2)
end

function -(sys1::DescriptorStateSpace{Tv1}, sys2::AbstractSparseDescriptorStateSpace{Tv2,Ti}) where {Tv1,Tv2,Ti}
    return sys1 + (-sys2)
end

function todss(sys::AbstractSparseDescriptorStateSpace)
    (; E, A, B, C, D) = sys
    return dss(A, E, B, C, D)
end
