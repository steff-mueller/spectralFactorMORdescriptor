"""
Represents a system without holding any structure information
(in constract to [`SemiExplicitIndex1DAE`](@ref), [`StaircaseDAE`](@ref)
and [`AlmostKroneckerDAE`](@ref)).
"""
struct UnstructuredDAE{Tv, T <: AbstractMatrix{Tv}} <: AbstractDescriptorStateSpaceT{Tv}
    E::T
    A::T
    B::T
    C::T
    D::Matrix{Tv}
    Ts::Float64
end

function UnstructuredDAE(
    E::T,
    A::T,
    B::T,
    C::T,
    D::Matrix{Tv}
) where {Tv, T <: AbstractMatrix{Tv}}
    UnstructuredDAE(E, A, B, C, D, 0.0)
end

function +(
    sys1::AbstractDescriptorStateSpaceT{Tv1},
    sys2::AbstractDescriptorStateSpaceT{Tv2}
) where {Tv1,Tv2}
    T = promote_type(Tv1, Tv2)
    n1 = size(sys1.A, 1)
    n2 = size(sys2.A, 1)
    A = [sys1.A  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.A]
    E = [sys1.E  spzeros(T,n1,n2);
         spzeros(T,n2,n1) sys2.E]
    B = [sys1.B ; sys2.B]
    C = [sys1.C sys2.C;]
    D = [sys1.D + sys2.D;]
    return UnstructuredDAE(E, A, B, C, D, 0.0)
end

function -(sys::AbstractDescriptorStateSpaceT{Tv}) where {Tv}
    (; E, A, B, C, D) = sys
    return UnstructuredDAE(E, A, B, -C, -D, 0.0)
end

function -(
    sys1::AbstractDescriptorStateSpaceT{Tv1},
    sys2::AbstractDescriptorStateSpaceT{Tv2}
) where {Tv1,Tv2}
    return sys1 + (-sys2)
end
