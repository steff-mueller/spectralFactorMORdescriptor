using DescriptorSystems
using GLMakie

function sigmaplot(systems::AbstractVector{TS};
    w = exp10.(range(-5, stop=2, length=100)), title = "Bodeplot"
) where TS <: AbstractDescriptorStateSpace
    f = Figure()
    ax = Axis(f[1, 1],
        title = title,
        xlabel = "w",
        ylabel = "σ_1(T(iw))",
        xscale = log10,
        yscale = log10
    )
    for (i,sys) in enumerate(systems)
        T(s) = Matrix(sys.C)/(s.*sys.E-sys.A)*sys.B+sys.D
        lines!(ax, w, opnorm.(T.(im.*w), 2), label="G_$i")
    end
    axislegend(ax)
    return f
end

function sigmaplot(sys::AbstractDescriptorStateSpace;
    w = exp10.(range(-5, stop=2, length=100)), title = "Bodeplot")
    return sigmaplot([sys], w=w, title=title)
end