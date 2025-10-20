import ForwardDiff: Dual
using StaticArrays

# TODO Based on this type do some generic things like use spin-scaling relations
# Note: Kind is needed because GGA enhancement or spin scaling etc. work differently
#       for exchange and correlation ... if it does not help, remove it again
abstract type Functional{Family,Kind} end

"""Return the family of a functional. Results are `:lda`, `:gga`, `:mgga` and
`:mggal` (Meta-GGA requiring Laplacian of ρ)"""
family(::Functional{F}) where {F} = F

"""
Return the functional kind: `:x` (exchange), `:c` (correlation), `:k` (kinetic) or
`:xc` (exchange and correlation combined)
"""
kind(::Functional{F,K}) where {F,K} = K

"""Return the identifier corresponding to a functional"""
function identifier end
Base.show(io::IO, fun::Functional) = print(io, identifier(fun))

@doc raw"""
True if the functional needs ``σ = 𝛁ρ ⋅ 𝛁ρ``.
"""
needs_σ(::Functional{F}) where {F} = (F in (:gga, :mgga, :mggal))

@doc raw"""
True if the functional needs ``τ`` (kinetic energy density).
"""
needs_τ(::Functional{F}) where {F} = (F in (:mgga, :mggal))

@doc raw"""
True if the functional needs ``Δ ρ``.
"""
needs_Δρ(::Functional{F}) where {F} = (F in (:mggal,))

"""
Does this functional support energy evaluations? Some don't, in which case
energy terms will not be returned by `potential_terms` and `kernel_terms`,
i.e. `e` will be `false` (a strong zero).
"""
has_energy(::Functional) = true

"""
Return adjustable parameters of the functional and their values.
"""
parameters(::Functional) = ComponentArray{Bool}()

"""
Return a isbits version of the functional (for GPU usage)
"""
to_isbits(func::Functional) = func

"""
Return a new version of the passed functional with its parameters adjusted.
This may not be a copy in case no changes are done to its internal parameters.
Generally the identifier of the functional will be changed to reflect the
change in parameter values unless `keep_identifier` is true.
To get the tuple of adjustable parameters and their current values check out
[`parameters`](@ref). It is not checked that the correct parameters are passed.

`change_parameters(f::Functional, params_new; keep_identifier=false)::Functional`
"""
function change_parameters end

# TODO These values are read-only for now and their defaults hard-coded for Float64
"""
Threshold for the density (below this value, functionals and derivatives
evaluate to zero). The threshold may depend on the floating-point type used
to represent densities and potentials, which is passed as the second argument.
"""
threshold_ρ(::Functional, T=Float64) = T(1e-15)  # TODO This might differ between functionals
threshold_σ(f::Functional, T=Float64) = threshold_ρ(f, T)^(4 // 3)
threshold_τ(::Functional, T=Float64)  = T(1e-20)
threshold_ζ(::Functional, T=Float64)  = eps(T)

# Drop dual types from threshold functions
threshold_ρ(f::Functional, T::Type{<:Dual}) = threshold_ρ(f, ForwardDiff.valtype(T))
threshold_σ(f::Functional, T::Type{<:Dual}) = threshold_σ(f, ForwardDiff.valtype(T))
threshold_τ(f::Functional, T::Type{<:Dual}) = threshold_τ(f, ForwardDiff.valtype(T))
threshold_ζ(f::Functional, T::Type{<:Dual}) = threshold_ζ(f, ForwardDiff.valtype(T))

# Util functions for Dual comparison on the GPU. TODO: overload operators directly?
# In fact, this is only truely necessary for AMD GPUs, so do that in an extensions?
# Finally, can we inherit the overloaded comparisons from DFTK?
_val(x) = x
_val(x::Dual) = _val(ForwardDiff.value(x))

# Silently drop extra arguments from evaluation functions
for fun in (:potential_terms, :kernel_terms)
    @eval begin
        $fun(func::Functional{:lda}, ρ, σ, args...)         = $fun(func, ρ)
        $fun(func::Functional{:gga}, ρ, σ, τ, args...)      = $fun(func, ρ, σ)
        $fun(func::Functional{:mgga}, ρ, σ, τ, Δρ, args...) = $fun(func, ρ, σ, τ)
    end
end

@doc raw"""
    potential_terms(f::Functional, ρ, [σ, τ, Δρ])

Evaluate energy and potential terms at a real-space grid of densities, density
derivatives etc. Not required derivatives for the functional type will be ignored.
Returns a named tuple with keys `e` (Energy per unit volume),
`Vρ` (``\frac{∂e}{∂ρ}``), `Vσ` (``\frac{∂e}{∂σ}``),
`Vτ` (``\frac{∂e}{∂τ}``), `Vl` (``\frac{∂e}{∂(Δρ)}``).
"""
function potential_terms end

@doc raw"""
    kernel_terms(f::Functional, ρ, [σ, τ, Δρ])

Evaluate energy, potential and kernel terms at a real-space grid of densities, density
derivatives etc. Not required derivatives for the functional type will be ignored.
Returns a named tuple with the same keys as `potential_terms` and additionally
second-derivative cross terms such as `Vρσ` (``\frac{∂^2e}{∂ρ∂σ}``).
"""
function kernel_terms end

#
# LDA
#
function potential_terms(func::Functional{:lda}, ρ::AbstractMatrix{T}) where {T}
    @assert has_energy(func)  # Otherwise custom implementation of this function needed
    s_ρ, n_p = size(ρ)
    TT = arithmetic_type(func, T)

    indices = similar(ρ, Int, n_p)
    copyto!(indices, collect(1:n_p))

    e  = similar(ρ, TT, n_p)
    Vρ = similar(ρ, TT, s_ρ, n_p)
    #@views for i = 1:n_p
    #    potential_terms!(e[i:i], Vρ[:, i], func, ρ[:, i]) #TODO: this call seems to be important, rather than the content of e
    #end
    #TODO: seems to work that way. Check if this is OK without modification to ForwardDiff or DFTK
    #      then refine, and check perf vs simply transfering to CPU and back
    #      Is allocating a SVector at each iteration expensive?
    #      Also, make sure whatever we do does not impact CPU perf: it actually seems to be faster that way!
    #      This makes the calculation of the XC energy (a massive bottleneck with @allowscalar) negligible!
    #      need all tests on a big system too, of course
    map!(e, indices) do i
        #TODO: assume spin 1 for now
        tmp = potential_terms(func, SVector(ρ[1, i]))
        Vρ[1, i:i] .= tmp.Vρ
        tmp.e
    end
    (; e, Vρ)
end
function potential_terms(func::Functional{:lda}, ρ::AbstractVector{T}) where {T}
    #TODO: do we really need to pass V as argument, can't we just spit it out?
    res = ForwardDiff.gradient!(DiffResults.GradientResult(ρ), ρ -> energy(func, ρ), ρ)
    #TODO: to me it looks like e is just the value of the above, while V is the full thing. Could simply
    #      return res, and do the cooking afterwards? Could even do a map on V, and the fill e in a second loop?
    (; e = DiffResults.value(res), Vρ = DiffResults.gradient(res))
end
#function potential_terms!(e, Vρ, func::Functional{:lda}, ρ::AbstractVector{T}) where {T}
#    res = ForwardDiff.gradient!(DiffResults.DiffResult(zero(eltype(e)), Vρ),
#                                ρ -> energy(func, ρ), ρ)
#    e .= DiffResults.value(res)
#    nothing
#end

function kernel_terms(func::Functional{:lda}, ρ::AbstractMatrix{T}) where {T}
    @assert has_energy(func)
    s_ρ, n_p = size(ρ)
    TT = arithmetic_type(func, T)

    indices = similar(ρ, Int, n_p)
    copyto!(indices, collect(1:n_p))

    e   = similar(ρ, TT, n_p)
    Vρ  = similar(ρ, TT, s_ρ, n_p)
    Vρρ = similar(ρ, TT, s_ρ, s_ρ, n_p)

    # TODO Needed to make forward-diff work with !isbits floating-point types (e.g. BigFloat)
    Vρ  .= zero(TT)
    Vρρ .= zero(TT)

    map!(e, indices) do i
        tmp = kernel_terms(func, SVector(ρ[1, i]))
        Vρ[1, i:i] .= tmp.Vρ
        Vρρ[1, i:i] .= tmp.Vρρ
        tmp.e
    end
    (; e, Vρ, Vρρ)
end
function kernel_terms(func::Functional{:lda}, ρ::AbstractVector{T}) where {T}
    res = ForwardDiff.hessian!(DiffResults.HessianResult(ρ), ρ -> energy(func, ρ), ρ)
    (; e = DiffResults.value(res), Vρ = DiffResults.gradient(res), Vρρ = DiffResults.hessian(res))
end
#function kernel_terms!(e, Vρ, Vρρ, func::Functional{:lda}, ρ::AbstractVector{T}) where {T}
#    res = ForwardDiff.hessian!(DiffResults.DiffResult(zero(eltype(e)), Vρ, Vρρ),
#                               ρ -> energy(func, ρ), ρ)
#    e .= DiffResults.value(res)
#    nothing
#end

function energy(func::Functional{:lda}, ρ::AbstractVector{T}) where {T}
    length(ρ) == 1 || error("Multiple spins not yet implemented for fallback functionals")
    ρtotal = ρ[1]
    if _val(ρtotal) <= _val(threshold_ρ(func, T))
        zero(T)
    else
        energy(func, ρtotal)
    end
end

#
# GGA
#
function potential_terms(func::Functional{:gga}, ρ::AbstractMatrix{T},
                         σ::AbstractMatrix{U}) where {T,U}
    @assert has_energy(func)  # Otherwise custom implementation of this function needed
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    TT = arithmetic_type(func, T, U)

    indices = similar(ρ, Int, n_p)
    copyto!(indices, collect(1:n_p))
    gpu_func = to_isbits(func)

    Vρ = similar(ρ, TT, s_ρ, n_p)
    Vσ = similar(ρ, TT, s_σ, n_p)
    e = similar(ρ, TT, n_p)
    map!(e, indices) do i
        #TODO: assume spin 1 for now
        tmp = potential_terms(gpu_func, SVector(ρ[1, i]), SVector(σ[1, i]))
        Vρ[1, i:i] .= tmp.Vρ
        Vσ[1, i:i] .= tmp.Vσ
        tmp.e
    end

    #@views for i = 1:n_p                                                                             
    #    potential_terms!(e[i:i], Vρ[:, i], Vσ[:, i], func, ρ[:, i], σ[:, i])
    #end
    (; e, Vρ, Vσ)
end
function potential_terms(func::Functional{:gga}, ρ::AbstractVector{T}, σ::AbstractVector{U}) where{T, U}
    #TODO: need all Dual types to be the same upon entry, or GPU compilation fails
    function energy_ρ(x::AbstractVector{T}) where {T}
        new_params = map(p -> T(p), func.parameters)
        energy(change_parameters(func, new_params), x, SVector(T(σ[1])))
    end
    res_ρ = ForwardDiff.gradient!(DiffResults.GradientResult(ρ), energy_ρ, ρ)
    function energy_σ(x::AbstractVector{U}) where {U}
        new_params = map(p -> U(p), func.parameters)
        energy(change_parameters(func, new_params), SVector(U(ρ[1])), x)
    end
    res_σ = ForwardDiff.gradient!(DiffResults.GradientResult(σ), energy_σ, σ)
    (; e = DiffResults.value(res_ρ), Vρ = DiffResults.gradient(res_ρ), Vσ = DiffResults.gradient(res_σ))
end
#function potential_terms!(e, Vρ, Vσ, func::Functional{:gga},
#                          ρ::AbstractVector, σ::AbstractVector)
#    res = ForwardDiff.gradient!(DiffResults.DiffResult(zero(eltype(e)), Vρ),
#                                ρ -> energy(func, ρ, σ), ρ)
#    ForwardDiff.gradient!(DiffResults.DiffResult(zero(eltype(e)), Vσ),
#                          σ -> energy(func, ρ, σ), σ)
#    e .= DiffResults.value(res)
#    nothing
#end

function kernel_terms(func::Functional{:gga}, ρ::AbstractMatrix{T},
                      σ::AbstractMatrix{U}) where {T,U}
    @assert has_energy(func)  # Otherwise custom implementation of this function needed
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    TT = arithmetic_type(func, T, U)

    indices = similar(ρ, Int, n_p)
    copyto!(indices, collect(1:n_p))
    gpu_func = to_isbits(func)

    e   = similar(ρ, TT, n_p)
    Vρ  = similar(ρ, TT, s_ρ, n_p)
    Vσ  = similar(ρ, TT, s_σ, n_p)
    Vρρ = similar(ρ, TT, s_ρ, s_ρ, n_p)
    Vρσ = similar(ρ, TT, s_ρ, s_σ, n_p)
    Vσσ = similar(ρ, TT, s_σ, s_σ, n_p)

    # TODO Needed to make forward-diff work with !isbits floating-point types (e.g. BigFloat)
    Vρ  .= zero(TT)
    Vσ  .= zero(TT)
    Vρρ .= zero(TT)
    Vρσ .= zero(TT)
    Vσσ .= zero(TT)

    map!(e, indices) do i
        tmp = kernel_terms(gpu_func, SVector(ρ[1, i]), SVector(σ[1, i]))
        Vρ[1, i:i] .= tmp.Vρ
        Vσ[1, i:i] .= tmp.Vσ
        Vρρ[1, 1, i:i] .= tmp.Vρρ
        Vσσ[1, 1, i:i] .= tmp.Vσσ
        Vρσ[1, 1, i:i] .= tmp.Vρσ
        tmp.e
    end

    #@views for i = 1:n_p
    #    kernel_terms!(e[i:i], Vρ[:, i], Vσ[:, i],
    #                  Vρρ[:, :, i], Vρσ[:, :, i], Vσσ[:, :, i],
    #                  func, ρ[:, i], σ[:, i])
    #end
    (; e, Vρ, Vσ, Vρρ, Vρσ, Vσσ)
end
function kernel_terms(func::Functional{:gga}, ρ::AbstractVector{T}, σ::AbstractVector{U}) where {T, U}
    function energy_ρ(x::AbstractVector{T}) where {T}
        new_params = map(p -> T(p), func.parameters)
        energy(change_parameters(func, new_params), x, SVector(T(σ[1])))
    end
    res_ρ = ForwardDiff.hessian!(DiffResults.HessianResult(ρ), energy_ρ, ρ)
    function energy_σ(x::AbstractVector{U}) where {U}
        new_params = map(p -> U(p), func.parameters)
        energy(change_parameters(func, new_params), SVector(U(ρ[1])), x)
    end
    res_σ = ForwardDiff.hessian!(DiffResults.HessianResult(σ), energy_σ, σ)

    #TODO: one should probably try and remain general, and do the fancy thing here too. AMD?
    dedρ_σ = σ -> ForwardDiff.gradient(ρ -> energy(func, ρ, σ), ρ)
    res_ρσ = ForwardDiff.jacobian!(DiffResults.JacobianResult(σ), dedρ_σ, σ)

    (; e = DiffResults.value(res_ρ), Vρ = DiffResults.gradient(res_ρ), Vρρ = DiffResults.hessian(res_ρ),
       Vσ = DiffResults.gradient(res_σ), Vσσ = DiffResults.hessian(res_σ), Vρσ = DiffResults.jacobian(res_ρσ))
end
#function kernel_terms!(e, Vρ, Vσ, Vρρ, Vρσ, Vσσ, func::Functional{:gga},
#                       ρ::AbstractVector, σ::AbstractVector)
#    res = ForwardDiff.hessian!(DiffResults.DiffResult(zero(eltype(e)), Vρ, Vρρ),
#                               ρ -> energy(func, ρ, σ), ρ)
#    res = ForwardDiff.hessian!(DiffResults.DiffResult(zero(eltype(e)), Vσ, Vσσ),
#                               σ -> energy(func, ρ, σ), σ)
#
#    dedρ = σ -> ForwardDiff.gradient(ρ -> energy(func, ρ, σ), ρ)
#    ForwardDiff.jacobian!(Vρσ, dedρ, σ)
#
#    e .= DiffResults.value(res)
#    nothing
#end

#TODO: here we need T and U to be different, as it goes into FD. But at the functional level, we can probably
#      assume we use the same type. Same for the FD map! kernel, U and T should be the same at that point
#      As things stand, energy is always called with the same types, due to GPU compilation restrictions
#      might as well assume same type everywhere. To be fair, that could also serve as a check
#TODO: need to explicitly acces Dual values for comparisons to work on the GPU. Overload <, > and == ?
function energy(func::Functional{:gga}, ρ::AbstractVector{T},
                σ::AbstractVector{U}) where {T,U}
    length(ρ) == 1 || error("Multiple spins not yet implemented for fallback functionals")
    @assert length(ρ) == 1
    TT = arithmetic_type(func, T, U)

    ρtotal = TT(ρ[1])
    σtotal = TT(σ[1])
    if _val(ρtotal) <= _val(threshold_ρ(func, T))
        zero(TT) #arithmetic_type(func, T, U))
    else
	σstable = _val(σtotal) > _val(threshold_σ(func, U)) ? σtotal : threshold_σ(func, U)
        energy(func, ρtotal, σstable)
    end
end
