# ==============================================================================
# >> BackendDispatch.jl
#
# This file handles all FFT-backend-dependent dispatching and utilities,
# including:
#   - make_plan()  (FFTW single, threaded, PencilMPI)
#   - forwardFT!, inverseFT!
#   - allocate_input/output() for PencilFFTs
#   - FourierArrayInfo struct and backend constructors
#
# By isolating these here, the Core physics code stays clean and backend-agnostic.
# ==============================================================================
module BackendDispatch
# ==============================================================================

export FFTPlanSpec, make_plan, forwardFT!, inverseFT!,
       FourierArrayInfo, allocate_fields


Base.@kwdef struct FFTPlanSpec
    dims::NTuple{3,Int}
    boxsize::NTuple{3,Float64}
    volume::Float64
    comm::Union{Nothing,Any} = nothing
    backend::Symbol = :fftw_single
end

# Always include FFTW backends (FFTW is a hard dependency)
include(joinpath(@__DIR__,"Backends","FFTW_Single.jl"))
include(joinpath(@__DIR__,"Backends","FFTW_Threads.jl"))
using .FFTW_Single
using .FFTW_Threads

# MPI/PencilFFTs backend will be loaded conditionally at __init__ time
# The module will be included dynamically if MPI is available

# Helper functions for PencilFFTs API compatibility
# These operate on the wrapped Plan struct and extract the inner PencilFFTPlan
# Will only work if PencilFFTs backend is loaded
function size_in end
function size_out end
function pencil_in end
function pencil_out end



function FFTPlanSpec(dims::NTuple{3,Int}, boxsize::NTuple{3,Real};
                     comm=nothing, backend=:fftw_single)
    bsize = ntuple(i -> Float64(boxsize[i]), 3)
    return FFTPlanSpec(; dims=dims, boxsize=bsize, volume=prod(bsize),
                comm=comm, backend=backend)
end


function forwardFT!(out, plan, inp; backend=:fftw_single)
    if backend == :fftw_single
        return FFTW_Single.forwardFT!(out, plan, inp)
    elseif backend == :fftw_threads
        return FFTW_Threads.forwardFT!(out, plan, inp)
    elseif backend == :pencil_mpi
        return Base.invokelatest(PencilFFTs_MPI.forwardFT!, out, plan, inp)
    end
end


function inverseFT!(out, plan, inp; backend=:fftw_single)
    if backend == :fftw_single
        return FFTW_Single.inverseFT!(out, plan, inp)
    elseif backend == :fftw_threads
        return FFTW_Threads.inverseFT!(out, plan, inp)
    elseif backend == :pencil_mpi
        return Base.invokelatest(PencilFFTs_MPI.inverseFT!, out, plan, inp)
    end
end

function make_plan(spec::FFTPlanSpec; backend=:fftw_single)
    if spec.backend == :fftw_single
        return FFTW_Single.make_plan(spec)
    elseif spec.backend == :fftw_threads
        return FFTW_Threads.make_plan(spec)
    elseif spec.backend == :pencil_mpi
        return Base.invokelatest(PencilFFTs_MPI.make_plan, spec)
    else
        error("Unknown backend $(spec.backend)")
    end
end
# ====================================================================
struct FourierArrayInfo
    n1::Int
    n2::Int
    n3::Int
    Ntotal::Int
    cn1::Int
    cn2::Int
    cn3::Int
    L1::Float64
    L2::Float64
    L3::Float64
    Volume::Float64
    # Fourier-space arrays: k
    kF1::Float64
    kF2::Float64
    kF3::Float64
    ak1::Vector{Float64}
    ak2::Vector{Float64}
    ak3::Vector{Float64}
    aik1::Vector{Int}
    aik2::Vector{Int}
    aik3::Vector{Int}
    # real-space arrays: x
    xH1::Float64
    xH2::Float64
    xH3::Float64
    ax1::Vector{Float64}
    ax2::Vector{Float64}
    ax3::Vector{Float64}
    aix1::Vector{Int}
    aix2::Vector{Int}
    aix3::Vector{Int}
    vk1::AbstractArray{Float64,3}
    vk2::AbstractArray{Float64,3}
    vk3::AbstractArray{Float64,3}
    akmag::AbstractArray{Float64,3}
end
# -----------------------------------------------------------------------------
function calcWavenumbers!(ak::AbstractVector{T}, kF::T, n::Int, cn::Int) where {T<:Real}
    @inbounds for indx = 1:cn
        ak[indx] = kF*(indx-1)
    end
    if(n > cn)
        @inbounds for indx = cn+1:n
            ak[indx] = kF*(indx-1-n)
        end
    end
end
# -----------------------------------------------------------------------------
function calcWaveindices!(aik::AbstractVector{T}, n::Int, cn::Int) where {T<:Integer}
    @inbounds for indx = 1:cn
        aik[indx] = (indx-1)
    end
    if(n > cn)
        @inbounds for indx = cn+1:n
            aik[indx] = (indx-1-n)
        end
    end
end
# -----------------------------------------------------------------------------
function FourierArrayInfo(spec::FFTPlanSpec;plan=nothing)
    n1,n2,n3 = spec.dims
    L1,L2,L3 = spec.boxsize

    Ntotal = prod(spec.dims)
    Volume = prod(spec.boxsize)
    
    # to compute the maximum 1D wavelength
    cn1 = div(n1,2)+1
    cn2 = div(n2,2)+1
    cn3 = div(n3,2)+1
    # Fundamental frequencies
    kF1 = 2π/L1; kF2 = 2π/L2; kF3 = 2π/L3
    # 1D Fourier arrays
    ak1 = Vector{Float64}(undef, cn1)
    ak2 = Vector{Float64}(undef, n2)
    ak3 = Vector{Float64}(undef, n3)
    calcWavenumbers!(ak1, kF1, cn1, cn1)
    calcWavenumbers!(ak2, kF2, n2, cn2)
    calcWavenumbers!(ak3, kF3, n3, cn3)
    aik1 = Vector{Int}(undef, cn1)
    aik2 = Vector{Int}(undef, n2)
    aik3 = Vector{Int}(undef, n3)
    calcWaveindices!(aik1, cn1, cn1)
    calcWaveindices!(aik2, n2, cn2)
    calcWaveindices!(aik3, n3, cn3)

    # real-space spacing
    xH1 = L1 / n1; xH2 = L2 / n2; xH3 = L3 / n3
    # 1D real-space arrays
    aix1 = collect(1:n1)
    aix2 = collect(1:n2)
    aix3 = collect(1:n3)
    ax1 = Vector{Float64}(undef, n1)
    ax2 = Vector{Float64}(undef, n2)
    ax3 = Vector{Float64}(undef, n3)
    @inbounds for i in eachindex(aix1)
        ax1[i] = aix1[i] * xH1
    end
    @inbounds for i in eachindex(aix2)
        ax2[i] = aix2[i] * xH2
    end
    @inbounds for i in eachindex(aix3)
        ax3[i] = aix3[i] * xH3
    end

    # For PencilFFTs (MPI), we compute the local array
    if spec.backend == :pencil_mpi
        @assert plan !== nothing "PencilFFT backend requires a valid plan"

        # --- Allocate a temporary k-space array to get the correct memory layout
        # PencilArrays may have dimension permutations, so we need to match that
        temp_k = allocate_output(plan.plan)
        mem_dims = size(parent(temp_k))  # Memory-order dimensions

        # --- Local dimensions (complex space) in logical order
        local_dims_k = size_out(plan)
        pen_k = pencil_out(plan)
        i1_range, i2_range, i3_range = range_local(pen_k)

        # Local Fourier 1D arrays in logical order
        ak1_loc = ak1[i1_range]  # Local k1 wavenumbers
        ak2_loc = ak2[i2_range]  # Local k2 wavenumbers
        ak3_loc = ak3[i3_range]  # Local k3 wavenumbers
        aik1_loc = aik1[i1_range]  # Local k1 indices
        aik2_loc = aik2[i2_range]  # Local k2 indices
        aik3_loc = aik3[i3_range]  # Local k3 indices

        # --- Local real-space dimensions
        pen_x = pencil_in(plan)
        local_dims_x = size_in(plan)
        ix1_range, ix2_range, ix3_range = range_local(pen_x)

        aix1_loc = aix1[ix1_range]
        aix2_loc = aix2[ix2_range]
        aix3_loc = aix3[ix3_range]
        ax1_loc = ax1[ix1_range]
        ax2_loc = ax2[ix2_range]
        ax3_loc = ax3[ix3_range]

        # --- Construct local k-grids and k-magnitude in MEMORY ORDER
        # Work directly with parent arrays to match memory layout
        parent_k = parent(temp_k)
        vk1 = similar(parent_k, Float64)
        vk2 = similar(parent_k, Float64)
        vk3 = similar(parent_k, Float64)
        akmag = similar(parent_k, Float64)

        # Fill using explicit loops - straightforward and always works
        # Create 1D k-value arrays
        k1_vals = [ak1[i] for i in i1_range]
        k2_vals = [ak2[j] for j in i2_range]
        k3_vals = [ak3[k] for k in i3_range]

        # Create temporary PencilArrays
        temp_k_real = similar(temp_k, Float64)
        vk1_pa = similar(temp_k_real)
        vk2_pa = similar(temp_k_real)
        vk3_pa = similar(temp_k_real)
        akmag_pa = similar(temp_k_real)

        # Fill in logical index space - PencilArray handles the permutation
        for k in 1:length(k3_vals), j in 1:length(k2_vals), i in 1:length(k1_vals)
            vk1_pa[i,j,k] = k1_vals[i]
            vk2_pa[i,j,k] = k2_vals[j]
            vk3_pa[i,j,k] = k3_vals[k]
            akmag_pa[i,j,k] = hypot(k1_vals[i], k2_vals[j], k3_vals[k])
        end

        # Keep as PencilArrays so that indexing works correctly with logical indices
        # When used in broadcasting or indexing with deltak (PencilArray), dimensions will match
        return FourierArrayInfo(
            n1, n2, n3, Ntotal, cn1, cn2, cn3,
            L1, L2, L3, Volume,
            kF1, kF2, kF3,
            ak1_loc, ak2_loc, ak3_loc,
            aik1_loc, aik2_loc, aik3_loc,
            xH1, xH2, xH3,
            ax1_loc, ax2_loc, ax3_loc,
            aix1_loc, aix2_loc, aix3_loc,
            vk1_pa, vk2_pa, vk3_pa,
            akmag_pa
        )
    end
    
    # Default: FFTW (serial / threads)         
    vk1   = ones(Float64, cn1, 1, 1)
    vk2   = ones(Float64, 1, n2, 1)
    vk3   = ones(Float64, 1, 1, n3)
    akmag = Array{Float64}(undef, cn1, n2, n3)
    
    vk1 .*= ak1
    @view(vk2[1,:,1]) .= ak2
    @view(vk3[1,1,:]) .= ak3
    @inbounds @. akmag = hypot(vk1, vk2, vk3)

    return FourierArrayInfo(n1,n2,n3,Ntotal,cn1,cn2,cn3,
        L1,L2,L3,Volume,
        kF1,kF2,kF3,
        ak1,ak2,ak3,
        aik1,aik2,aik3,
        xH1,xH2,xH3,
        ax1,ax2,ax3,
        aix1,aix2,aix3,
        vk1,vk2,vk3,
        akmag)
end
# ------------------------------------------------------------------------------
# Internal helper: allocate arrays
# ------------------------------------------------------------------------------
"""
    allocate_fields(cfg::LNConfig, fftplan)

Allocate real-space and Fourier-space arrays for the mock field.
For PencilFFTs (MPI), this uses `allocate_input` and `allocate_output`.
For FFTW backends, it uses standard Array allocation.
"""
function allocate_fields(spec::FFTPlanSpec;fftplan=nothing)
    n1, n2, n3 = spec.dims

    if spec.backend == :pencil_mpi
       @assert fftplan !== nothing "PencilFFT backend requires a valid plan"
        # PencilFFTs handles local array shapes internally
        # Extract the inner PencilFFTPlan from the wrapped Plan struct
        inner_plan = fftplan.plan
        field_real    = allocate_input(inner_plan)
        field_fourier = allocate_output(inner_plan)
    else
        field_real    = Array{Float64}(undef, n1, n2, n3)
        field_fourier = Array{ComplexF64}(undef, div(n1,2)+1, n2, n3)
    end

    return field_real, field_fourier
end

# ==============================================================================
end
# ==============================================================================
