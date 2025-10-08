module CosmoFFTs

using Logging
using Base.Threads
using FFTW

include(joinpath(@__DIR__, "Utils", "Wisdom.jl"))
using .Wisdom: wisdom_filename, load_wisdom, save_wisdom, reset_wisdom

include(joinpath(@__DIR__, "BackendDispatch.jl"))
using .BackendDispatch: FFTPlanSpec, FourierArrayInfo, allocate_fields

# Check if MPI and PencilFFTs are available for the MPI backend
# These will be loaded lazily only when needed
const HAVE_MPI = Ref(false)
const HAVE_PENCILFFT = Ref(false)

const DEFAULT_BACKEND = Ref(:fftw_single)
const DEFAULT_THREADS = Ref(1)
const WISDOM_FILE = Ref{String}("")

export FFTPlanSpec, FourierArrayInfo, allocate_fields,
       make_plan, forwardFT!, inverseFT!,
       default_backend, default_threads, set_backend!, set_threads!, reinitialize!

default_backend() = DEFAULT_BACKEND[]
default_threads() = DEFAULT_THREADS[]

function set_backend!(backend::Symbol)
    DEFAULT_BACKEND[] = backend
    return backend
end

function set_threads!(nthreads::Integer)
    DEFAULT_THREADS[] = Int(nthreads)
    return DEFAULT_THREADS[]
end

"""
    reinitialize!()

Reinitialize the CosmoFFTs backend. Call this after changing backend or threads
settings with `set_backend!()` or `set_threads!()` to apply the changes.

# Example
```julia
using CosmoFFTs
set_backend!(:pencil_mpi)
reinitialize!()  # Load the PencilFFTs MPI backend
```
"""
function reinitialize!()
    __init__()
    return nothing
end

function make_plan(spec::BackendDispatch.FFTPlanSpec)
    return BackendDispatch.make_plan(spec; backend=DEFAULT_BACKEND[])
end

function make_plan(; dims::NTuple{3,Int}, boxsize::NTuple{3,Real},
                   backend::Symbol=DEFAULT_BACKEND[], comm=nothing)
    spec = FFTPlanSpec(dims, boxsize; comm=comm, backend=backend)
    return make_plan(spec)
end

function forwardFT!(out, plan, inp)
    return BackendDispatch.forwardFT!(out, plan, inp; backend=DEFAULT_BACKEND[])
end

function inverseFT!(out, plan, inp)
    return BackendDispatch.inverseFT!(out, plan, inp; backend=DEFAULT_BACKEND[])
end

function load_backend_module(filepath::String)
    # Load backend module at runtime (NOT during precompilation)
    # This should only be called from __init__() after checking we're not precompiling
    fullpath = joinpath(@__DIR__, filepath)
    modname = Symbol(basename(filepath)[1:end-3])

    if !isdefined(BackendDispatch, modname)
        # Direct include works fine at runtime, just not during precompilation
        Base.include(BackendDispatch, fullpath)

        if !isdefined(BackendDispatch, modname)
            error("Failed to load backend module $modname from $filepath")
        end
    end
    return modname
end

include(joinpath(@__DIR__, "__init__.jl"))

end
