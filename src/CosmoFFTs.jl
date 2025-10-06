module CosmoFFTs

using Logging
using Base.Threads
using FFTW

include(joinpath(@__DIR__, "Utils", "Wisdom.jl"))
using .Wisdom: wisdom_filename, load_wisdom, save_wisdom, reset_wisdom

include(joinpath(@__DIR__, "BackendDispatch.jl"))
using .BackendDispatch: FFTPlanSpec, FourierArrayInfo, allocate_fields

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

function load_backend(modsyms, filepath::String; force::Bool=false)
    for sym in (modsyms isa Tuple || modsyms isa Vector ? modsyms : (modsyms,))
        @eval using $(sym)
    end
    fullpath = joinpath(@__DIR__, filepath)
    modname = Symbol(basename(filepath)[1:end-3])
    if force || !isdefined(BackendDispatch, modname)
        Base.include(BackendDispatch, fullpath)
        # After including, the module should be defined in BackendDispatch
        # Make it accessible via qualified names by evaluating in BackendDispatch context
        if isdefined(BackendDispatch, modname)
            Core.eval(BackendDispatch, :(using .$modname))
        else
            error("Failed to load backend module $modname from $filepath")
        end
    end
    return nothing
end

include(joinpath(@__DIR__, "__init__.jl"))

end
