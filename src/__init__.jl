function __init__()
    # Skip backend initialization during precompilation
    # This avoids Julia 1.11+ errors about eval into closed modules
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing

    global_logger(ConsoleLogger(stderr, Logging.Info))

    backend_env = get(ENV, "COSMOFFTS_BACKEND", nothing)
    backend_env === nothing && (backend_env = get(ENV, "LNFFT_BACKEND", nothing))
    backend_env === nothing && (backend_env = get(ENV, "LN_BACKEND", nothing))
    if backend_env !== nothing
        DEFAULT_BACKEND[] = Symbol(String(backend_env))
    end

    threads_env = get(ENV, "COSMOFFTS_THREADS", nothing)
    threads_env === nothing && (threads_env = get(ENV, "LNFFT_THREADS", nothing))
    threads_env === nothing && (threads_env = get(ENV, "LN_THREADS", nothing))
    if threads_env !== nothing
        try
            DEFAULT_THREADS[] = parse(Int, threads_env)
        catch e
            @warn "Invalid threads setting $(threads_env); falling back to 1" exception=(e, catch_backtrace())
            DEFAULT_THREADS[] = 1
        end
    end

    WISDOM_FILE[] = wisdom_filename(nthreads=DEFAULT_THREADS[])
    reset_flag = get(ENV, "COSMOFFTS_WISDOM_RESET", get(ENV, "LNFFT_WISDOM_RESET", get(ENV, "LN_WISDOM_RESET", "0")))
    reset_flag == "1" && reset_wisdom(WISDOM_FILE[])

    backend = DEFAULT_BACKEND[]

    if backend == :fftw_single
        @info "Initializing FFTW single-thread backend"
        FFTW.set_num_threads(1)
        load_wisdom(WISDOM_FILE[])
        atexit(() -> save_wisdom(WISDOM_FILE[]))
    elseif backend == :fftw_threads
        @info "Initializing FFTW threaded backend" threads=DEFAULT_THREADS[]
        FFTW.set_num_threads(DEFAULT_THREADS[])
        load_wisdom(WISDOM_FILE[])
        atexit(() -> save_wisdom(WISDOM_FILE[]))
    elseif backend == :pencil_mpi
        @info "Initializing PencilFFTs (MPI) backend"

        # Load MPI and PencilFFTs at runtime (not during precompilation)
        if !HAVE_MPI[]
            try
                Base.require(Main, :MPI)
                HAVE_MPI[] = true
            catch e
                @error "Failed to load MPI for pencil_mpi backend" exception=(e, catch_backtrace())
                rethrow()
            end
        end

        if !HAVE_PENCILFFT[]
            try
                Base.require(Main, :PencilFFTs)
                HAVE_PENCILFFT[] = true
            catch e
                @error "Failed to load PencilFFTs for pencil_mpi backend" exception=(e, catch_backtrace())
                rethrow()
            end
        end

        # Load the backend module at runtime (safe now that we're not precompiling)
        load_backend_module("Backends/PencilFFTs_MPI.jl")

        # Import PencilFFTs functions into BackendDispatch
        Core.eval(BackendDispatch, quote
            import PencilFFTs: allocate_input, allocate_output, pencil, range_local
            size_in(plan) = size(allocate_input(plan.plan))
            size_out(plan) = size(allocate_output(plan.plan))
            pencil_in(plan) = pencil(allocate_input(plan.plan))
            pencil_out(plan) = pencil(allocate_output(plan.plan))
        end)

        FFTW.set_num_threads(DEFAULT_THREADS[])
        load_wisdom(WISDOM_FILE[])
        try
            MPI_mod = Base.require(Main, :MPI)
            if !Base.invokelatest(getfield(MPI_mod, :Initialized))
                Base.invokelatest(getfield(MPI_mod, :Init))
                atexit(() -> begin
                    if Base.invokelatest(getfield(MPI_mod, :Initialized)) && !Base.invokelatest(getfield(MPI_mod, :Finalized))
                        Base.invokelatest(getfield(MPI_mod, :Finalize))
                    end
                end)
            end
            @info "PencilFFTs backend initialized" ranks=Base.invokelatest(getfield(MPI_mod, :Comm_size), getfield(MPI_mod, :COMM_WORLD))
        catch e
            @warn "MPI initialization failed" exception=(e, catch_backtrace())
        end
        atexit(() -> save_wisdom(WISDOM_FILE[]))
    else
        @warn "Unknown backend $(backend); defaulting to :fftw_single"
        DEFAULT_BACKEND[] = :fftw_single
        return __init__()
    end

    return nothing
end
