
using Test, Random
using CosmoFFTs

const HAS_MPI = let available = false
    try
        @eval using MPI, PencilFFTs
        available = true
    catch
        available = false
    end
    available
end

function with_backend(f, backend::Symbol, threads::Integer=1)
    old_backend = CosmoFFTs.default_backend()
    old_threads = CosmoFFTs.default_threads()
    CosmoFFTs.set_backend!(backend)
    CosmoFFTs.set_threads!(threads)
    CosmoFFTs.__init__()
    try
        return f()
    finally
        CosmoFFTs.set_backend!(old_backend)
        CosmoFFTs.set_threads!(old_threads)
        CosmoFFTs.__init__()
    end
end

@testset "FFTW single backend" begin
    with_backend(:fftw_single, 1) do
        spec = FFTPlanSpec((4, 4, 4), (1.0, 1.0, 1.0))
        plan = CosmoFFTs.make_plan(spec)

        real, freq = CosmoFFTs.allocate_fields(spec)
        fill!(real, 0.0)
        fill!(freq, 0.0 + 0im)

        @test size(real) == (4, 4, 4)
        @test size(freq) == (3, 4, 4)

        CosmoFFTs.forwardFT!(freq, plan, real)
        work = similar(real)
        CosmoFFTs.inverseFT!(work, plan, freq)
        @test isapprox(work, real; atol=1e-12, rtol=1e-12)

        finfo = CosmoFFTs.FourierArrayInfo(spec)
        @test finfo.n1 == 4
        @test length(finfo.ak1) == 3
    end
end

@testset "FFTW threaded backend" begin
    with_backend(:fftw_threads, max(2, Base.Threads.nthreads())) do
        spec = FFTPlanSpec((6, 8, 4), (1.0, 1.0, 1.0); backend=:fftw_threads)
        plan = CosmoFFTs.make_plan(spec)
        real, freq = CosmoFFTs.allocate_fields(spec)
        Random.rand!(real)
        original = copy(real)
        CosmoFFTs.forwardFT!(freq, plan, real)
        work = similar(real)
        CosmoFFTs.inverseFT!(work, plan, freq)
        @test isapprox(work, original; atol=1e-10, rtol=1e-10)
    end
end

if HAS_MPI
    @testset "PencilFFTs MPI backend" begin
        with_backend(:pencil_mpi, 1) do
            if !MPI.Initialized()
                @test_skip "MPI failed to initialize"
            else
                comm = MPI.COMM_WORLD
                spec = FFTPlanSpec((8, 8, 8), (1.0, 1.0, 1.0); backend=:pencil_mpi, comm=comm)
                plan = CosmoFFTs.make_plan(spec)
                real, freq = CosmoFFTs.allocate_fields(spec; fftplan=plan)
                Random.rand!(real)
                original = copy(real)
                work = similar(real)

                CosmoFFTs.forwardFT!(freq, plan, real)
                CosmoFFTs.inverseFT!(work, plan, freq)

                local_err = maximum(abs.(work .- original))
                global_err = MPI.Allreduce(local_err, MPI.MAX, comm)
                @test global_err < 1e-8
            end
        end
    end
else
    @testset "PencilFFTs MPI backend" begin
        @test_skip "MPI/PencilFFTs not available"
    end
end
