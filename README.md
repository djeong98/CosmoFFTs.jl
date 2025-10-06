# CosmoFFTs.jl

CosmoFFTs.jl provides the FFT planning and dispatch utilities that power the LNMock lognormal mock generator. It packages real/complex FFT plan construction, backend abstraction (single-threaded FFTW, threaded FFTW, PencilFFTs MPI), and helper routines such as Fourier grid metadata and array allocation into a reusable library.

## Quick start

```julia
julia> using CosmoFFTs
julia> spec = FFTPlanSpec((64, 64, 64), (100.0, 100.0, 100.0))
julia> plan = make_plan(spec)
julia> real, fourier = allocate_fields(spec)
julia> forwardFT!(fourier, plan, real)
```

By default the package selects the single-threaded FFTW backend. Choose a different backend with:

```julia
julia> set_backend!(:fftw_threads)
julia> set_threads!(4)
```

To use the MPI pencil-decomposed backend, load MPI and PencilFFTs prior to using the package and select `:pencil_mpi`.

## FFTW wisdom

Wisdom files are stored beneath a local `FFTWwisdom/` directory in the current working directory. Set the environment variable `COSMOFFTS_WISDOM_RESET=1` (legacy aliases `LNFFT_WISDOM_RESET=1` or `LN_WISDOM_RESET=1` also work) to clear wisdom during initialization.

## Environment overrides

- `COSMOFFTS_BACKEND` (legacy: `LNFFT_BACKEND`, `LN_BACKEND`): select `:fftw_single`, `:fftw_threads`, or `:pencil_mpi`
- `COSMOFFTS_THREADS` (legacy: `LNFFT_THREADS`, `LN_THREADS`): override FFTW thread count for the threaded or PencilFFTs backends
- `COSMOFFTS_WISDOM_RESET` (legacy: `LNFFT_WISDOM_RESET`, `LN_WISDOM_RESET`): reset stored FFTW wisdom

## Testing

Run the package tests with:

```julia
julia --project=CosmoFFTs -e 'using Pkg; Pkg.test()'
```

MPI tests are not included by default; set `set_backend!(:pencil_mpi)` in your own scripts to exercise the distributed backend.
