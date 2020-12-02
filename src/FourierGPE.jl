module FourierGPE

using Reexport
using SpecialFunctions
using LazyArrays
using FillArrays
using DifferentialEquations
using LoopVectorization
using Tullio
using RecursiveArrayTools
using PaddedViews
using LaTeXStrings
using ColorSchemes

@reexport using FFTW
@reexport using Parameters
@reexport using JLD2
@reexport using FileIO
using Plots
const c1 = cgrad(ColorSchemes.inferno.colors)
const c2 = cgrad(ColorSchemes.RdBu_11.colors)
const c3 =:mediumseagreen

include("types.jl")
include("arrays.jl")
include("transforms.jl")
include("evolution.jl")
include("analysis.jl")

# simulation
export Simulation, TransformLibrary, UserParams
export xvec, kvec, xvecs, kvecs, dfft, dfftall, crandn_array, crandnpartition 
export maketransarrays, makearrays, xspace, xspace!, kspace, kspace!
export nlin, nlin!, Lgp, Lgp!, V, Params
export initsim!, runsim, internalnorm
export Transforms, @pack_Transforms!, @unpack_Transforms
export Sim, @pack_Sim!, @unpack_Sim, testsim
export showpsi, c1, c2, c3
export Params, @pack_Params!, @unpack_Params
export k2, @pack!, @unpack, makeT, definetransforms #makeTMixed,
export Field, XField, KField, gradient, velocity, current
# analysis
export energydecomp, helmholtz, kinetic_density
export incompressible_spectrum, compressible_spectrum, qpressure_spectrum
export incompressible_density, compressible_density, qpressure_density
export ic_density, iq_density, cq_density
export bessel_reduce, sinc_reduce, gv
export log10range, zeropad, autocorrelate, convolve
end # module
