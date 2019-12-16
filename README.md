# FourierGPE.jl
Simple GPE solver using FFTW

- Intended to provide simple, fast, and flexible modelling of Bose-Einstein condensate experiments.
- Allocation free, adaptive evolution in kspace.
- Establish some useful defaults and runnable examples for time evolution using `DifferentialEquations.jl`
- Arbitrary spatial dimensions.
- Not de-aliased (projective methods available elsewhere)

## Installing

```julia
]add https://github.com/AshtonSBradley/FourierGPE.jl.git
using FourierGPE
```
## Enabling MKL
```julia
using Pkg;
ENV["JULIA_FFTW_PROVIDER"]="MKL"
Pkg.build("FFTW")
```
Note that MKL for FFTW is provided independently of the package [MKL.jl](https://github.com/JuliaComputing/MKL.jl).

## Getting started
To get started see runnable examples in the `/examples` directory, or for more information see [FGPEexamples.jl](https://github.com/AshtonSBradley/FGPEexamples.jl)

<img src="https://github.com/AshtonSBradley/FGPEexamples.jl/blob/master/media/3dquenchslab.gif" width="900">
