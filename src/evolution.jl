"""
    initsim!(sim;flags=FFTW.MEASURE)

Initialize all arrays, measures and transform libraries for a
particular simulation.
"""
function initsim!(sim;flags=FFTW.MEASURE)
    @unpack L,N = sim
    X,K,dX,dK,DX,DK,T = makearraystransforms(L,N)
    espec = 0.5*k2(L,N)
    @pack! sim = T,X,K,espec
    return nothing
end

"""
    V(x,t) = ...

Define the system potential, with default zero. Should be defined as a scalar
function (without `.`), suitable for broadcasting on spatial arrays via `V.(...)`.
"""
V(x,t) = 0.0
V(x,y,t) = 0.0
V(x,y,z,t) = 0.0

"""
    χ = nlin(ϕ,sim,t)

Evalutes nonlinear terms in `x`-space, returning to `k`-space.
"""
function nlin(ϕ,sim,t)
    @unpack g,x,y = sim
    ψ = xspace(ϕ,sim)
    ψ .*= @. g*abs2(ψ) + V(x,y',t)
    return kspace(ψ,sim)
end

"""
    nlin!(ϕ,sim::Sim{D},t)

Mutating evaluation of position space nonlinear terms.
Dispatches on dimension `D`, using potential `V(x...,t)`.
"""
function nlin!(dϕ,ϕ,sim::Sim{1},t)
    @unpack g,X,V0 = sim; x = X[1]
    dϕ .= ϕ
    xspace!(dϕ,sim)
    @. dϕ *= V0 + V(x,t) + g*abs2(dϕ)
    kspace!(dϕ,sim)
    return nothing
end

function nlin!(dϕ,ϕ,sim::Sim{2},t)
    @unpack g,X,V0 = sim; x,y = X
    y = y'
    dϕ .= ϕ
    xspace!(dϕ,sim)
    @. dϕ *= V0 + V(x,y,t) + g*abs2(dϕ)
    kspace!(dϕ,sim)
    return nothing
end

function nlin!(dϕ,ϕ,sim::Sim{3},t)
    @unpack g,X,V0 = sim; x,y,z = X
    y = y'; z = reshape(z,(1,1,length(z)))
    dϕ .= ϕ
    xspace!(dϕ,sim)
    @. dϕ *= V0 + V(x,y,z,t) + g*abs2(dϕ)
    kspace!(dϕ,sim)
    return nothing
end

"""
    χ = Lgp!(dϕ,ϕ,sim,t)

In-place evaluation of Gross-Pitaevskii equation defined by `nlin!` and parameters in `sim`.
Allows imaginary time `γ`, and evolves in rotating frame defined by chemical potential `μ`.
"""
function Lgp!(dϕ,ϕ,sim,t)
    @unpack γ,μ,espec = sim
    nlin!(dϕ,ϕ,sim,t)
    @. dϕ = -im*(1.0 - im*γ)*(dϕ + (espec - μ)*ϕ)
    return nothing
end

"""
    χ = Lgp(ϕ,sim,t)

Evaluate Gross-Pitaevskii equation defined by `nlin` and parameters in `sim`.
Allows imaginary time `γ`, and evolves in rotating frame defined by chemical potential `μ`.
"""
function Lgp(ϕ,sim,t)
    @unpack μ,γ,espec = sim
    chi = nlin(ϕ,sim,t)
    return @. -im*(1.0 - im *γ)*((espec - μ)*ϕ + chi )
end

function internalnorm(u,t)
    return sum((abs2.(u) .> 1e-6*maximum(abs2.(u))).*abs2.(u))
end

"""
    runsim(sim,ϕ;info,tplot,nfiles)

Call DifferentialEquations to solve Gross-Pitaevskii equation.
"""
function runsim(sim,ϕ=sim.ϕi;info=true,tplot=false,nfiles=false)
    @unpack nfiles,path,filename = sim

    function savefunction(ψ...)
        isdir(path) || mkpath(path)
        i = findfirst(x->x== ψ[2],sim.t)
        padi = lpad(string(i),ndigits(length(sim.t)),"0")
        info && println("⭆ Save $i at t = $(trunc(ψ[2];digits=3))")
        # tofile = path*"/"*filename*padi*".jld2"
        tofile = joinpath(path,filename*padi*".jld2")
        save(tofile,"ψ",ψ[1],"t",ψ[2])
    end

    savecb = FunctionCallingCallback(savefunction;
                     funcat = sim.t, # times to save at
                     func_everystep=false,
                     func_start = true,
                     tdir=1)

    prob = ODEProblem(Lgp!,ϕ,(sim.ti,sim.tf),sim)
    info && @info "⭆ 𝒅𝜳 Evolving in kspace"
    info && @info "⭆ Damping γ = $(sim.γ)"
    (info && nfiles) && @info "⭆ Saving to "*path
    nfiles ?
    (sol = solve(prob,alg=sim.alg,saveat=sim.t[end],reltol=sim.reltol,callback=savecb,dense=false,maxiters=1e10,progress=true)) :
    (sol = solve(prob,alg=sim.alg,saveat=sim.t,reltol=sim.reltol,dense=false,maxiters=1e10,progress=true))
    info && @info "⭆ Finished."
return sol
end

# TODO vortex lattice in 2D, persistent current in 3D examples

function testsim(sim)
    err = false
    sol = try
            runsim(sim;info=false)
        catch e
            err = true
        end
return sol,err
end

"""
    showpsi(x,y,ψ)
plot n and S in a 2D cross-section
Think x and y should be swapped in the heatmap, though.
"""
function showpsi(x,y,ψ)
    p1 = heatmap(x,y,abs2.(ψ),aspectratio=1,c=cgrad(ColorSchemes.bone_1.colors))
    xlims!(x[1],x[end]);ylims!(y[1],y[end])
    xlabel!(L"x");ylabel!(L"y")
    title!(L"|\psi|^2")
    p2 = heatmap(x,y,angle.(ψ),aspectratio=1,c=cgrad(ColorSchemes.turbo.colors))
    xlims!(x[1],x[end]);ylims!(y[1],y[end])
    xlabel!(L"x");ylabel!(L"y")
    title!(L"\textrm{phase} (\psi)")
    p = plot(p1,p2,size=(600,300))
    return p
end
