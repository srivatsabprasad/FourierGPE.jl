using LaTeXStrings, Pkg, Revise

using FourierGPE

# ==== Units: ========================
# Units of ξ for length, 1/μ for time
# related by μ ≡ ħ²/mξ² = mc^2 where
# c = ħ/mξ is the speed of sound of
# the uniform system.

# ==== define user parameters =======
@with_kw mutable struct Params <: UserParams @deftype Float64
    # user parameters:
    κ = 0.1
end
par = Params()

# ==== set simulation parameters ====
L=(15.,15.,15.)
N=(64,64,64)
γ = 0.05
tf = 1.5/γ
Nt = 200
t = LinRange(0.,tf,Nt)
# ========= Initialize simulation ======
sim = Sim(L,N,par)
@pack! sim = γ,tf,Nt,t
@unpack_Sim sim

# uniform potential
import FourierGPE.V
V(x,y,z,t) = 0.5*(x^2+y^2+4*z^2)

# ========= useful state functions
ψ0(x,y,z,μ,g) = sqrt(μ/g)*sqrt(max(1.0-V(x,y,z,0.0)/μ,0.0)+im*0.0)
healing(x,y,z,μ,g) = 1/sqrt(g*abs2(ψ0(x,y,z,μ,g)))

x,y,z = X
#make initial state
ψi = ψ0.(x,y',reshape(z,1,1,length(z)),μ,g)
ψi = randn(N)+im*randn(N)
ϕi = kspace(ψi,sim)
@pack! sim = ϕi,γ,tf,t

sim

# ====== Evolve in k space ==========
sol = runsim(sim)
# ===================================


# ====== animate slice using Plots ===
using Plots
gr(titlefontsize=12,size=(500,300),colorbar=false)

function showpsi(x,y,ψ)
    p1 = heatmap(x,y,abs2.(ψ),aspectratio=1)
    xlabel!(L"x/\xi");ylabel!(L"y/\xi")
    title!(L"|\psi|^2")
    p2 = heatmap(x,y,angle.(ψ),aspectratio=1)
    xlabel!(L"x/\xi");ylabel!(L"y/\xi")
    title!(L"\textrm{phase}(\psi)")
    p = plot(p1,p2,size=(600,300))
    return p
end

# pull out the ground state
ϕg = sol[end]
ψg = xspace(ϕg,sim)
showpsi(x,y,ψg[:,:,1])

# animate a slice
# dipole decay
anim = @animate for i=1:Nt
    showpsi(x,y,xspace(sol[i],sim)[:,:,1])
end

gif(anim,"./examples/3dquench.gif",fps=30)


# ========== animate isosurface in Makie ========
using Makie, AbstractPlotting

function dense(phi)
    ψm = xspace(phi,sim)
    density = abs2.(ψm)
    pmax = maximum(density)
    return density/pmax
end

function densityfilm(sol,Nt)
    saveto="examples/3dtrap.gif"
    scene = Scene()
    tindex = Node(1)
    scene = volume(lift(i -> dense(sol[i]), tindex),
    algorithm = :iso,
    show_axis=false,
    isovalue=3f0(.15))

    R = 70
    eyepos = Vec3f0(R,R,R)
    lookat = Vec3f0(18,18,0)

    record(scene, saveto, 1:Nt) do i
        update_cam!(scene, eyepos, lookat)
        rotate_cam!(scene, 0., -0.25, 0.)
        tindex[] = i
    end
    p = scene[end];
    return
end

p = densityfilm(sol,Nt)

# p.attributes.attribute
