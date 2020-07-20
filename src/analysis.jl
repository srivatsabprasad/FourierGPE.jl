"""
	gradient(psi::XField{D})

Compute the `D` vector gradient components of an `XField` of spatial dimension `D`.
The `D` gradient components returned are `D`-dimensional arrays.
"""
function gradient(psi::XField{1})
	@unpack psiX,K = psi; kx = K[1]; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	return ψx
end

function gradient(psi::XField{2})
	@unpack psiX,K = psi; kx,ky = K; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	return ψx,ψy
end

function gradient(psi::XField{3})
	@unpack psiX,K = psi; kx,ky,kz = K; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
	return ψx,ψy,ψz
end

"""
	current(psi::XField{D})

Compute the `D` current components of an `XField` of spatial dimension `D`.
The `D` cartesian components returned are `D`-dimensional arrays.
"""
function current(psi::XField{1})
	@unpack psiX,K = psi; kx = K[1]; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	jx = @. imag(conj(ψ)*ψx)
	return jx
end

function current(psi::XField{2})
	@unpack psiX,K = psi; kx,ky = K; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	return jx,jy
end

function current(psi::XField{3})
	@unpack psiX,K = psi; kx,ky,kz = K; ψ = psiX
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
	jx = @. imag(conj(ψ)*ψx)
	jy = @. imag(conj(ψ)*ψy)
	jz = @. imag(conj(ψ)*ψz)
	return jx,jy,jz
end

"""
	velocity(psi::XField{D})

Compute the `D` velocity components of an `XField` of spatial dimension `D`.
The `D` velocities returned are `D`-dimensional arrays.
"""
function velocity(psi::XField{1})
	@unpack psiX,K = psi; kx = K[1]; ψ = psiX
	rho = abs2.(ψ)
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	vx = @. imag(conj(ψ)*ψx)/rho
	return vx
end

function velocity(psi::XField{2})
	@unpack psiX,K = psi; kx,ky = K; ψ = psiX
	rho = abs2.(ψ)
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	return vx,vy
end

function velocity(psi::XField{3})
	@unpack psiX,K = psi; kx,ky,kz = K; ψ = psiX
	rho = abs2.(ψ)
	ϕ = fft(ψ)
	ψx = ifft(im*kx.*ϕ)
	ψy = ifft(im*ky'.*ϕ)
	ψz = ifft(im*reshape(kz,1,1,length(kz)).*ϕ)
	vx = @. imag(conj(ψ)*ψx)/rho
	vy = @. imag(conj(ψ)*ψy)/rho
	vz = @. imag(conj(ψ)*ψz)/rho
	return vx,vy,vz
end

"""
	Wi,Wc = helmholtz(wx,wy,...,psi::XField{D})

Computes a 2 or 3 dimensional Helmholtz decomposition of the vector field with components
`wx`, `wy`, or `wx`, `wy`, `wz`. `psi` is passed to provide requisite arrays in `k`-space.
Returned fields `Wi`, `Wc` are incompressible and compressible respectively.
"""
function helmholtz(wx, wy, psi::XField{2})
    @unpack K, K2 = psi; kx, ky = K
    wxk = fft(wx); wyk = fft(wy)
    kdotw = @. kx * wxk + ky' * wyk
    wxkc = @. kdotw * kx / K2; wxkc[1] = zero(wxkc[1])
    wykc = @. kdotw * ky' / K2; wykc[1] = zero(wykc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wxc = ifft(wxkc); wyc = ifft(wykc)
  	wxi = ifft(wxki); wyi = ifft(wyki)
  	Wi = (wxi, wyi); Wc = (wxc, wyc)
    return Wi, Wc
end

function helmholtz(wx, wy, wz, psi::XField{3})
    @unpack K, K2 = psi; kx, ky, kz = K
    wxk = fft(wx); wyk = fft(wy); wzk = fft(wz)
    kzt = reshape(kz, 1, 1, length(kz))
    kdotw = @. kx * wxk + ky' * wyk + kzt * wzk
    wxkc = @. kdotw * kx / K2; wxkc[1] = zero(wxkc[1])
    wykc = @. kdotw * ky' / K2; wykc[1] = zero(wykc[1])
    wzkc = @. kdotw * kzt / K2; wzkc[1] = zero(wzkc[1])
    wxki = @. wxk - wxkc
    wyki = @. wyk - wykc
    wzki = @. wzk - wzkc
    wxc = ifft(wxkc); wyc = ifft(wykc); wzc = ifft(wzkc)
    wxi = ifft(wxki); wyi = ifft(wyki); wzi = ifft(wzki)
  		Wi = (wxi, wyi, wzi); Wc = (wxc, wyc, wzc)
    return Wi, Wc
end

function helmholtz(W::NTuple{N,Array{Float64,N}}, psi::XField{N}) where N
    return helmholtz(W..., psi)
end

"""
	et,ei,ec = energydecomp(psi::Xfield{D})

Decomposes the hydrodynamic kinetic energy of `psi`, returning the total `et`, incompressible `ei`,
and compressible `ec` energy densities in position space. `D` can be 2 or 3 dimensions.
"""
function energydecomp(psi::XField{2})
    @unpack psiX = psi
    rho = abs2.(psiX)
    vx, vy = velocity(psi)
    wx = @. sqrt(rho)*vx; wy = @. sqrt(rho)*vy
    Wi, Wc = helmholtz(wx,wy,psi)
    wxi, wyi = Wi; wxc, wyc = Wc
    et = @. abs2(wx) + abs2(wy); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc); ec *= 0.5
    return et, ei, ec
end

function energydecomp(psi::XField{3})
	@unpack psiX = psi
    rho = abs2.(psiX)
    vx, vy, vz = velocity(psi)
    wx = @. sqrt(rho) * vx; wy = @. sqrt(rho) * vy; wz = @. sqrt(rho) * vz
    Wi, Wc = helmholtz(wx,wy,wz,psi)
    wxi, wyi, wzi = Wi; wxc, wyc, wzc = Wc
    et = @. abs2(wx) + abs2(wy) + abs2(wz); et *= 0.5
    ei = @. abs2(wxi) + abs2(wyi) + abs2(wzi); ei *= 0.5
    ec = @. abs2(wxc) + abs2(wyc) + abs2(wzc); ec *= 0.5
    return et, ei, ec
end

"""
	zeropad(A)

Zero-pad the 2D array `A` to twice the size with the same element type as `A`.
"""
function zeropad(a)
    s = size(a)
    (isodd.(s) |> any) && error("Array dims must be divisible by 2")
    S = @. 2 * s
    t = @. s / 2 |> Int
    z = Zeros{eltype(a)}(t...)
    M = Hcat([z; z], a, [z; z])
    return Vcat([z z z z], M, [z z z z]) |> Matrix
end

"""
	log10range(a,b,n)

Create a vector that is linearly spaced in log space, containing `n` values bracketed by `a` and `b`.
"""
function log10range(a,b,n)
	@assert a>0
    x = LinRange(log10(a),log10(b),n)
    return @. 10^x
end

@doc raw"""
	A = convolve(ψ1,ψ2,X,K)

Computes the convolution of two complex fields according to

```math
A(\rho) = \int d^2r\;\psi_1(r-\rho)\psi_2(r)
```
using FFTW.
"""
function convolve(ψ1,ψ2,X,K;periodic=false)
    DX,DK = dfftall(X,K)
    if periodic == false
	ϕ1 = zeropad(ψ1)
    ϕ2 = zeropad(ψ2)
    else
        ϕ1 = ψ1
        ϕ2 = ψ2
    end
	χ1 = fft(ϕ1)*prod(DX)
	χ2 = fft(ϕ2)*prod(DX)
	return ifft(χ1.*χ2)*prod(DK) |> fftshift
end

@doc raw"""
	autocorrelate(ψ,X,K,periodic=false)

Return the autocorrelation integral of a complex field ``\psi``, ``A``, given by

```
A(\rho)=\int d^2r\;\psi^*(r-\rho)\psi(r)
```

defined on a cartesian grid on a cartesian grid using FFTW.

`X` and `K` are tuples of vectors `x`,`y`,`kx`, `ky`.

This method is useful for evaluating spectra from cartesian data.
"""
function autocorrelate(ψ,X,K;periodic=false)
    DX,DK = dfftall(X,K)
    if periodic == false
        ϕ = zeropad(ψ)
    else
        ϕ = ψ
    end
	χ = fft(ϕ)*prod(DX)
	return ifft(abs2.(χ))*prod(DK) |> fftshift
end

"""
	kespectrum(k,ψ,X,K)

Calculates the kinetic enery spectrum for wavefunction ``\\psi``, at the
points `k`. Arrays `X`, `K` should be computed using `makearrays`.
"""
function kespectrum(k,ψ,X,K)
    x,y = X; kx,ky = K
    dx,dy = diff(x)[1],diff(y)[1]
    DX,DK = dfftall(X,K)
    k2field = k2(K)
    psi = XField(ψ,X,K,k2field)
    ψx,ψy = gradient(psi)

	cx = autocorrelate(ψx,X,K)
	cy = autocorrelate(ψy,X,K)
    Ci = cx .+ cy

    # make ρ
    Nx = 2*length(x)
    Lx = x[end] - x[begin] + dx
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yp = xp
    ρ = hypot.(xp,yp')

    # do spectra
    Ek = zero(k)
    for (j,kj) in enumerate(k)
        Ek[j]  = 0.5*kj*sum(@. besselj0(kj*ρ)*Ci)*dx*dy |> real
    end
    return Ek
end

"""
	ikespectrum(k,ψ,X,K)

Caculate the incompressible kinetic enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function ikespectrum(k,ψ,X,K)
    x,y = X; kx,ky = K
 	dx,dy = diff(x)[1],diff(y)[1]
	DX,DK = dfftall(X,K)
    k2field = k2(K)
    psi = XField(ψ,X,K,k2field)
    vx,vy = velocity(psi)
    rho = abs2.(ψ)
    wx = @. sqrt(rho)*vx; wy = @. sqrt(rho)*vy
    Wi, Wc = helmholtz(wx,wy,psi)
    wix,wiy = Wi

	cix = autocorrelate(wix,X,K)
	ciy = autocorrelate(wiy,X,K)
    Ci = cix .+ ciy

	# make ρ
    Nx = 2*length(x)
    Lx = x[end]-x[1] + dx
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yp = xp
    ρ = hypot.(xp,yp')

    Eki = zero(k)
    for (j,kj) in enumerate(k)
        Eki[j]  = 0.5*kj*sum(@. besselj0(kj*ρ)*Ci)*dx*dy |> real
    end
    return Eki
end

"""
	ckespectrum(k,ψ,X,K)

Caculate the compressible kinetic enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function ckespectrum(k,ψ,X,K)
    x,y = X; kx,ky = K
 	dx,dy = diff(x)[1],diff(y)[1]
	DX,DK = dfftall(X,K)
    k2field = k2(K)
    psi = XField(ψ,X,K,k2field)
    vx,vy = velocity(psi)
    rho = abs2.(ψ)
    wx = @. sqrt(rho)*vx; wy = @. sqrt(rho)*vy
    Wi, Wc = helmholtz(wx,wy,psi)
    wcx,wcy = Wc

	ccx = autocorrelate(wcx,X,K)
	ccy = autocorrelate(wcy,X,K)
    Cc = ccx .+ ccy

    Nx = 2*length(x)
    Lx = x[end]-x[1] + dx
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yp = xp
    ρ = hypot.(xp,yp')

    Ekc = zero(k)
    for (j,kj) in enumerate(k)
        Ekc[j]  = 0.5*kj*sum(@. besselj0(kj*ρ)*Cc)*dx*dy |> real
    end
    return Ekc
end

"""
	qpspectrum(k,ψ,X,K)

Caculate the quantum pressure enery spectrum for wavefunction ``\\psi``, via Helmholtz decomposition.
Input arrays `X`, `K` must be computed using `makearrays`.
"""
function qpespectrum(k,ψ,X,K)
    x,y = X; kx,ky = K
 	dx,dy = diff(x)[1],diff(y)[1]
	DX,DK = dfftall(X,K)
    k2field = k2(K)
    psi = XField(abs.(ψ) |> complex,X,K,k2field)
    rnx,rny = gradient(psi)
    rnx ./= abs.(ψ) # test needed
    rny ./= abs.(ψ)
	cx = autocorrelate(rnx,X,K)
	cy = autocorrelate(rny,X,K)
    Cq = cx .+ cy

	# make ρ
    Nx = 2*length(x)
    Lx = x[end]-x[1] + dx
    xp = LinRange(-Lx,Lx,Nx+1)[1:Nx]
    yp = xp
    ρ = hypot.(xp,yp')

    Eq = zero(k)
    for (j,kj) in enumerate(k)
        Eq[j]  = 0.5*kj*sum(@. besselj0(kj*ρ)*Cq)*dx*dy |> real
    end
    return Eq
end
