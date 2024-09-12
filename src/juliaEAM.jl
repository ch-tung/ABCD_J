ENV["CELLLISTMAP_8.3_WARNING"] = "false"

using Pkg
Pkg.activate(".")

using Printf
using AtomsCalculators
# using ASEconvert # use this PR:https://github.com/mfherbst/ASEconvert.jl/pull/17, Pkg.add(url="https://github.com/tjjarvinen/ASEconvert.jl.git", rev="atomscalculators")
using Unitful: Å, nm
# using PythonCall
# ENV["PYTHON"] = "/SNS/users/ccu/miniconda3/envs/analysis/bin/python"
# install the following packages in julia REPL
# using CondaPkg
# CondaPkg.add_pip("IPython")
# CondaPkg.add_pip("nglview")
using StaticArrays: SVector

# using GLMakie
using Molly
using Zygote
using LinearAlgebra
import Interpolations:cubic_spline_interpolation, linear_interpolation, interpolate, BSpline, Cubic, scale, Line, OnGrid, extrapolate, Gridded, extrapolate, Flat
using DelimitedFiles

# 1. Define interaction
# Helper function to get the type of a Interpolation object
constructor_interp = cubic_spline_interpolation

function get_spline_type(constructor_interp=cubic_spline_interpolation)
    dummy_spline = constructor_interp(0.0:1.0:2.0, [0.0, 1.0, 2.0])
    # dummy_spline = extrapolate(dummy_spline, Flat())
    return typeof(dummy_spline)
end

# Helper function to get the type of the derivative of a CubicSplineInterpolation object
function get_deriv_type(constructor_interp=cubic_spline_interpolation)
    dummy_spline = constructor_interp(0.0:1.0:2.0, [0.0, 1.0, 2.0])
    dummy_deriv = deriv(dummy_spline)
    return typeof(dummy_deriv)
end

function deriv(spline::get_spline_type(constructor_interp))
    d_spline(x) = gradient(spline, x)[1]
    return d_spline
end

mutable struct EAM
    Nelements::Int
    elements::Vector{String}
    nrho::Int
    drho::Float64
    nr::Int
    dr::Float64
    cutoff::Float64
    embedded_data::Matrix{Float64}
    density_data::Matrix{Float64}
    Z::Vector{Int}
    mass::Vector{Float64}
    a::Vector{Float64}
    lattice::String
    rphi_data::Array{Float64, 3}
    r_range::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    rho_range::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    r::Vector{Float64}
    rho::Vector{Float64}

    embedded_energy::Vector{get_spline_type(constructor_interp)}
    electron_density::Vector{get_spline_type(constructor_interp)}
    d_embedded_energy::Vector{get_spline_type(constructor_interp)}
    d_electron_density::Vector{get_spline_type(constructor_interp)}
    phi::Matrix{get_spline_type(constructor_interp)}
    d_phi::Matrix{get_spline_type(constructor_interp)}

    EAM() = new()
end

function set_splines(calculator::EAM)
    calculator.embedded_energy = Vector{get_spline_type(constructor_interp)}(undef, calculator.Nelements)
    calculator.electron_density = Vector{get_spline_type(constructor_interp)}(undef, calculator.Nelements)
    calculator.d_embedded_energy = Vector{get_spline_type(constructor_interp)}(undef, calculator.Nelements)
    calculator.d_electron_density = Vector{get_spline_type(constructor_interp)}(undef, calculator.Nelements)

    for i in 1:calculator.Nelements
        ci_embedded_energy = constructor_interp(calculator.rho_range, calculator.embedded_data[i, :])
        ci_electron_density = constructor_interp(calculator.r_range, calculator.density_data[i, :])
        # ci_embedded_energy = extrapolate(ci_embedded_energy, Flat())
        # ci_electron_density = extrapolate(ci_electron_density, Flat())
        calculator.embedded_energy[i] =  ci_embedded_energy # arrays of embedded energy functions, [N_types]
        calculator.electron_density[i] = ci_electron_density # arrays of electron density functions, [N_types]
        f_embedded_energy = deriv(calculator.embedded_energy[i])
        f_electron_density = deriv(calculator.electron_density[i])
        d_embedded_energy_knot = f_embedded_energy.(calculator.rho)
        d_electron_density_knot = f_electron_density.(calculator.r)
        ci_d_embedded_energy = constructor_interp(calculator.rho_range,d_embedded_energy_knot)
        ci_d_electron_density = constructor_interp(calculator.r_range,d_electron_density_knot)
        # ci_d_embedded_energy = extrapolate(ci_d_embedded_energy, Flat())
        # ci_d_electron_density = extrapolate(ci_d_electron_density, Flat())
        calculator.d_embedded_energy[i] = ci_d_embedded_energy # arrays of derivative of embedded energy functions, [N_types]
        calculator.d_electron_density[i] = ci_d_electron_density # arrays of derivative of electron density functions, [N_types]
        # calculator.d_embedded_energy[i] = deriv(calculator.embedded_energy[i]) # arrays of derivative of embedded energy functions, [N_types]
        # calculator.d_electron_density[i] = deriv(calculator.electron_density[i]) # arrays of derivative of electron density functions, [N_types]
    end

    calculator.phi = Matrix{get_spline_type(constructor_interp)}(undef, calculator.Nelements, calculator.Nelements) # arrays of pairwise energy functions, [N_types, N_types]
    calculator.d_phi = Matrix{get_spline_type(constructor_interp)}(undef, calculator.Nelements, calculator.Nelements) # arrays, [N_types, N_types]

    # ignore the first point of the phi data because it is forced
    # to go through zero due to the r*phi format in alloy and adp
    for i in 1:calculator.Nelements
        for j in i:calculator.Nelements
            ci_phi = constructor_interp(calculator.r_range[2:end], calculator.rphi_data[i, j, :][2:end] ./ calculator.r[2:end]) 
            # ci_phi = extrapolate(ci_phi, Flat())
            calculator.phi[i, j] = ci_phi
            f_d_phi = deriv(calculator.phi[i, j])
            d_phi_knot = f_d_phi.(calculator.r[2:end])
            ci_d_phi = constructor_interp(calculator.r_range[2:end],d_phi_knot)
            # ci_d_phi = extrapolate(ci_d_phi, Flat())
            calculator.d_phi[i, j] = ci_d_phi
            # calculator.d_phi[i, j] = deriv(calculator.phi[i, j])

            if j != i
                calculator.phi[j, i] = calculator.phi[i, j]
                calculator.d_phi[j, i] = calculator.d_phi[i, j]
            end
        end
    end
end


"""
    read_potential!(calculator::EAM, fd::String)

Reads the potential data from a file and populates the fields of the `calculator` object.

# Arguments
- `calculator::EAM`: The EAM calculator object to populate with potential data.
- `fd::String`: The file path to the potential data file.

# Description
This function reads the potential data from the specified file and assigns the values to the corresponding fields of the `calculator` object. The file should be in a specific format, with each line containing the relevant data for a specific field.

The function starts reading the file from the 4th line and converts the lines into a list of strings. It then extracts the number of elements, element symbols, and other parameters from the list. Next, it reads the embedded energy and electron density data for each element, as well as the r*phi data for each interaction between elements. Finally, it sets up the ranges and arrays for the potential data and calls the `set_splines` function to calculate the splines.

Note: This function assumes that the potential data file is formatted correctly and contains the required information in the expected order.

"""
function read_potential!(calculator::EAM, fd::String)
    lines = readdlm(fd, '\n', String) # read the files, split by new line

    function lines_to_list(lines) # convert the entries in lines to list
        data = []
        for line in lines
            append!(data, split(line))
        end
        return data
    end

    i = 4 # start from the 4th line
    data = lines_to_list(lines[i:end])

    calculator.Nelements = parse(Int, data[1]) # number of elements
    d = 2
    calculator.elements = data[d:d+calculator.Nelements-1] # the elements symbols starts from the 2nd entries
    d += calculator.Nelements

    calculator.nrho = parse(Int, data[d]) 
    calculator.drho = parse(Float64, data[d+1])
    calculator.nr = parse(Int, data[d+2])
    calculator.dr = parse(Float64, data[d+3])# the cutoff radius in angstroms
    calculator.cutoff = parse(Float64, data[d+4]) 

    calculator.embedded_data = zeros(calculator.Nelements, calculator.nrho)
    calculator.density_data = zeros(calculator.Nelements, calculator.nr)
    calculator.Z = zeros(Int, calculator.Nelements)
    calculator.mass = zeros(calculator.Nelements)
    calculator.a = zeros(calculator.Nelements)
    calculator.lattice = ""
    d += 5

    # reads in the part of the eam file for each element
    for elem in 1:calculator.Nelements
        calculator.Z[elem] = parse(Int, data[d]) # the atomic number
        calculator.mass[elem] = parse(Float64, data[d+1]) # the atomic mass
        calculator.a[elem] = parse(Float64, data[d+2]) # the lattice constant
        calculator.lattice *= data[d+3] # the lattice type
        d += 4

        calculator.embedded_data[elem, :] = parse.(Float64, data[d:(d+calculator.nrho-1)]) # the embedded energy of the element
        d += calculator.nrho
        calculator.density_data[elem, :] = parse.(Float64, data[d:(d+calculator.nr-1)]) # the electron density of the element
        d += calculator.nr
    end

    # reads in the r*phi data for each interaction between elements
    calculator.rphi_data = zeros(calculator.Nelements, calculator.Nelements, calculator.nr)

    for i in 1:calculator.Nelements
        for j in 1:i
            calculator.rphi_data[j, i, :] = parse.(Float64, data[d:(d+calculator.nr-1)])
            d += calculator.nr
        end
    end

    calculator.r_range = (0:calculator.nr-1)*calculator.dr
    calculator.rho_range = (0:calculator.nrho-1)*calculator.drho
    calculator.r = collect(calculator.r_range)
    calculator.rho = collect(calculator.rho_range)

    set_splines(calculator)
end

"""
    vector(c1, c2, boundary_side_lengths)

Displacement between two coordinate values from c1 to c2, accounting for
periodic boundary conditions.

The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
For the [`TriclinicBoundary`](@ref) an approximation is used to find the closest
version by default.

vector_1D is from Molly.jl
"""
function Molly.vector(c1::SVector{3, Float64}, c2::SVector{3, Float64}, boundary_side_lengths::SVector{3, Float64})
    return @inbounds SVector(
        vector_1D(c1[1], c2[1], boundary_side_lengths[1]),
        vector_1D(c1[2], c2[2], boundary_side_lengths[2]),
        vector_1D(c1[3], c2[3], boundary_side_lengths[3]),
    )
end

function get_neighbors(neig, i)
    neighbors::Vector{Int} = []
    for j in 1:length(neig)
        neig_j = neig[j]
        if neig_j[1] == i
            append!(neighbors, neig_j[2])
        end
        if neig_j[2] == i
            append!(neighbors, neig_j[1])
        end
    end
    return unique(neighbors)
end

function get_neighbors_all(sys::Molly.System)
    neighbors_all = [Int[] for _ in 1:length(sys.atoms)]
    n_threads = 1
    neig = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    for i in 1:length(neig)
        pair_i = neig[i]
        append!(neighbors_all[pair_i[1]], pair_i[2])
        append!(neighbors_all[pair_i[2]], pair_i[1])
    end
    return neighbors_all
end

function convert_neighbors(neig, n_atoms)
    neighbors_all = [Int[] for _ in 1:n_atoms]
    for i in 1:length(neig)
        pair_i = neig[i]
        append!(neighbors_all[pair_i[1]], pair_i[2])
        append!(neighbors_all[pair_i[2]], pair_i[1])
    end
    return neighbors_all
end

function get_type(index_list::Vector{Int64}, typelist::Vector{Int64})
    list_type_index = Vector{Int}(undef, length(index_list))
    for i in 1:length(index_list)
        list_type_index[i] = indexin(1, typelist)[1]
    end
    return list_type_index
end

"""
calculate_energy(eam::EAM, sys::Molly.System, neighbors_all)

Calculate the total energy of a system using the Embedded Atom Method (EAM).

# Arguments
- `eam::EAM`: The EAM calculator.
- `sys::Molly.System`: The system object containing atom coordinates and types.
- `neighbors_all`: A precomputed list of neighbors for each atom.

# Returns
- `energy::Float64`: The total energy of the system in electron volts (eV).
"""

function calculate_energy(eam::EAM, sys::Molly.System, neighbors_all::Vector{Vector{Int}})
    coords = [ustrip(coord) for coord in sys.coords]
    boundary = @SVector [(ustrip(sys.boundary[i])) for i in 1:3]
    return calculate_energy(eam, coords, boundary, neighbors_all)
end

function calculate_energy(eam::EAM, coords::Vector{SVector{3, Float64}}, boundary::SVector{3, Float64}, neighbors_all::Vector{Vector{Int}})    
    n_threads = 1

    pair_energy::Float64 = 0.0
    embedding_energy::Float64 = 0.0
    total_density::Vector{Float64} = zeros(length(coords))
    
    # for single element system, only one type is considered
    typelist = [1]
    i_type::Int = 1 in typelist ? indexin(1, typelist)[1] : error("1 not found in typelist")
    eam_phi_ix = eam.phi[i_type, :]

    for i::Int in 1:length(coords)
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]
        coords_i = coords[i]
        
        if isempty(neighbors)
            continue
        end

        n_neighbors::Int = length(neighbors)
        d_i::Vector{Float64} = zeros(n_neighbors)
        for (index_j, j::Int) in enumerate(neighbors)
            d_i[index_j] = minimum((sqrt(sum(vector(coords_i, coords[j], boundary).^2)),eam.r[end])) # unit already in Å
        end

        eam_embedded_energy_i = eam.embedded_energy[i_type]
        for j_type::Int in 1:eam.Nelements
            # use = get_type(neighbors, typelist) .== j_type
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            d_use = d_i

            pair_energy += sum(eam_phi_ix[j_type].(d_use))
            total_density[i] += sum(eam.electron_density[j_type].(d_use))
        end
        embedding_energy += eam_embedded_energy_i.(total_density[i])
    end

    components = Dict("pair" => pair_energy/2, "embedding" => embedding_energy)
    energy::Float64 = sum(values(components))
    return energy*u"eV"
end

"""
    calculate_forces(eam::EAM, sys::Molly.System, neighbors_all)

Calculate the forces on particles in a molecular system using the Embedded Atom Method (EAM).

# Arguments
- `eam::EAM`: An instance of the EAM potential.
- `sys::Molly.System`: The molecular system.
- `neighbors_all`: A matrix containing the neighbors of each particle in the system.

# Returns
- `forces_particle`: A matrix containing the forces on each particle in the system.
"""
function calculate_forces(eam::EAM, sys::Molly.System, neighbors_all::Vector{Vector{Int}})
    coords = [ustrip(coord) for coord in sys.coords]
    boundary = @SVector [ustrip(sys.boundary[i]) for i in 1:3]
    return calculate_forces(eam, coords, boundary, neighbors_all)
end

function calculate_forces(eam::EAM, coords::Vector{SVector{3, Float64}}, boundary::SVector{3, Float64}, neighbors_all::Vector{Vector{Int}})
    n_threads::Int = 1
    
    ## for single element system, only one type is considered
    typelist::Vector{Int} = [1]
    i_type::Int = 1 in typelist ? indexin(1, typelist)[1] : error("1 not found in typelist")
    d_electron_density_i = eam.d_electron_density[i_type]
    
    # preallocate
    # initialize forces_particle
    forces_particle::Matrix{Float64} = zeros(length(coords),3)
    
    # initialize total_density
    total_density::Vector{Float64} = zeros(length(coords))

    r_all::Vector{Matrix{Float64}} = []
    d_all::Vector{Vector{Float64}} = []

    n_neighbors_all::Vector{Int} = [length(neighbors_all[i]) for i in 1:length(coords)]
    n_neighbors_max::Int = maximum(n_neighbors_all)

    r_i::Matrix{Float64} = zeros(n_neighbors_max,3)
    d_i::Vector{Float64} = zeros(n_neighbors_max)
    for i::Int in 1:length(coords)
        # i_type::Int = indexin(1, typelist)[1]
        
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]    
        if isempty(neighbors)
            continue
        end

        n_neighbors::Int = length(neighbors)
        coords_i = coords[i]
    
        # distance between atom i and its neighbors
        # r_i::Matrix{Float64} = zeros(n_neighbors,3)
        # d_i::Vector{Float64} = zeros(n_neighbors)
        for (index_j::Int, j::Int) in enumerate(neighbors)
            r_ij = (vector(coords_i, coords[j], boundary)) # Å
            d_ij = sqrt(sum(r_ij.^2))
            r_i[index_j,1:3] = r_ij
            d_i[index_j] = minimum((d_ij,eam.r[end]))
        end

        push!(r_all, r_i[1:n_neighbors,1:3])
        push!(d_all, d_i[1:n_neighbors])
    
        for j_type::Int in 1:eam.Nelements # iterate over all types
            # use = get_type(neighbors, typelist) .== j_type # get the index of the neighbors with type j
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            d_use = d_all[i]
    
            density = sum(eam.electron_density[j_type].(d_use)) # electron density
            total_density[i] += density # total electron density around atom i
        end
    end
    
    # calculate forces on particles
    for i::Int in 1:length(coords)
        # i_type::Int = indexin(1, typelist)[1]
            
        # neighbors = get_neighbors(neig, i)
        neighbors::Vector{Int} = neighbors_all[i]
        if isempty(neighbors)
            continue
        end
        n_neighbors::Int = length(neighbors)
        coords_i = coords[i]
    
        r_i = r_all[i]
        d_i = d_all[i]
        
        # derivative of the embedded energy of atom i
        d_embedded_energy_i::Float64 = eam.d_embedded_energy[i_type].(total_density[i])
    
        ur_i = r_i
    
        # unit directional vector
        ur_i ./= d_i
        
        for j_type::Int in 1:eam.Nelements
            # use = get_type(neighbors, typelist) .== j_type # get the index of the neighbors with type j
            # if !any(use)
            #     continue
            # end
            # d_use = d_i[use]
            # ur_use = ur_i[use, :]
            # neighbors_use = neighbors[use]
            d_use = d_i
            ur_use::Matrix{Float64} = ur_i[:,:]
            neighbors_use = neighbors
    
            total_density_j = total_density[neighbors_use]
            
            scale::Vector{Float64} = eam.d_phi[i_type, j_type].(d_use)
            scale .+= (d_embedded_energy_i .* eam.d_electron_density[j_type].(d_use)) 
            scale .+= (eam.d_embedded_energy[j_type].(total_density_j) .* d_electron_density_i.(d_use))
    
            forces_particle[i, :] .+= (ur_use' * scale)
        end
    end

    return [SVector{3,Float64}(forces_particle[idx_f,:]) for idx_f in 1:size(forces_particle)[1]]*u"eV/Å"
end