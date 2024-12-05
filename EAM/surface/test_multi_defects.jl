cd(@__DIR__)
ENV["CELLLISTMAP_8.3_WARNING"] = "false"
include("../lammps_wrap/juliaEAM.jl")
include("../../src/lammpsIO.jl")

using Pkg
Pkg.activate(".")

using Printf
using AtomsCalculators
using Unitful: Å, nm
using StaticArrays: SVector
using Molly
using LinearAlgebra
using DelimitedFiles
using UnitfulAtomic
import PeriodicTable
using ProgressMeter
using Random

## Define Copper
# include("../lammps_wrap/juliaEAM.jl")

Cu_LatConst = 3.615/10 # nm
atom_mass = 63.546u"u"  # Atomic mass of aluminum in grams per mole

## `AtomCalculators` force/energy calculator and `Molly` simulator
eam = EAM()
fname = "Cu_Zhou04.eam.alloy"
read_potential!(eam, fname)

### Define customized interaction type in `AtomsCalculators`
struct EAMInteractionJulia
    calculator::Any  # Holds the ASE EAM calculator reference
    f_energy::Any    # Holds the energy function
    f_forces::Any    # Holds the forces function
    f_atomstress::Any  # Holds the atomic level stresses function
    pair_coeff_string::String  # Holds the pair coefficient string
end

include("../lammps_wrap/simulator.jl")

eamJulia = EAMInteractionJulia(eam,calculate_energy,calculate_forces_LAMMPS,calculate_atomstress,"pair_coeff * * Cu_Zhou04.eam.alloy Cu")
function Molly.forces(sys::System, interaction::EAMInteractionJulia, penalty_coords, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"), neighbors_all::Vector{Vector{Int}};
    n_threads::Integer=Threads.nthreads(), nopenalty_atoms=[]) 
    
    fs = interaction.f_forces(sys, interaction.pair_coeff_string)

    # Add penalty term to forces
    if penalty_coords != nothing
        fs += penalty_forces(sys, penalty_coords, sigma, W, nopenalty_atoms=nopenalty_atoms) # ev/Å
        # print(maximum(norm.(penalty_forces(sys, penalty_coords, sigma, W))),"\n")
    end
    return fs
end

function initialize_system_dump(;loggers=(coords=CoordinateLogger(1),),filename_dump="")
    n_atoms, box_size, coords_molly, attypes = lmpDumpReader(filename_dump)
    molly_atoms = [Molly.Atom(index=i, charge=0, mass=atom_mass, 
                    #   σ=2.0u"Å" |> x -> uconvert(u"nm", x), ϵ=ϵ_kJ_per_mol
                    ) for i in 1:length(coords_molly)]
    # Specify boundary condition
    boundary_condition = Molly.CubicBoundary(box_size[1],box_size[2],box_size[3])

    atom_positions_init = copy(coords_molly)
    molly_atoms_init = copy(molly_atoms)

    DNF = DistanceNeighborFinder(
        eligible=trues(length(molly_atoms_init), length(molly_atoms_init)),
        n_steps=1e3,
        dist_cutoff=7u"Å")
    TNF = TreeNeighborFinder(
        eligible=trues(length(molly_atoms_init), length(molly_atoms_init)), 
        n_steps=1e3,
        dist_cutoff=7u"Å")

    # Initialize the system with the initial positions and velocities
    system_init = Molly.System(
    atoms=molly_atoms_init,
    atoms_data = [AtomData(element="Cu", atom_type=string(attypes[ia])) for (ia,a) in enumerate(molly_atoms_init)],
    coords=atom_positions_init,  # Ensure these are SVector with correct units
    boundary=boundary_condition,
    # loggers=Dict(:kinetic_eng => Molly.KineticEnergyLogger(100), :pot_eng => Molly.PotentialEnergyLogger(100)),
    neighbor_finder = DNF,
    loggers=loggers,
    energy_units=u"eV",  # Ensure these units are correctly specified
    force_units=u"eV/Å"  # Ensure these units are correctly specified
    )
    return system_init
end

## Initialize sys

filename_dump = "./out_surface.dump"
molly_system = initialize_system_dump(filename_dump = filename_dump)
neighbors_all = get_neighbors_all(molly_system)

## run ABC

# frozen_atoms = []
z_coords = [coords[3] for coords in molly_system.coords]
frozen_atoms = [index for (index, z) in enumerate(z_coords) if z ≤ 1u"Å"]

nopenalty_atoms = []
N_free = length(molly_system.coords)-length(nopenalty_atoms)
print("exclude ",length(nopenalty_atoms)," atoms from E_phi calculation\n")
print("using ",N_free," atoms for E_phi calculation\n")

# sigma = sqrt(0.006*3*N_free)
sigma = sqrt(0.18)
W = 0.1
@printf("sigma^2 = %e, %e Å/dof^1/2\n W = %e eV\n",ustrip(sigma^2), ustrip(sigma/sqrt(3*N_free)),ustrip(W))

simulator = ABCSimulator(sigma=sigma*u" Å", W=W*u"eV", 
                         max_steps=10000, max_steps_minimize=30, step_size_minimize=5e-3u"ps", tol=1e-3u"eV/Å")

simulate!(molly_system, simulator, eamJulia, n_threads=1, 
         #   fname="output_stress_cube8.txt", fname_dump="stress_cube8.dump", fname_min_dump="min_stress_cube8.dump",
         fname="test3.txt", fname_dump="test3.dump", fname_min_dump="test3.dump", # for speed test
         neig_interval=30, loggers_interval=10, dump_interval=10, start_dump=0,
         minimize_only=false, 
         d_boost=1e-6u"Å", 
         frozen_atoms=frozen_atoms, nopenalty_atoms=nopenalty_atoms, 
         p_drop = 1-1/32, p_keep=0, n_memory=0, n_search=100,
         p_stress = 1-192/7180, n_stress=12)