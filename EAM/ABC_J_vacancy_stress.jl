cd(@__DIR__)
ENV["CELLLISTMAP_8.3_WARNING"] = "false"
include("../src/juliaEAM.jl")
include("../src/lammpsIO.jl")

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

al_LatConst = 4.0495/10 # nm
atom_mass = 26.9815u"u"  # Atomic mass of aluminum in grams per mole

## `AtomCalculators` force/energy calculator and `Molly` simulator
eam = EAM()
fname = "Al99.eam.alloy"
read_potential!(eam, fname)

### Define customized interaction type in `AtomsCalculators`
struct EAMInteractionJulia
    calculator::Any  # Holds the ASE EAM calculator reference
    f_energy::Any    # Holds the energy function
    f_forces::Any    # Holds the forces function
    f_atomstress::Any  # Holds the atomic level stresses function
end

include("../src/simulator.jl")

eamJulia = EAMInteractionJulia(eam,calculate_energy,calculate_forces,calculate_atomstress)

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
        dist_cutoff=8u"Å")
    TNF = TreeNeighborFinder(
        eligible=trues(length(molly_atoms_init), length(molly_atoms_init)), 
        n_steps=1e3,
        dist_cutoff=8u"Å")

    # Initialize the system with the initial positions and velocities
    system_init = Molly.System(
    atoms=molly_atoms_init,
    atoms_data = [AtomData(element="Al", atom_type=string(attypes[ia])) for (ia,a) in enumerate(molly_atoms_init)],
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

filename_dump = "./LAMMPS/out_vacancy_cube8.dump"
molly_system = initialize_system_dump(filename_dump = filename_dump)

neighbors_all = get_neighbors_all(molly_system)
F_multiplier = ones(length(molly_system.coords))

fs = eamJulia.f_forces(eamJulia.calculator, molly_system, neighbors_all)
max_force_before = maximum(norm.(fs.*F_multiplier))
energy_before = eamJulia.f_energy(eamJulia.calculator, molly_system, neighbors_all)

simulator = ABCSimulator(sigma=1.0*u"Å", W=1.0*u"eV", max_steps=1, max_steps_minimize=20, step_size_minimize=5e-3u"ps", tol=1e-6u"eV/Å")
Minimize_FIRE!(molly_system, simulator, eamJulia, nothing, neighbors_all;
         n_threads=1, frozen_atoms=[], neig_interval=5, print_nsteps=true,
         mass=26.9815u"u")

frozen_atoms = []
nopenalty_atoms = []
N_free = length(molly_system.coords)-length(nopenalty_atoms)
print("exclude ",length(nopenalty_atoms)," atoms from E_phi calculation\n")
print("using ",N_free," atoms for E_phi calculation\n")

neighbors_all = get_neighbors_all(molly_system)

# sigma = sqrt(0.006*3*N_free)
sigma = sqrt(0.27)
W = 0.1
@printf("sigma^2 = %e, %e Å/dof^1/2\n W = %e eV\n",ustrip(sigma^2), ustrip(sigma/sqrt(3*N_free)),ustrip(W))

simulator = ABCSimulator(sigma=sigma*u"Å", W=W*u"eV", 
                         max_steps=40, max_steps_minimize=30, step_size_minimize=5e-3u"ps", tol=1e-3u"eV/Å")

# using ProfileView
# ProfileView.@profview 
# @code_warntype 
simulate!(molly_system, simulator, eamJulia, n_threads=1, 
        #   fname="output_stress_cube8.txt", fname_dump="stress_cube8.dump", fname_min_dump="min_stress_cube8.dump",
          fname="test.txt", fname_dump="test.dump", fname_min_dump="test.dump", # for speed test
          neig_interval=30, loggers_interval=10, dump_interval=100, start_dump=0,
          minimize_only=false, 
          d_boost=1e-6u"Å", 
          frozen_atoms=frozen_atoms, nopenalty_atoms=nopenalty_atoms, 
          p_drop = 1-1/32, p_keep=0, n_memory=0, n_search=100,
          p_stress = 1-12/2047)