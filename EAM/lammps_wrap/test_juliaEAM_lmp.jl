cd(@__DIR__)
ENV["CELLLISTMAP_8.3_WARNING"] = "false"
include("../../src/juliaEAM.jl")
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

al_LatConst = 4.0495/10 # nm
atom_mass = 26.9815u"u"  # Atomic mass of aluminum in grams per mole

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

filename_dump = "./out_divacancy.dump"
molly_system = initialize_system_dump(filename_dump = filename_dump)
neighbors_all = get_neighbors_all(molly_system)

using LAMMPS

# Create a LAMMPS instance
lmp = LMP()

# Define the atom style
command(lmp, "units metal")
command(lmp, "dimension 3")
command(lmp, "boundary p p p")
command(lmp, "atom_style atomic")

# Define the simulation box according to molly_system.boundary
boundary = molly_system.boundary
box_x = ustrip(boundary[1])
box_y = ustrip(boundary[2])
box_z = ustrip(boundary[3])
command(lmp, "region box block 0 $(box_x) 0 $(box_y) 0 $(box_z)")
command(lmp, "create_box 1 box")

# Get atom positions, indices and types
ustripped_coords = map(ustrip, molly_system.coords)
pos = convert(Matrix,(reshape(reinterpret(Float64, ustripped_coords),3,:)))
types = map(atoms_data -> parse(Int32, atoms_data.atom_type), molly_system.atoms_data)
indices = map(atoms -> Int32(atoms.index), molly_system.atoms)

# Create atoms in LAMMPS
LAMMPS.create_atoms(lmp, pos, indices, types)

# Set atom type and mass
command(lmp, "mass 1 26.9815")

# Define interatomic potential
command(lmp, "pair_style eam/alloy")
command(lmp, "pair_coeff * * Al99.eam.alloy Al")

# Compute force 
command(lmp, "compute f all property/atom fx fy fz")

f = gather(lmp, "c_f", Float64)

print(f)