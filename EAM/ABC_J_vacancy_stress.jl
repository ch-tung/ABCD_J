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

### Define `Molly` style ABCSimulator
# Define the ABCSimulator structure
"""
In the constructor function ABCSimulator, default values are provided for each of these fields. 
If you create a SteepestDescentMinimizer without specifying the types, default values 
will determine the types of the fields. For example, if you create a ABCSimulator without specifying sigma, 
it will default to 0.1*u"Å", and S will be the type of this value.
"""
struct ABCSimulator{S,W,D,F,L}
    sigma::S 
    W::W
    max_steps::Int
    max_steps_minimize::Int
    step_size_minimize::D
    tol::F
    log_stream::L
end

"""
ABCSimulator(; sigma=0.1*u"Å", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.1u"Å", tol=1e-4u"eV/Å", log_stream=devnull)

Constructor for ABCSimulator.

## Arguments
- `sigma`: The value of sigma in units of nm.
- `W`: The value of W in units of eV.
- `max_steps`: The maximum number of steps for the simulator.
- `max_steps_minimize`: The maximum number of steps for the minimizer.
- `step_size_minimize`: The step size for the minimizer in units of nm.
- `tol`: The tolerance for convergence in units of kg*m*s^-2.
- `log_stream`: The stream to log the output.

## Returns
- An instance of ABCSimulator.
"""
function ABCSimulator(;
                        sigma=0.1*u"Å", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.1u"Å",tol=1e-4u"eV/Å",
                        log_stream=devnull)
    return ABCSimulator(sigma, W, max_steps, max_steps_minimize, step_size_minimize, tol, log_stream)
end

# Penalty function with Gaussuan form
"""
Returns a penalty function of system coordinate x with Gaussuan form
x:      System coordinate
x_0:    Reference system coordinate
sigma:  Spatial extent of the activation, per sqrt(degree of freedom)
W:      Strenth of activation, per degree of freedom
pbc:    Periodic boundary conditions
"""
function f_phi_p(x::Vector{SVector{3, typeof(1.0u"Å")}}, x_0, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"); nopenalty_atoms=[])
    N::Int = length(x)
    E_multiplier = ones(length(x))
    for atom in nopenalty_atoms
        E_multiplier[atom] = 0
    end
    
    sigma2_new = sigma^2
    EDSQ = (A, B) -> sum(sum(map(x -> x.^2, (A-B).*E_multiplier)))
    # phi_p = sum([W * exp(-EDSQ(x,c) / (2*sigma2_new)) for c in x_0]) # unit eV
    phi_p = 0.0u"eV"
    for c in x_0
        if EDSQ(x,c)<9*sigma2_new
            phi_p_individual = W * (exp(-EDSQ(x,c) / (2*sigma2_new)) - exp(-9/2))
            phi_p += phi_p_individual
        end
    end
    return phi_p
end

function grad_f_phi_p(x::Vector{SVector{3, typeof(1.0u"Å")}}, x_0, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"); nopenalty_atoms=[])
    N::Int = length(x)
    E_multiplier = ones(length(x))
    for atom in nopenalty_atoms
        E_multiplier[atom] = 0
    end

    sigma2_new = sigma^2
    EDSQ = (A, B) -> sum(sum(map(x -> x.^2, (A-B).*E_multiplier)))

    grad_phi_p = [(@SVector zeros(Float64,3))*u"eV/Å" for i in 1:N] # unit eV/Å
    for c in x_0
        if EDSQ(x,c)<9*sigma2_new
            grad_phi_p_individual = W * exp(-EDSQ(x,c) / (2*sigma2_new)) / (2*sigma2_new) * 2*(c-x).*E_multiplier
            grad_phi_p += grad_phi_p_individual
        end
    end
    return grad_phi_p
end

# Calculate the gradient of the penalty energy
function penalty_forces(sys::System, penalty_coords, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"); nopenalty_atoms=[])
    # Function of the penalty energy for a given coordinate
    # f_phi_p_coords = x -> f_phi_p(x, penalty_coords, sigma, W)

    # Calculate the gradient of the penalty energy, The penalty force is the negative gradient of the penalty energy
    # penalty_fs = -gradient(f_phi_p_coords, sys.coords)[1] # unit eV/Å
    penalty_fs = -grad_f_phi_p(sys.coords, penalty_coords, sigma, W, nopenalty_atoms=nopenalty_atoms) # unit eV/Å

    return penalty_fs
end

# Define the forces function with penalty term
"""
Evaluate the forces acting on the system with penalty term
If there is no penalty term, the penalty_coords should be set to nothing, 
and return the forces identical to the original forces function
"""
function Molly.forces(sys::System, interaction::EAMInteractionJulia, penalty_coords, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"), neighbors_all::Vector{Vector{Int}};
    n_threads::Integer=Threads.nthreads(), nopenalty_atoms=[]) 

    
    fs = interaction.f_forces(interaction.calculator, sys, neighbors_all)

    # Add penalty term to forces
    if penalty_coords != nothing
        fs += penalty_forces(sys, penalty_coords, sigma, W, nopenalty_atoms=nopenalty_atoms) # ev/Å
        # print(maximum(norm.(penalty_forces(sys, penalty_coords, sigma, W))),"\n")
    end
    return fs
end

"""
f_energy_phi(sys::System, sim::Simulator, penalty_coords)

Compute the total energy of the system `sys` including the potential energy contribution from the penalty coordinates.

# Arguments
- `sys::System`: The system for which the energy is to be computed.
- `sim::Simulator`: The simulator object containing simulation parameters.
- `penalty_coords`: The penalty coordinates used to calculate the potential energy contribution.

# Returns
- `E`: The total energy of the system with penalty terms.

"""
function f_energy_phi(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, penalty_coords, neighbors_all; nopenalty_atoms=[])
    E_phi = 0*u"eV"
    if penalty_coords!=nothing
        E_phi += f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W, nopenalty_atoms=nopenalty_atoms)
    end
    E = interaction.f_energy(interaction.calculator, sys, neighbors_all) + E_phi
    return E
end

"""
Minimize_FIRE!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, penalty_coords, neighbors_all::Vector{Vector{Int}};
                n_threads::Integer=1, frozen_atoms=[], neig_interval::Int=1000, print_nsteps=false,
                mass::typeof(1.0u"u")=26.9815u"u",acoef_0::Float64=0.1,alpha::Float64=0.99)

Minimizes the system using the Fast Inertial Relaxation Engine (FIRE) algorithm.

# Arguments
- `sys::System`: The system to be minimized.
- `sim::ABCSimulator`: The ABC simulator.
- `interaction::EAMInteractionJulia`: The EAM interaction.
- `penalty_coords`: The penalty coordinates.
- `neighbors_all::Vector{Vector{Int}}`: The list of neighbor indices for each atom.
- `n_threads::Integer`: The number of threads to use for parallelization. Default is 1.
- `frozen_atoms`: The indices of atoms that are frozen during the minimization. Default is an empty array.
- `neig_interval::Int`: The interval at which neighbor lists are updated. Default is 1000.
- `print_nsteps`: Whether to print the number of steps during the minimization. Default is false.
- `mass::typeof(1.0u"u")`: The mass of the atoms. Default is 26.9815u"u".
- `acoef_0::Float64`: The initial value of the acceleration coefficient. Default is 0.1.
- `alpha::Float64`: The parameter controlling the step size. Default is 0.99.
"""
function Minimize_FIRE!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, 
                        penalty_coords, neighbors_all::Vector{Vector{Int}};
                        step_size_minimize = sim.step_size_minimize, max_steps_minimize = sim.max_steps_minimize,
                        n_threads::Integer=1, frozen_atoms=[], neig_interval::Int=1000, print_nsteps=false,
                        mass::typeof(1.0u"u")=26.9815u"u",acoef_0::Float64=0.1,alpha::Float64=0.99)
    dt_0 = step_size_minimize # time unit (ps)
    dt = dt_0
    acoef = acoef_0
    # Set F_multiplier of frozen_atoms to zero
    F_multiplier = ones(length(sys.coords))
    for atom in frozen_atoms
        F_multiplier[atom] = 0
    end
  
    E = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all)

    v_0 = [(@SVector zeros(3))*u"eV/Å/u*ps" for i in 1:length(sys.coords)] # force/mass*time
    v = v_0
    for step_n in 1:max_steps_minimize
        if step_n % neig_interval == 0
            neighbors_all = get_neighbors_all(sys)
        end

        # 1. skier force
        F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all)
        F = F.*F_multiplier
        
        P = sum(sum([v[i].*F[i] for i in 1:length(v)]))   
        vv = sum(sum([v[i].*v[i] for i in 1:length(v)]))
        FF = sum(sum([F[i].*F[i] for i in 1:length(v)]))
        if ustrip(P)>0.0
            # dt *= 1.1
            dt = min(dt*1.2, 1e3*dt_0)
            v = (1-acoef)*v + acoef*sqrt(vv/FF)*F
            acoef = acoef * alpha
            
        else
            # dt *= 0.5
            dt = max(dt*0.5, 1e-1*dt_0)
            v = v_0
            acoef = acoef_0
        end

        # 2. MD
        accl = F/mass # force/mass
        v += accl.*dt # force/mass*time

        # 3. update coordinate
        coords_update = v .* dt
        sys.coords .+= coords_update # force/mass*time*2

        E_trial = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all)
        # print(E,"\n")
        # if E_trial>E
        #     # print("zeroing velocity")
        #     v*=0
        # else
        #     E = E_trial
        # end

    end
    # neighbors_all = get_neighbors_all(sys)
    # F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all)
    # F = F.*F_multiplier
    # max_force = maximum(norm.(F))
    # @printf("max force = %e eV/Å ",ustrip(max_force))
    return sys
end


"""
Minimize_MD!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, penalty_coords, neighbors_all::Vector{Vector{Int}};
                        n_threads::Integer=1, frozen_atoms=[], neig_interval::Int=1000, print_nsteps=false,
                        mass::typeof(1.0u"u")=26.9815u"u", acoef_0::Float64=0.1,alpha::Float64=0.99,constrained=false)

Minimizes the molecular dynamics (MD) system using the ABC algorithm.

# Arguments
- `sys::System`: The molecular dynamics system.
- `sim::ABCSimulator`: The ABC simulator.
- `interaction::EAMInteractionJulia`: The EAM interaction.
- `penalty_coords`: The penalty coordinates.
- `neighbors_all::Vector{Vector{Int}}`: The neighbor list.

# Optional Arguments
- `n_threads::Integer=1`: The number of threads to use.
- `frozen_atoms=[]`: The list of frozen atoms.
- `neig_interval::Int=1000`: The neighbor interval.
- `print_nsteps=false`: Whether to print the number of steps.
- `mass::typeof(1.0u"u")=26.9815u"u"`: The mass of the atoms.
- `acoef_0::Float64=0.1`: The initial value of the a coefficient.
- `alpha::Float64=0.99`: The alpha coefficient.
- `constrained=false`: Whether the system is constrained.
- `gamma`: The damping strength

# Returns
- `sys`: The updated molecular dynamics system.

"""
function Minimize_MD!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, 
                      penalty_coords, neighbors_all::Vector{Vector{Int}};
                      step_size_minimize = sim.step_size_minimize, max_steps_minimize = sim.max_steps_minimize,
                      n_threads::Integer=1, frozen_atoms=[], nopenalty_atoms=[], neig_interval::Int=1000, print_nsteps=false,
                      mass::typeof(1.0u"u")=26.9815u"u",constrained=false, etol = 1e-4, gamma=0.0)
    N = length(sys.coords)
    dt_0 = step_size_minimize # time unit (ps)
    dt = dt_0
    # acoef = acoef_0
    # Set F_multiplier of frozen_atoms to zero
    F_multiplier = ones(length(sys.coords))
    for atom in frozen_atoms
        F_multiplier[atom] = 0
    end
    
    # energy before move
    E_0 = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all, nopenalty_atoms=nopenalty_atoms)
    E = E_0
    deltaE_current = E_0*0
    deltaE_1 = E_0*0
    # print(E,"\n")

    v_0 = [(@SVector zeros(3))*u"eV/Å/u*ps" for i in 1:length(sys.coords)] # force/mass*time
    v = v_0
    for step_n in 1:sim.max_steps_minimize
        if step_n % neig_interval == 0
            neighbors_all = get_neighbors_all(sys)
        end
        # 1. calculate Force
        F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all, nopenalty_atoms=nopenalty_atoms)
        F = F.*F_multiplier

        # 2. MD
        accl = F/mass # force/mass
        v += accl.*dt - gamma*v # force/mass*time
        coords_update = v .* dt # force/mass*time^2

        # zeroing CM
        coords_update_cm = sum(coords_update)/N
        coords_update = [c - coords_update_cm for c in coords_update]

        sys.coords .+= coords_update # force/mass*time*2

        # energy after move
        E_trial = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all, nopenalty_atoms=nopenalty_atoms)
        if step_n == 1
        # terminate condition
        else 
            deltaE_current = abs(E_trial-E)
            deltaE_1 = abs(E_trial-E_0)
            if deltaE_current<etol*deltaE_1
                break
            end
        end

        # print(E,"\n")
        if E_trial>E
            # print("zeroing velocity")
            v = v_0
            dt = max(dt*0.9, 3e-1*dt_0)
        else
            E = E_trial
            dt = min(dt*1.1, 3*dt_0)
        end

    end
    # neighbors_all = get_neighbors_all(sys)
    # F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all, nopenalty_atoms=nopenalty_atoms)
    # F = F.*F_multiplier
    # max_force = maximum(norm.(F))
    return sys
end


# Implement the simulate! function for ABCSimulator
"""
    simulate!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia; 
               n_threads::Integer=Threads.nthreads(), run_loggers::Bool=true, fname::String="output_MD.txt", fname_dump="out.dump", fname_min_dump="out_min.dump",
               neig_interval::Int=1, loggers_interval=1, dump_interval = 1, start_dump = 1, n_memory = 50,
               minimize_only::Bool=false, 
               d_boost=1.0e-2u"Å", beta=0.0, E_th = sim.W*0.1,
               frozen_atoms=[], nopenalty_atoms=[],
               p_drop=0.0, p_keep=0.5, drop_interval::Int=1)

Simulates the system using the ABC algorithm with molecular dynamics (MD) steps for structural minimization.

# Arguments
- `sys::System`: The system to be simulated.
- `sim::ABCSimulator`: The ABC simulator object.
- `interaction::EAMInteractionJulia`: The EAM interaction object.

# Optional Arguments
- `n_threads::Integer`: The number of threads to use for parallelization. Default is the number of available threads.
- `run_loggers::Bool`: Whether to run the loggers during the simulation. Default is `true`.
- `fname::String`: The name of the output file for energy calculation. Default is "output_MD.txt".
- `fname_dump::String`: The name of the output file for dump data. Default is "out.dump".
- `fname_min_dump::String`: The name of the output file for minimized dump data. Default is "out_min.dump".
- `neig_interval::Int`: The interval at which to update the neighbor list. Default is 1.
- `loggers_interval::Int`: The interval at which to run the loggers. Default is 1.
- `dump_interval::Int`: The interval at which to dump data. Default is 1.
- `start_dump::Int`: The step number at which to start dumping data. Default is 1.
- `memory::Int`: The number of previous coordinates to store for penalty calculation. Default is 50.
- `minimize_only::Bool`: Whether to only perform minimization without simulation. Default is `false`.
- `d_boost::Float`: The boost factor for perturbing the system coordinates. Default is 1.0e-2Å.
- `beta::Float`: The beta value for the ABC algorithm. Default is 0.0.
- `E_th::Float`: The threshold energy for penalty calculation. Default is sim.W * exp(-3).
- `frozen_atoms::Array{Int}`: The indices of atoms to be frozen during minimization. Default is an empty array.
- `nopenalty_atoms::Array{Int}`: The indices of atoms to be excluded from penalty calculation. Default is an empty array.
- `p_drop::Float`: The probability of dropping atoms from penalty calculation. Default is 0.0.
- `p_keep::Float`: The probability of keeping atoms from the previous drop list. Default is 0.5.
- `drop_interval::Int`: The interval at which to perform atom dropping. Default is 1.

# Returns
- `sys::System`: The simulated system.
"""
function simulate!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia; 
                   n_threads::Integer=Threads.nthreads(), run_loggers::Bool=true, fname::String="output_MD.txt", fname_dump="out.dump", fname_min_dump="out_min.dump",
                   neig_interval::Int=1, loggers_interval=1, dump_interval = 1, start_dump = 1,
                   minimize_only::Bool=false, 
                   d_boost=1.0e-2u"Å", beta=0.0, E_th = sim.W*exp(-3),
                   frozen_atoms=[], nopenalty_atoms=[],
                   p_drop=0.0, p_keep=0.5, drop_interval::Int=1, n_memory = 50, n_search=60,
                   p_stress = 1e-2)
    N = length(sys.coords)
    neighbors_all = get_neighbors_all(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)

    # open an empty output file for energy calculation
    open(fname, "w") do file
        write(file, "")
    end

    # Set d_multiplier of frozen_atoms to zero
    d_multiplier = ones(N)
    for atom in frozen_atoms
        d_multiplier[atom] = 0
    end

    # 0. Call Minimize! without penalty_coords before the loop
    # Minimize_momentum!(sys, sim, interaction, nothing, neighbors_all; n_threads=n_threads, frozen_atoms=frozen_atoms, neig_interval=neig_interval, beta=beta, print_nsteps=true)
    Minimize_MD!(sys, sim, interaction, nothing, neighbors_all; n_threads=n_threads, frozen_atoms=frozen_atoms, neig_interval=neig_interval, mass=26.9815u"u", nopenalty_atoms=nopenalty_atoms)
    E = interaction.f_energy(interaction.calculator, sys, neighbors_all)

    if minimize_only
        return sys
    end

    # Run the loggers, Log the step number (or other details as needed)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    @printf("step %d: ",0)
    print(E)
    print("\n")

    # Calculate atomic stress
    stresses = interaction.f_atomstress(eam, sys, neighbors_all)
    vm_stress = f_vm_stress(stresses)

    # dump first data file
    fname_min_path = replace(fname_min_dump, ".dump" => "/")
    if isdir(fname_min_path)
        rm(fname_min_path,force=true, recursive=true)
    end
    mkdir(fname_min_path)
    fname_min_data = fname_min_path*"data."*string(0,pad=3)
    open(fname_min_data, "w") do file_min
        lmpDataWriter(file_min,0,sys,fname_min_data)
    end

    # dump first dump file
    open(fname_min_dump, "w") do file_min
        lmpDumpWriter_prop(file_min,0,sys,fname_min_dump, vm_stress)
    end

    ## 1. Store the initial coordinates
    penalty_coords = [copy(sys.coords)]  

    # Get the indices of the particles sorted by von Mises stress in descending order
    sorted_indices = sortperm(vm_stress, rev=false)
    # Select the particles with the top von Mises stress
    n_top = round(Int, p_stress*N)
    nopenalty_atoms_stress = sorted_indices[1:n_top]

    # F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all, nopenalty_atoms=nopenalty_atoms)
    # F = F.*d_multiplier
    # max_force_0 = maximum(norm.(F))
    p = Progress(sim.max_steps)
    step_counter = 0
    dr = sys.coords*0
    # open(fname_min_dump, "a") do file_min_dump
        # open(fname_min_data, "a") do file_min_data
    for step_n in 1:sim.max_steps
        step_counter += 1
        next!(p)
        ## 2. Slightly perturb the system coordinates
        for i in 1:N
            random_direction = randn(size(sys.coords[i]))
            sys.coords[i] += d_boost * random_direction*d_multiplier[i]
        end

        ## 3. Call Minimize! with penalty_coords, update system coordinates
        ## energy before minimization
        coords_before = copy(sys.coords)
        sys_prev = deepcopy(sys)

        ## run the minimization algorithm
        Minimize_MD!(sys, sim, interaction, penalty_coords, neighbors_all; n_threads=n_threads, frozen_atoms=frozen_atoms, neig_interval=neig_interval, mass=26.9815u"u", nopenalty_atoms=nopenalty_atoms_stress)
        E = interaction.f_energy(interaction.calculator, sys, neighbors_all)
        dr = sys_prev.coords-sys.coords

        ## energy after minimization
        # neighbors_all = get_neighbors_all(sys)
        E_phi = f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W, nopenalty_atoms=nopenalty_atoms_stress)

        ## 4. Output and dump
        if step_n % loggers_interval==0
            # Run the loggers, Log the step number (or other details as needed)
            run_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads)
        end

        ## 5. Check stress convergence
        if step_n >= start_dump 
            if E_phi<E_th && step_counter>1
                # do one more minimization step
                # Minimize_MD!(sys, sim, interaction, [], neighbors_all; n_threads=n_threads, frozen_atoms=frozen_atoms, neig_interval=1, mass=26.9815u"u", nopenalty_atoms=nopenalty_atoms_stress)
                Minimize_FIRE!(sys, sim, interaction, [], neighbors_all;
                max_steps_minimize=20, step_size_minimize=5e-3u"ps",
                n_threads=n_threads, frozen_atoms=frozen_atoms, neig_interval=10, mass=26.9815u"u")

                # update neighbor list
                neighbors_all = get_neighbors_all(sys)

                ## energy after minimization
                E = interaction.f_energy(interaction.calculator, sys, neighbors_all)
                E_phi = f_phi_p(sys.coords, penalty_coords, sim.sigma, sim.W, nopenalty_atoms=nopenalty_atoms_stress)
                E_after = E+E_phi

                ## write energy and penalty energy to text file
                open(fname, "a") do file_E
                    write(file_E, string(ustrip(E))*" "*string(ustrip(E_phi))*"\n")
                end

                # clear penalty coordinates
                # penalty_coords = length(penalty_coords) < n_memory ? penalty_coords : penalty_coords[end-n_memory+1:end]
                penalty_coords = []

                # reset step counter
                step_counter = 0

                # Calculate atomic stress 
                stresses = interaction.f_atomstress(eam, sys, neighbors_all)
                vm_stress = f_vm_stress(stresses)
                stresses_prev = interaction.f_atomstress(eam, sys_prev, neighbors_all)
                vm_stress_prev = f_vm_stress(stresses_prev)
                open(fname_min_dump, "a") do file_min_dump
                    lmpDumpWriter_prop(file_min_dump,step_n-1,sys_prev,fname_min_dump,vm_stress_prev)
                    lmpDumpWriter_prop(file_min_dump,step_n,sys,fname_min_dump,vm_stress)
                end

                fname_min_data = fname_min_path*"data."*string(step_n,pad=3)
                fname_min_data_prev = fname_min_path*"data."*string(step_n-1,pad=3)
                
                open(fname_min_data, "a") do file_min_data
                    lmpDataWriter(file_min_data,step_n,sys,fname_min_data)  
                end
                open(fname_min_data_prev, "a") do file_min_data
                    lmpDataWriter(file_min_data,step_n-1,sys_prev,fname_min_data_prev)
                end

                # lmpDumpWriter(file,step_n,sys,fname_min_dump)

                # update penalty list
                # Get the indices of the particles sorted by von Mises stress in descending order
                sorted_indices = sortperm(vm_stress, rev=false)
                # Select the particles with the top von Mises stress
                n_top = round(Int, p_stress*N)
                nopenalty_atoms_stress = sorted_indices[1:n_top]

                continue
            end
            # if step_n % dump_interval == 0
            #     # lmpDumpWriter(file,step_n,sys,fname_dump)
            #     # print("step ",step_n,"\n")
            # end
        end
        
        ## write energy and penalty energy to text file
        open(fname, "a") do file_E
            write(file_E, string(ustrip(E))*" "*string(ustrip(E_phi))*"\n")
        end

        ## 6. Update penalty_coords for the next step
        push!(penalty_coords, copy(sys.coords))
        if step_counter>=n_search
            # clear penalty coordinates if taking too much step without identifying the minimum
            penalty_coords=[]
            # reset the step counter
            step_counter = 0
        end
        # if step_n>n_memory
        #     popfirst!(penalty_coords)
        # end

    end
        # end
    # end
    return sys
end

### Initialize System and run simulator
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
    # Initialize the system with the initial positions and velocities
    system_init = Molly.System(
    atoms=molly_atoms_init,
    atoms_data = [AtomData(element="Al", atom_type=string(attypes[ia])) for (ia,a) in enumerate(molly_atoms_init)],
    coords=atom_positions_init,  # Ensure these are SVector with correct units
    boundary=boundary_condition,
    # loggers=Dict(:kinetic_eng => Molly.KineticEnergyLogger(100), :pot_eng => Molly.PotentialEnergyLogger(100)),
    neighbor_finder = DistanceNeighborFinder(
    eligible=trues(length(molly_atoms_init), length(molly_atoms_init)),
    n_steps=1e3,
    dist_cutoff=8u"Å"),
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

# x_coords = [coords[1] for coords in molly_system.coords]
# y_coords = [coords[2] for coords in molly_system.coords]
# z_coords = [coords[3] for coords in molly_system.coords]
# # frozen_atoms = [index for (index, (x, y, z)) in enumerate(zip(x_coords, y_coords, z_coords)) if -7.9u"Å" ≤ x ≤ 110.2u"Å" && -7.9u"Å" ≤ y ≤ 110.2u"Å" && 16.1u"Å" ≤ z ≤ 16.3u"Å"]
frozen_atoms = []
# print("\n\nfreeze ", length(frozen_atoms), " atoms\n")
# F_multiplier = ones(length(molly_system.coords))
# for atom in frozen_atoms
#     F_multiplier[atom] = 0
# end

# coord_vacancy =@SVector [20.249u"Å", 20.249u"Å", 36.4482u"Å"]
# r_focus = [sqrt(sum((coord-coord_vacancy).^2)) for coord in molly_system.coords]
# # nopenalty_atoms = [index for (index, r_focus_i) in enumerate(r_focus) if r_focus_i > 0.55*sqrt(2)*al_LatConst*10u"Å"]
nopenalty_atoms = []
N_free = length(molly_system.coords)-length(nopenalty_atoms)
print("exclude ",length(nopenalty_atoms)," atoms from E_phi calculation\n")
print("using ",N_free," atoms for E_phi calculation\n")

# Minimize_FIRE!(molly_system, simulator, eamJulia, nothing, neighbors_all;
#          n_threads=1, frozen_atoms=frozen_atoms, neig_interval=5, print_nsteps=true,
#          mass=26.9815u"u")

neighbors_all = get_neighbors_all(molly_system)
fs = eamJulia.f_forces(eamJulia.calculator, molly_system, neighbors_all)
max_force_after = maximum(norm.(fs.*F_multiplier))
energy_after = eamJulia.f_energy(eamJulia.calculator, molly_system, neighbors_all)
@printf("\n\nmax force before minimization = %e eV/Å\n", ustrip(max_force_before))
@printf("max force after minimization = %e eV/Å\n", ustrip(max_force_after))
@printf("\nmax energy before minimization = %f eV/Å\n", ustrip(energy_before))
@printf("max energy after minimization = %f eV/Å\n", ustrip(energy_after))

N_free = length(molly_system.coords)-length(nopenalty_atoms)
# sigma = sqrt(0.006*3*N_free)
sigma = sqrt(0.27)
W = 0.1
@printf("sigma^2 = %e, %e Å/dof^1/2\n W = %e eV\n",ustrip(sigma^2), ustrip(sigma/sqrt(3*N_free)),ustrip(W))

# simulator = ABCSimulator(sigma=sigma*u"Å", W=W*u"eV", 
#                          max_steps=1000, max_steps_minimize=500, step_size_minimize=5e-2u"Å", tol=1e-5u"eV/Å")
simulator = ABCSimulator(sigma=sigma*u"Å", W=W*u"eV", 
                         max_steps=1000, max_steps_minimize=128, step_size_minimize=2e-3u"ps", tol=1e-3u"eV/Å")
# using ProfileView
# ProfileView.@profview 
simulate!(molly_system, simulator, eamJulia, n_threads=1, 
          fname="output_stress_cube8.txt", fname_dump="stress_cube8.dump", fname_min_dump="min_stress_cube8.dump",
          neig_interval=64, loggers_interval=10, dump_interval=100, start_dump=0,
          minimize_only=false, 
          d_boost=1e-6u"Å", 
          frozen_atoms=frozen_atoms, nopenalty_atoms=nopenalty_atoms, 
          p_drop = 1-1/32, p_keep=0.16, n_memory=0, n_search=100,
          p_stress = 1-12/2047)