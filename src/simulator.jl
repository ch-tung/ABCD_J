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
    # E_multiplier = ones(length(x))
    # for atom in nopenalty_atoms
    #     E_multiplier[atom] = 0
    # end

    penalty_atoms = setdiff(1:N, nopenalty_atoms)
    
    sigma2_new = sigma^2
    EDSQ = (A, B) -> sum(sum(map(x -> x.^2, (A-B))))
    # phi_p = sum([W * exp(-EDSQ(x,c) / (2*sigma2_new)) for c in x_0]) # unit eV
    phi_p = 0.0u"eV"
    # Threads.@threads 
    tasks = []
    for c in x_0
        task = Threads.@spawn begin
            xp = x[penalty_atoms]
            cp = c[penalty_atoms]
            if EDSQ(xp,cp)<9*sigma2_new
                phi_p_individual = W * (exp(-EDSQ(xp,cp) / (2*sigma2_new)) - exp(-9/2))
                phi_p += phi_p_individual
            end
        end
        push!(tasks, task)
    end
    for task in tasks
        wait(task)
    end
    return phi_p
end

function grad_f_phi_p(x::Vector{SVector{3, typeof(1.0u"Å")}}, x_0, sigma::typeof(1.0u"Å"), W::typeof(1.0u"eV"); nopenalty_atoms=[])
    N::Int = length(x)
    # E_multiplier = ones(length(x))
    # for atom in nopenalty_atoms
    #     E_multiplier[atom] = 0
    # end

    penalty_atoms = setdiff(1:N, nopenalty_atoms)

    sigma2_new = sigma^2
    EDSQ = (A, B) -> sum(sum(map(x -> x.^2, (A-B))))

    grad_phi_p = [(@SVector zeros(Float64,3))*u"eV/Å" for i in 1:N] # unit eV/Å
    # Threads.@threads 
    tasks = []
    for c in x_0
        task = Threads.@spawn begin
            xp = x[penalty_atoms]
            cp = c[penalty_atoms]
            if EDSQ(xp,cp)<9*sigma2_new
                grad_phi_p_individual = W * exp(-EDSQ(xp,cp) / (2*sigma2_new)) / (2*sigma2_new) * 2*(cp-xp)
                grad_phi_p[penalty_atoms] += grad_phi_p_individual
            end
        end
        push!(tasks, task)
    end
    for task in tasks
        wait(task)
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

include("minimize.jl")

# Implement the simulate! function for ABCSimulator
"""
    simulate!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia; 
               n_threads::Integer=Threads.nthreads(), run_loggers::Bool=true, fname::String="output_MD.txt", fname_dump="out.dump", fname_min_dump="out_min.dump",
               neig_interval::Int=1, loggers_interval::Int=1, dump_interval::Int=1, start_dump::Int=1,
               minimize_only::Bool=false, 
               d_boost=1.0e-2u"Å", beta=0.0, E_th = sim.W*exp(-3),
               frozen_atoms=[], nopenalty_atoms=[],
               p_drop::Float64=0.0, p_keep=0.5, drop_interval::Int=1, n_memory::Int=50, n_search::Int=60,
               p_stress::Float64=1e-2)

Simulates the system using the ABC algorithm with an EAM interaction potential.

# Arguments
- `sys::System`: The system to be simulated.
- `sim::ABCSimulator`: The ABC simulator object.
- `interaction::EAMInteractionJulia`: The EAM interaction potential object.

# Optional Arguments
- `n_threads::Integer`: The number of threads to use for parallelization. Default is the number of available threads.
- `run_loggers::Bool`: Whether to run the loggers during the simulation. Default is `true`.
- `fname::String`: The name of the output file for energy calculation. Default is "output_MD.txt".
- `fname_dump::String`: The name of the dump file for system configurations. Default is "out.dump".
- `fname_min_dump::String`: The name of the dump file for minimized system configurations. Default is "out_min.dump".
- `neig_interval::Int`: The interval for updating the neighbor list. Default is 1.
- `loggers_interval::Int`: The interval for running the loggers. Default is 1.
- `dump_interval::Int`: The interval for dumping system configurations. Default is 1.
- `start_dump::Int`: The step number to start dumping system configurations. Default is 1.
- `minimize_only::Bool`: Whether to only perform minimization without simulation. Default is `false`.
- `d_boost::Float64`: The boost factor for perturbing the system coordinates. Default is 1.0e-2 Å.
- `beta::Float64`: The beta value for the ABC algorithm. Default is 0.0.
- `E_th::Float64`: The threshold energy for convergence. Default is `sim.W*exp(-3)`.
- `frozen_atoms::Array{Int}`: The indices of the frozen atoms. Default is an empty array.
- `nopenalty_atoms::Array{Int}`: The indices of the atoms with no penalty. Default is an empty array.
- `p_drop::Float64`: The probability of dropping an atom during the ABC algorithm. Default is 0.0.
- `p_keep::Float64`: The probability of keeping an atom during the ABC algorithm. Default is 0.5.
- `drop_interval::Int`: The interval for dropping atoms during the ABC algorithm. Default is 1.
- `n_memory::Int`: The number of previous penalty coordinates to store. Default is 50.
- `n_search::Int`: The number of steps to search for the minimum before clearing the penalty coordinates. Default is 60.
- `p_stress::Float64`: The percentage of atoms with the top von Mises stress to be penalized. Default is 1e-2.
"""

function simulate!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia; 
                   n_threads::Integer=Threads.nthreads(), run_loggers::Bool=true, fname::String="output_MD.txt", fname_dump="out.dump", fname_min_dump="out_min.dump",
                   neig_interval::Int=1, loggers_interval::Int=1, dump_interval::Int=1000, start_dump::Int=1,final=false,
                   minimize_only::Bool=false, 
                   d_boost=1.0e-2u"Å", beta=0.0, E_th = sim.W*exp(-3),
                   frozen_atoms=[], nopenalty_atoms=[],
                   p_drop::Float64=0.0, p_keep=0.5, drop_interval::Int=1, n_memory::Int=50, n_search::Int=60,
                   p_stress::Float64=1e-2)
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
    open(fname_min_dump, "w") do file_min_dump
        write(file_min_dump, "")
    end
    output = lmpDumpWriter_prop(0,sys,vm_stress)
    # open(fname_min_dump, "w") do file_min
    #     lmpDumpWriter_prop(file_min,0,sys,fname_min_dump, vm_stress)
    # end

    ## 1. Store the initial coordinates
    penalty_coords = [copy(sys.coords)]  

    # Get the indices of the particles sorted by von Mises stress in descending order
    sorted_indices = sortperm(vm_stress, rev=false)
    # Select the particles with the top von Mises stress
    n_top = round(Int, p_stress*N)
    nopenalty_atoms_stress = sorted_indices[1:n_top]

    p = Progress(sim.max_steps, 10)
    step_counter = 0
    dr = sys.coords*0

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
        # dr = sys_prev.coords-sys.coords

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
                output *= lmpDumpWriter_prop(step_n-1,sys_prev,vm_stress_prev)
                output *= lmpDumpWriter_prop(step_n,sys,vm_stress)
                # open(fname_min_dump, "a") do file_min_dump
                #     lmpDumpWriter_prop(file_min_dump,step_n-1,sys_prev,fname_min_dump,vm_stress_prev)
                #     lmpDumpWriter_prop(file_min_dump,step_n,sys,fname_min_dump,vm_stress)
                # end
                if final
                    fname_min_data = fname_min_path*"data."*string(step_n,pad=3)
                    fname_min_data_prev = fname_min_path*"data."*string(step_n-1,pad=3)
                    open(fname_min_data, "a") do file_min_data
                        lmpDataWriter(file_min_data,step_n,sys,fname_min_data)  
                    end
                    open(fname_min_data_prev, "a") do file_min_data
                        lmpDataWriter(file_min_data,step_n-1,sys_prev,fname_min_data_prev)
                    end
                    fname_min_final = fname_min_path*"final."*string(step_n,pad=3)
                    open(fname_min_final, "a") do fname_min_final
                        lmpFinalWriter(fname_min_final,step_n,sys,fname_min_final)
                    end
                end

                # update penalty list
                # Get the indices of the particles sorted by von Mises stress in descending order
                sorted_indices = sortperm(vm_stress, rev=false)
                # Select the particles with the top von Mises stress
                n_top = round(Int, p_stress*N)
                nopenalty_atoms_stress = sorted_indices[1:n_top]

                continue
            end

            # dump the system configuration and clear output string
            if step_n % dump_interval == 0
                open(fname_min_dump, "a") do file_min_dump
                    write(file_min_dump, output)
                end
                output = ""
            end
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

    # dump the rest of the system configuration
    open(fname_min_dump, "a") do file_min_dump
        write(file_min_dump, output)
    end
    return sys
end
