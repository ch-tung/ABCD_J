"""
Minimize_momentum!(sys, sim, penalty_coords; 
                   n_threads::Integer, frozen_atoms=[], neig_inteval::Int=1000, beta::Float64=0.0, print_nsteps=false)

Minimizes the system `sys` energy using the simulatior `sim` providing penalty coordinates `penalty_coords`.

# Arguments
- `sys`: The system to be minimized.
- `sim`: The simulation object.
- `penalty_coords`: The penalty coordinates used in the minimization.
- `n_threads`: The number of threads to use in the minimization.
- `frozen_atoms`: (optional) A list of atoms to be frozen during the minimization.
- `n_threads`: The number of threads to use in the minimization.
- `frozen_atoms`: (optional) A list of atoms to be frozen during the minimization.
- `neig_inteval`: (optional) Inteval between neighbor list updates.
- `beta`: (optional) Momentum coefficient. if beta=0 reduced to gradient descent.
- `print_nsteps`: (optional) Print number of steps required to converge.
"""
function Minimize_momentum!(sys::System, sim::ABCSimulator, interaction::EAMInteractionJulia, penalty_coords, neighbors_all::Vector{Vector{Int}}; 
                            n_threads::Integer=1, frozen_atoms=[], neig_inteval::Int=1000, beta::Float64=0.0, print_nsteps::Bool=false)
    hn = sim.step_size_minimize

    # 1. initialize
    # initialize system energy with penalty
    E = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all)
    
    # Set F_multiplier of frozen_atoms to zero
    F_multiplier = ones(length(sys.coords))
    for atom in frozen_atoms
        F_multiplier[atom] = 0
    end

    F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all)
    F = F.*F_multiplier
    max_force_0 = maximum(norm.(F))
    F_copy = copy(F)
    n_accept = 0
    n_reject = 0
    string_convergence="\n"
    for step_n in 1:sim.max_steps_minimize
    # step_n = 0
    # while n_accept < sim.max_steps_minimize
        # step_n+=1
        # Calculate the forces using the new forces function
        # penalty_coords is fixed throughout the minimization

        # 2. evaluate force 
        if step_n % neig_inteval == 0
            neighbors_all = get_neighbors_all(sys)
        end
        F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all)
        F = F.*F_multiplier
        max_force = maximum(norm.(F))

        if step_n > 1
            # 3. check convergence          
            if  max_force < sim.tol
                if print_nsteps
                    print(" Minimization converged after ", step_n, " steps\n")
                end
                break
            end

            # 4. modify force with Momentum
            F = beta*F_copy + (1-beta)*F
            max_force = maximum(norm.(F))
        end

        # 5. update coordinate
        coords_copy = copy(sys.coords)
        coords_update = hn * F ./ max_force # ensure that the maximum move at the each step is sim.step_size_minimize
        sys.coords += coords_update

        # 6. copy neighbor list
        neighbors_all_copy = copy(neighbors_all)
        
        # 7. if energy didn't reduce, revert coordinate back to the previous step.
        # System energy after applying displacements in this step
        if step_n % neig_inteval == 0
            neighbors_all = get_neighbors_all(sys)
        end
        # neighbors_all = get_neighbors_all(sys)
        E_trial = f_energy_phi(sys, sim, interaction, penalty_coords, neighbors_all)
        if E_trial < E
            hn = hn * 6/5
            # hn = min(hn, 1e-1u"Å")
            E = E_trial
            F_copy = copy(F) # copy force for momentum term in the next step
            n_accept += 1
            string_convergence*="1"
        else
            # revert to previous coordinate
            sys.coords = coords_copy
            neighbors_all = neighbors_all_copy
            hn = hn / 5
            # hn = max(hn, 1e-6u"Å")
            # F_copy = copy(F)
            n_reject += 1
            string_convergence*="0"
        end
    end
    
    neighbors_all = get_neighbors_all(sys)
    F = forces(sys, interaction, penalty_coords, sim.sigma, sim.W, neighbors_all)
    F = F.*F_multiplier
    max_force = maximum(norm.(F))
    @printf("max force = %e eV/Å ,n_accept: %d, n_reject: %d ",ustrip(max_force), n_accept, n_reject)
    # print(string_convergence,"\n")
    return sys
end

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
            continue
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
    return sys
end