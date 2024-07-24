# [ABCD\_J: Autonomous Basin Climbing Dynamics in Julia](https://github.com/ch-tung/ABCD_J.git)

This project incorporates Julia into a new metadynamics molecular simulation program. The simulation framework is developed using the Julia packages [Molly.jl](https://github.com/JuliaMolSim/Molly.jl), [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl), and [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl).

## `src/`: Julia sourcecodes

### `juliaEAM.jl`: Pure julia based EAM calculator.

#### Read the potential data from a file and populate the fields of the `calculator` object

```julia
read_potential!(calculator::EAM, fd::String)
```

**Arguments:**

*   `calculator`: The EAM calculator object to populate with potential data.
*   `fd`: The file path to the potential data file.

#### Calculate the total energy of a system using the Embedded Atom Method (EAM)

```julia
calculate_energy(eam::EAM, sys::Molly.System, neighbors_all)
```

**Arguments:**

*   `eam`: The EAM potential parameters.
*   `sys`: The system object containing atom coordinates and types.
*   `neighbors_all`: A precomputed list of neighbors for each atom.

**Returns:**

*   `energy`: The total energy of the system in electron volts (eV).

#### Calculate the forces on particles using the Embedded Atom Method (EAM)

```julia
calculate_forces(eam::EAM, sys::Molly.System, neighbors_all)
```

**Arguments:**

*   `eam`: An instance of the EAM potential.
*   `sys`: The molecular system.
*   `neighbors_all`: A precomputed list of neighbors for each atom.

**Returns:**

*   `forces_particle`: An array of Svector containing the forces on each particle in the system.

**Example:**

```julia
include("src/JuliaEAM.jl")

# 1. Read potential
eam = EAM()
fname = "Al99.eam.alloy"
read_potential!(eam, fname)

# 2. Calculate potential energy
atoms_ase_sim = convert_ase_custom(molly_system)
neighbors_all = get_neighbors_all(molly_system)

# Run first time before timing
AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
calculate_energy(eam, molly_system, neighbors_all)

println("Calculating potential energy using ASE EAM calculator")
@time E_ASE = AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
println("Calculating potential energy using my EAM calculator")
@time E_my = calculate_energy(eam, molly_system, neighbors_all)
@printf("ASE EAM calculator: %e eV\n", ustrip(E_ASE))
@printf("My EAM calculator: %e eV\n", ustrip(E_my))
@printf("Difference: %e eV\n", ustrip(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal) - calculate_energy(eam, molly_system, neighbors_all)))

function eam_ASE()
    AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
end
function eam_my()
    calculate_energy(eam, molly_system, neighbors_all)
end

n_repeat = 10
t0 = time()
E_ASE = repeat(eam_ASE, n_repeat)
t1 = time()
E_ASE = repeat(eam_my, n_repeat)
t2 = time()

println("time/atom/step by ASE EAM calculator: ", (t1-t0)/n_repeat/length(molly_system.atoms), " seconds")
println("time/atom/step by my EAM calculator: ", (t2-t1)/n_repeat/length(molly_system.atoms), " seconds")

# 3. Calculate force
# Run first time before timing
forces_ASE = AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
forces_my = calculate_forces(eam, molly_system, neighbors_all)

println("Calculating forces using ASE EAM calculator")
@time forces_ASE = AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
println("Calculating forces using my EAM calculator")
@time forces_my = calculate_forces(eam, molly_system, neighbors_all)

@printf("Sum of forces by ASE EAM calculator: [%e %e %e] eV/Å\n", ustrip(sum(forces_ASE))...)
@printf("Sum of forces by my EAM calculator: [%e %e %e] eV/Å\n", ustrip(sum(forces_my))...)

forces_err = forces_my - forces_ASE
index_max_forces_err = argmax([sqrt(sum(fe.^2)) for fe in forces_err])
@printf("Max force error: %e eV/Å\n", ustrip(sqrt(sum(forces_err[index_max_forces_err].^2))))

function eam_ASE_f()
    AtomsCalculators.forces(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal)
end
function eam_my_f()
    calculate_forces(eam, molly_system, neighbors_all)
end

eam_ASE_f()
eam_my_f()
n_repeat = 10
t0 = time()
E_ASE = repeat(eam_ASE_f, n_repeat)
t1 = time()
E_ASE = repeat(eam_my_f, n_repeat)
t2 = time()

println("time/atom/step by ASE EAM calculator: ", (t1-t0)/n_repeat/length(molly_system.atoms), " seconds")
println("time/atom/step by my EAM calculator: ", (t2-t1)/n_repeat/length(molly_system.atoms), " seconds")
```

**Outputs:**

```bash
Calculating potential energy using ASE EAM calculator
  0.095757 seconds (1.43 M allocations: 66.707 MiB, 40.00% gc time)
Calculating potential energy using my EAM calculator
  0.018727 seconds (28.82 k allocations: 13.442 MiB)
ASE EAM calculator: -3.187108e+04 eV
My EAM calculator: -3.187108e+04 eV
Difference: 1.091394e-11 eV
time/atom/step by ASE EAM calculator: 5.454806508701079e-6 seconds
time/atom/step by my EAM calculator: 1.814375071708839e-6 seconds

Calculating forces using ASE EAM calculator
  0.056560 seconds (1.43 M allocations: 67.147 MiB)
Calculating forces using my EAM calculator
  0.047702 seconds (86.48 k allocations: 45.297 MiB, 26.39% gc time)
Sum of forces by ASE EAM calculator: [-2.368150e-12 -2.400567e-12 2.278473e-14] eV/Å
Sum of forces by my EAM calculator: [6.916488e-14 -9.799304e-15 2.103179e-14] eV/Å
Max force error: 1.862710e-06 eV/Å
time/atom/step by ASE EAM calculator: 5.650160250819211e-6 seconds
time/atom/step by my EAM calculator: 4.7440777693101735e-6 seconds
```

Tested on an AMD EPYC 9334 2.7 GHz CPU on `analysis.sns.gov` cluster. For reference, the serial LAMMPS EAM calculator takes 1.85e-6 CPU seconds per atom per step for energy calculation on 3.47 GHz Intel Xeon processors. ([reference](frozen_atoms))

---

## `EAM/`: Incorporating the EAM forcefield to benchmark the Al adatom toy model
### `test_ABC_J.ipynb`: The pure Julia based ABC simulator function, calls `/src/juliaEAM.jl` for evaluating the EAM interaction

Updates: 

Ongoing
- [ ] Asymmetrical penalty
- [ ] Using ABC to study vacancy diffusion problem
- [ ] Improving visualization

Done
- [x] Assigning the penalty energy only on focused particles
- [x] Using energy difference as termination condition in minimization algorithm
- [x] FIRE algorithm and MD-downhill energyminimization
- [x] Slightly distort BC to create asymmetry
- [x] Pre-evaluating the gradient of gaussian penalty for efficiency
- [x] Truncating the penalty function to $3 \sigma$
- [x] Setting intervals between neighbor list identification for improved efficiency
- [x] Pushing the system slightly before minimization to prevent trapping on flat top PEL

Failed
- [x] ~~Using momentum gradient descent algorithm for structural relaxation~~
- [x] ~~Not updating structure if force does not converge~~

---

### Previous testings

### `test_minimize.ipynb`: Integrating the EAM forcefield for metallic element interactions from the Python [ASE](https://gitlab.com/ase/ase) package into the Molly system/simulator framework

![EAM](https://github.com/ch-tung/ABCD_J/blob/d94d2efdf2ac7988aea6161c3682c5a385521688/EAM.png?raw=true)

---

### `test_JuliaEAM.ipynb`: Calculating EAM interactions using Julia

![JuliaEAM](https://github.com/ch-tung/ABCD_J/blob/389a2293c5a2f5351905167a624ee9843e1ae479/JuliaEAM.png?raw=true)

---

### `test_ABC.ipynb`: Defining a custom ABC simulator function

![ABC](https://github.com/ch-tung/ABCD_J/blob/7fe5081cf97966e08ad64c2363283cd5736bd206/ABC.png?raw=true)

#### Constructor for ABCSimulator

```julia
ABCSimulator(; sigma=0.01*u"nm", W=1e-2*u"eV", max_steps=100, max_steps_minimize=100, step_size_minimize=0.01*u"nm", tol=1e-10*u"kg*m*s^-2", log_stream=devnull)
```

**Arguments:**

*   `sigma`: The value of sigma in units of nm.
*   `W`: The value of W in units of eV.
*   `max_steps`: The maximum number of steps for the simulator.
*   `max_steps_minimize`: The maximum number of steps for the minimizer.
*   `step_size_minimize`: The step size for the minimizer in units of nm.
*   `tol`: The tolerance for convergence in units of kg\*m\*s^-2.
*   `log_stream`: The stream to log the output.

#### Simulates the system using the `ABCSimulator`

```julia
simulate!(sys, sim::ABCSimulator; n_threads::Integer=Threads.nthreads(), frozen_atoms=[], run_loggers=true, fname="output.txt")
```

**Arguments:**

*   `sys`: The system to be simulated.
*   `sim`: An instance of the ABCSimulator.
*   `n_threads`: The number of threads to use for parallel execution. Defaults to the number of available threads.
*   `frozen_atoms`: A list of atoms that should be frozen during the simulation.
*   `run_loggers`: A boolean indicating whether to run the loggers during the simulation.
*   `fname`: The name of the output file.

**Example:**

```julia
molly_system = initialize_system()

# 1. Start from an energy minimum
simulator = SteepestDescentMinimizer(step_size=1e-3*u"nm", tol=1e-12*u"kg*m*s^-2", log_stream=devnull)
Molly.simulate!(molly_system, simulator)
atoms_ase_sim = convert_ase_custom(molly_system)
println(AtomsCalculators.potential_energy(pyconvert(AbstractSystem, atoms_ase_sim), eam_cal))

# 2. Specify the atoms to be frozen
z_coords = [coords[3] for coords in molly_system.coords]
frozen_atoms = [index for (index, z_coord) in enumerate(z_coords) if z_coord < al_LatConst*2.9*0.1*u"nm"]
println(length(frozen_atoms))

# 3. Run ABCSimulator
sigma = 2e-3
W = 0.1
@printf("sigma = %e nm/dof^1/2\n W = %e eV", ustrip(sigma), ustrip(W))
simulator = ABCSimulator(sigma=sigma*u"nm", W=W*u"eV", max_steps=100, max_steps_minimize=60, step_size_minimize=1.5e-3*u"nm", tol=1e-12*u"kg*m*s^-2")
simulate!(molly_system, simulator, n_threads=1, fname="output_test.txt", frozen_atoms=frozen_atoms)

# 4. Visualize
using GLMakie
color_0 = :blue
color_1 = :red
colors = [index < length(molly_system.coords) ? color_0 : color_1 for (index, value) in enumerate(molly_system.coords)]
visualize(molly_system.loggers.coords, boundary_condition, "test.mp4", markersize=0.1, color=colors)
```

![PE](https://github.com/ch-tung/ABCD_J/blob/main/PE_steps.png?raw=true)  
Penalty energy can gradually push the configuration out of its initial energy minimum. The potential energy gradually saturating before a steep drop. Once passing the saddle point, the penalty term essentially becomes zero. The unreasonable results for steps over 700 were due to incorrect penalty force calculations in PBC, which are easy to fix.
