### **Pseudo-code for `calculate_forces`**

---

1. **Initialize Parameters and Variables**
   - Define particle type list and find the index of the current type.
   - Preallocate arrays for:
     - `forces_particle`: Forces on each particle.
     - `total_density`: Total electron density for each particle.
     - Neighbor-related distances and positions: `r_all`, `d_all`.

2. **Precompute Neighbor Distances and Electron Densities**
   For each particle $i$:
   - Retrieve the list of neighbors.
   - If no neighbors, skip the particle.
   - Compute pairwise distances $r_{ij}$ and their magnitudes $d_{ij}$ for all neighbors $j$.
   - **Store distance vectors $r_{ij}$ and magnitudes $d_{ij}$ for reuse.**

   For each element type in the neighbors of $i$:
   - Compute the electron density contribution for all neighbors.
   - Sum contributions to update the **total density for particle $i$.**

3. **Compute Forces**
   For each particle $i$:
   - Retrieve neighbors and corresponding precomputed distance data.
   - Compute the **derivative of the embedded energy with respect to the total density**.
   - Normalize the distance vectors to unit direction vectors.

   For each element type in the neighbors of $i$:
   - Retrieve distance and directional data for neighbors of type $j$.
   - Compute:
        **a. Pairwise Forces**:
    Evaluate the pairwise force between atom $i$ and its neighbors using the spline:
    $$
    \text{scale} = \phi'(r_{ij})
    $$  
    Where $\phi'(r_{ij})$ is pre-tabulated in `eam.d_phi`.

        **b. Add Electron Density Contribution from $i$**:  
    For atom $i$, compute its contribution to the force based on the derivative of its embedded energy and the derivative of electron density:
    $$
    \text{scale} += \left( \frac{\partial E_i}{\partial \rho_i} \cdot \frac{\partial \rho}{\partial r_{ij}} \right)
    $$
    This term uses `d_embedded_energy_i` and `eam.d_electron_density[j_type]`.

        **c. Add Electron Density Contribution from Neighbors $j$**:
        For each neighbor $j$, compute their contribution to the force using their embedded energy derivative and electron density derivative:
    $$
    \text{scale} += \left( \frac{\partial E_j}{\partial \rho_j} \cdot \frac{\partial \rho}{\partial r_{ij}} \right)
    $$
    This term uses `eam.d_embedded_energy[j_type](total_density_j)` and `d_electron_density_i`.
   
    Now the force of neighbor $j$ on particle $i$ is the unit vector between them multiplied with $\text{scale}$.

4. **Return Forces**
    Convert forces to the required units and return as a vector of 3D force vectors.