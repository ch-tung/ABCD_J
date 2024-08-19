using Unitful
using StaticArrays
using DelimitedFiles
"""
    lmpDumpReader(filename_dump)

Reads a LAMMPS dump file and extracts the number of atoms, box size, and atomic coordinates.

# Arguments
- `filename_dump::String`: The path to the LAMMPS dump file.

# Returns
- `n_atoms::Int`: The number of atoms in the system.
- `box_size::Vector{Unitful.Length{Float64}}`: The size of the simulation box in each dimension.
- `coords_molly::Vector{SVector{3, Unitful.Length{Float64}}}': The atomic coordinates in MOLLY format.
"""
function lmpDumpReader(filename_dump)
    lines = readdlm(filename_dump, '\n', String) # read the files, split by new line
    function lines_to_list(lines) # convert the entries in lines to list
        data = []
        for line in lines
            push!(data, split(line))
        end
        return data
    end
    data = lines_to_list(lines)
    n_atoms = parse(Int,data[4][1])
    box_size = []
    box_origin = []
    for i_box in 1:3
        data_box = [parse(Float64,d) for d in data[i_box+5]]
        box_size_i = data_box[2]-data_box[1]
        push!(box_size,box_size_i*1u"Å")
        push!(box_origin,data_box[1])
    end
    id = zeros(Int,n_atoms)
    coords = []
    attypes = []
    for i_atoms in 1:n_atoms
        data_atoms = [parse(Float64,d) for d in data[i_atoms+9]]
        id[i_atoms] = data_atoms[1]
        push!(coords, data_atoms[3:5]-box_origin)
        push!(attypes, Int(data_atoms[2]))
    end
    coords_molly = [SVector{3}(c*1u"Å") for c in coords]
    return n_atoms, box_size, coords_molly, attypes
end

"""
    lmpDumpWriter(file, timestep, sys, fname_dump)

Write system information to a LAMMPS dump file.

# Arguments
- `file`: The file object to write to.
- `timestep`: The current timestep of the simulation.
- `sys`: The Molly system object containing the coordinates and boundary information.
- `fname_dump`: The filename of the dump file.
"""
function lmpDumpWriter(file,timestep,sys,fname_dump)
    # open(fname_dump, "a") do file
    write(file, "ITEM: TIMESTEP\n")
    write(file, string(timestep)*"\n")
    write(file, "ITEM: NUMBER OF ATOMS\n")
    write(file, string(length(sys.coords))*"\n")
    write(file, "ITEM: BOX BOUNDS pp pp pp\n")
    write(file, "0 "*string(ustrip(sys.boundary[1]))*"\n")
    write(file, "0 "*string(ustrip(sys.boundary[2]))*"\n")
    write(file, "0 "*string(ustrip(sys.boundary[3]))*"\n")
    write(file, "ITEM: ATOMS id type xu yu zu\n")
    for (i_c, coord) in enumerate(sys.coords)
        atomdata = sys.atoms_data[i_c]
        write(file, string(i_c)*" "*string(atomdata.atom_type)*" "*join(ustrip(coord)," ")*"\n")
    end
    # end
end

"""
    lmpDataWriter(file, timestep, sys, fname_dump)

Write LAMMPS data file.

# Arguments
- `file::IO`: The file object to write the data to.
- `timestep::Int`: The timestep of the simulation.
- `sys::System`: The system object containing the coordinates and atom data.
- `fname_dump::String`: The name of the dump file.
"""
function lmpDataWriter(file,timestep,sys,fname_dump)
    # open(fname_dump, "a") do file
    n_types = length(unique([ad.atom_type for ad in molly_system.atoms_data]))
    write(file, "# LAMMPS data file written by lammpsIO.jl\n")       #1
    write(file, "\n")                                                #2
    write(file, string(length(sys.coords))*" atoms\n")               #3
    write(file, string(n_types)*" atom types\n")                     #4
    write(file, "\n")                                                #5
    write(file, "0 "*string(ustrip(sys.boundary[1]))*" xlo xhi\n")   #6
    write(file, "0 "*string(ustrip(sys.boundary[2]))*" ylo yhi\n")   #7
    write(file, "0 "*string(ustrip(sys.boundary[3]))*" zlo zhi\n")   #8
    write(file, "\n")                                                #9
    write(file, "Atoms  # atomic\n")                                 #10
    write(file, "\n")                                                #11
    for (i_c, coord) in enumerate(sys.coords)
        atomdata = sys.atoms_data[i_c]
        write(file, string(i_c)*" "*string(atomdata.atom_type)*" "*join(ustrip(coord)," ")*"\n")
    end
    # end
end