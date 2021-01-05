# TODO the two functions elemental and binary need to be put into one
# and comments need to be added.

function elemental(i, j, d, D, atoms_numbers)
    
    n_atoms = size(atoms_numbers, 1)

    i = i .+ 1
    j = j .+ 1

    n_i  = zeros(Int, n_atoms)
    for ii=1:size(i, 1)
        n_i[i[ii]] += 1 
    end
    
    i_grouped = [Array{Int}(undef, n) for n in n_i]
    counter_group = ones(Int, n_atoms)

    for ii=1:size(i, 1)
        i_grouped[i[ii]][counter_group[i[ii]], 1] = ii
        counter_group[i[ii]] += 1
    end
    
    n_triplets::Int = 0
    for ii=1:n_atoms
        n_triplets += (counter_group[ii]-1) * (counter_group[ii]-2) / 2
    end

    D = copy(transpose(D))
    d_out = Array{Float64}(undef, 3, n_triplets)
    counter = 1
    for i_atom=1:n_atoms
      @inbounds for iii=1:n_i[i_atom] for jjj=iii+1:n_i[i_atom]
            @inbounds ii = i_grouped[i_atom][iii]
            @inbounds jj = i_grouped[i_atom][jjj]
            
            b = 0.
            for s=1:3
                @inbounds b += (D[s, ii] - D[s, jj])^2
            end
            b = sqrt(b)

            @inbounds d_out[1, counter] = d[ii]
            @inbounds d_out[2, counter] = d[jj]
            @inbounds d_out[3, counter] = b
            counter += 1
        end
        end
    end
    return d_out
end

function binary(i, j, d, D, atoms_numbers)
    
    atoms_tuples = sort(collect(Set(atoms_numbers)))
    A, B = atoms_tuples
    n_atoms = size(atoms_numbers, 1)
    s = size(atoms_tuples, 1)
    a= [(atoms_tuples[i], atoms_tuples[j], atoms_tuples[k]) for i=1:s for j=1:s for k=j:s]
    sizes_dict = Dict( aa => 0 for aa=a)

    i = i .+ 1
    j = j .+ 1

    bins = zeros(Int, 2, n_atoms)
    n_i  = zeros(Int, n_atoms)
    for ii=1:size(i, 1)
        n_i[i[ii]] += 1 
        if atoms_numbers[j[ii]] == atoms_tuples[1]
            bins[1, i[ii]] += 1
        else
            bins[2, i[ii]] += 1
        end
    end
    
    i_grouped = [Array{Int}(undef, n) for n in n_i]
    counter_group = ones(Int, n_atoms)

    for ii=1:size(i, 1)
        i_grouped[i[ii]][counter_group[i[ii]], 1] = ii
        counter_group[i[ii]] += 1
    end

    for ii=1:n_atoms
        if bins[1, ii] > 1
            sizes_dict[(atoms_numbers[ii], atoms_tuples[1], atoms_tuples[1])] += (bins[1, ii] - 1)bins[1, ii] / 2 
        end

        if (bins[1, ii] > 0) & (bins[2, ii] > 0)
            sizes_dict[(atoms_numbers[ii], atoms_tuples[1], atoms_tuples[2])] += bins[1, ii] * bins[2, ii]
        end
        
        if bins[2, ii] > 1
            sizes_dict[(atoms_numbers[ii], atoms_tuples[2], atoms_tuples[2])] += (bins[2, ii] - 1)bins[2, ii] / 2 
        end
    end

    D = copy(transpose(D))
    ds = [Array{Float64}(undef, 3, sizes_dict[aa]) for aa=a] 
    counter = [1 for aa=a]
    for i_atom=1:n_atoms
      @inbounds for iii=1:n_i[i_atom] for jjj=iii+1:n_i[i_atom]
            @inbounds ii = i_grouped[i_atom][iii]
            @inbounds jj = i_grouped[i_atom][jjj]
            
            @inbounds if atoms_numbers[j[ii]] < atoms_numbers[j[jj]]
                @inbounds tup = (atoms_numbers[i_atom], atoms_numbers[j[ii]], atoms_numbers[j[jj]])
            else
                @inbounds tup = (atoms_numbers[i_atom], atoms_numbers[j[jj]], atoms_numbers[j[ii]])
            end
            
            b = 0.
            for s=1:3
               @inbounds b += (D[s, ii] - D[s, jj])^2
            end
            b = sqrt(b)

            if     (tup[1] == A) & (tup[2] == A) & (tup[3] == A)
                idx = 1
            elseif (tup[1] == A) & (tup[2] == A) & (tup[3] == B)
                idx = 2
            elseif (tup[1] == A) & (tup[2] == B) & (tup[3] == B)
                idx = 3
            elseif (tup[1] == B) & (tup[2] == A) & (tup[3] == A)
                idx = 4
            elseif (tup[1] == B) & (tup[2] == A) & (tup[3] == B)
                idx = 5
            else
                idx = 6
            end
            c = counter[idx]
            @inbounds ds[idx][1, c] = d[ii]
            @inbounds ds[idx][2, c] = d[jj]
            @inbounds ds[idx][3, c] = b
            counter[idx] += 1
        end
        end
    end
    return ds
end

function get_3b_from_2b_desc(i, j, d, D, atoms_numbers)
    if length(Set(atoms_numbers)) == 1
        elemental(i, j, d, D, atoms_numbers)
    else
        binary(i, j, d, D, atoms_numbers)
    end
end

