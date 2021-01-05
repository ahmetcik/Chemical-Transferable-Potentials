function legendre_pos_m(lmax, x, Plm)
    """ Associated Legendre polynomials only with positive M. """

    a = (1. - x^2)
    b = sqrt(a)
    Plm[1] = 1.
    if lmax > 0
        Plm[2] = x 
        Plm[3] = - b
        if lmax > 1
            Plm[4] = 0.5 * (3. * x^2 - 1.)
            Plm[5] = -3. * x * b
            Plm[6] =  3. * a

            if lmax > 2
                Plm[7] = 0.5 * (5. * x^3 - 3. * x)
                Plm[8] = -3. / 2. * (5. * x^2 - 1. ) * b
                Plm[9] =  15. * x * a
                Plm[10] =  -15. * a * b
            end
        end
    end
    
    

    i_lminus = 4
    i_l = 7
    for l=4:lmax
        i_lplus = l*(l+1) ÷ 2 + 1
        
        Plm[i_lplus] = ((2*l-1) * x * Plm[i_l] - (l-1) * Plm[i_lminus]) / l
        
        for m=1:l
            @inbounds Plm[i_lplus+m] = ((l+1-m) * x * Plm[i_lplus+m-1] - (l+m-1) * Plm[i_l+m-1]) / b
        end

        i_lminus = i_l
        i_l = i_lplus

    end
    return nothing
end

function fact_quotient(l, m)
    if 2m > 21
        fact = factorial(big(2m))
    else
        fact = factorial(2m)
    end
    return 1/(binomial(l+m, l-m) * fact)
end
function get_ylm(lmax)
    ylm = Array{Float64}(undef, lmax+1, lmax+1)
    for l=0:lmax for m=0:lmax
        if m == 0
            ylm[l+1, m+1] = sqrt((2l +1) / 4pi)
        else
            # abs needed as very small numbers could become negative
            ylm[l+1, m+1] = (-1)^m * sqrt(abs((2l + 1) / 2pi * fact_quotient(l,  m)))
        end
    
    end end
    return ylm
end

function get_sincos(lmax, phi, sincos)
    for m=1:lmax
        sincos[1, m]= sin(m * phi)
        sincos[2, m]= cos(m * phi)
    end
    return nothing
end


function spherical_harmonics_series(lmax, x, y, z, Plm, Ylm, ylm, sincos)
    theta = atan(sqrt(x^2 + y^2), z)
    phi = atan(y, x)
    legendre_pos_m(lmax, cos(theta), Plm)
    get_sincos(lmax, phi, sincos)
    c = 1
    for l=0:lmax
        i_l = (l+1)^2 - l

        @inbounds Ylm[i_l] = ylm[l+1, 1] * Plm[c]
        c += 1
        for m=1:l
            @inbounds Ylm[i_l-m] = ylm[l+1, m+1] * Plm[c] * sincos[1, m]
            @inbounds Ylm[i_l+m] = ylm[l+1, m+1] * Plm[c] * sincos[2, m]
            c+=1
        end
    end
    return nothing
end


function sum_environmental_to_structural2(d, r_cut, cutoff_width, lmax, nmax, shift_3d)
    n = size(d, 1)
    d = reshape(d, n, 3)
    n_spherical = (lmax+1)^2
    g = zeros( n_spherical * (nmax+1))
    r_cut_start = r_cut - cutoff_width
    
    Plm = Array{Float64}(undef, lmax*(lmax+1) ÷ 2 + lmax+1)
    Ylm = Array{Float64}(undef, (lmax+1)^2)
    ylm = get_ylm(lmax)
    sincos = Array{Float64}(undef, 2, lmax)
    polynomials = ones(Float64, nmax+1)

    for i=1:n
        @inbounds d1 = d[i, 1]
        @inbounds d2 = d[i, 2]
        @inbounds d3 = d[i, 3]
        @inbounds r1 = d1 - shift_3d[1]
        @inbounds r2 = d2 - shift_3d[2]
        @inbounds r3 = d3 - shift_3d[3]

        r_radial = sqrt(r1^2 + r2^2 + r3^2)
        spherical_harmonics_series(lmax, r1, r2, r3, Plm, Ylm, ylm, sincos)

       # polynomials
        @inbounds polynomials[1] = 1
        for k=1:nmax
            @inbounds polynomials[k+1] = polynomials[k] * r_radial
        end

        c = 1
        if d1 > r_cut_start || d2 > r_cut_start
            if d1 > r_cut_start && d2 > r_cut_start
                f_cut = 0.25 * (cos(pi / cutoff_width * (d1 - r_cut_start)) + 1.) * (cos(pi / cutoff_width * (d2 - r_cut_start)) + 1.)
            elseif d1 > r_cut_start
                f_cut = 0.5 * (cos(pi / cutoff_width * (d1 - r_cut_start)) + 1.)
            else 
                f_cut = 0.5 * (cos(pi / cutoff_width * (d2 - r_cut_start)) + 1.)
            end

            for j=1:n_spherical for k=1:nmax+1
                @inbounds g[c] += Ylm[j] * polynomials[k] * f_cut
                c+=1
            end end
        else
            for j=1:n_spherical for k=1:nmax+1
                @inbounds g[c] += Ylm[j] *  polynomials[k]
                c+=1
            end end
        end
    end
    return g       
end





function sum_environmental_to_structural2(d, r_cut, cutoff_width, r_centers, sigmas)
    n = size(d, 1)
    d = reshape(d, n, 3)
    s = size(sigmas, 1)
    g = zeros( s)
    r_cut_start = r_cut - cutoff_width
    

    for i=1:n
        @inbounds d1 = d[i, 1]
        @inbounds d2 = d[i, 2]
        @inbounds d3 = d[i, 3]
        
        
       
        c = 1
        if d1 > r_cut_start || d2 > r_cut_start
            if d1 > r_cut_start && d2 > r_cut_start
                f_cut = 0.25 * (cos(pi / cutoff_width * (d1 - r_cut_start)) + 1.) * (cos(pi / cutoff_width * (d2 - r_cut_start)) + 1.)
            elseif d1 > r_cut_start
                f_cut = 0.5 * (cos(pi / cutoff_width * (d1 - r_cut_start)) + 1.)
            else 
                f_cut = 0.5 * (cos(pi / cutoff_width * (d2 - r_cut_start)) + 1.)
            end


            for j=1:s
                @inbounds g[j] += exp(-0.5 * ((d1-r_centers[j, 1])^2 + (d2-r_centers[j, 2])^2 + (d3-r_centers[j, 3])^2) / sigmas[j]^2) * f_cut
            end
        else

            for j=1:s
                @inbounds g[j] += exp(-0.5 * ((d1-r_centers[j, 1])^2 + (d2-r_centers[j, 2])^2 + (d3-r_centers[j, 3])^2) / sigmas[j]^2) 

            end
        end
    end
    return g       
end








