function [u, v, w] = multi_resolution_LK(I1, I2, scales, r,non_zero_mask)

    % Initialize the displacement estimates
    u = zeros(size(I1));
    v = zeros(size(I1));
    w = zeros(size(I1));

    for s = scales:-1:0
        % Subsample the images
        new_size = round(size(I1) * (1/(2^s)));
        I1_s = imresize3(I1, new_size, 'linear');
        I2_s = imresize3(I2, new_size, 'linear');

        % Calculate the derivatives
        [Ix, Iy, Iz, It] = LK3D_derivatives(I1_s, I2_s);

        % Upsample the displacement estimates to the current scale
        u_s = imresize3(u, size(I1_s), 'linear') * 2;
        v_s = imresize3(v, size(I1_s), 'linear') * 2;
        w_s = imresize3(w, size(I1_s), 'linear') * 2;

        % Calculate the optical flow at the current scale using the method from the paper
        A = cat(4, Ix, Iy, Iz);
        b = -It;

        % Solve the linear system for each neighborhood
        du_s = zeros(size(I1_s));
        dv_s = zeros(size(I1_s));
        dw_s = zeros(size(I1_s));

        for x = 1+r:size(I1_s, 1)-r
            for y = 1+r:size(I1_s, 2)-r
                for z = 1+r:size(I1_s, 3)-r
                    idx = x-r:x+r;
                    idy = y-r:y+r;
                    idz = z-r:z+r;
        
                    A_local = A(idx, idy, idz, :);
                    b_local = b(idx, idy, idz);
        
                    A_local = reshape(A_local, [], 3);
                    b_local = b_local(:);
        
                    % Calculate the structure tensor
                    ATA = A_local' * A_local;
        
                    % Calculate the eigenvalues
                    eigenvalues = eig(ATA);
        
                    if all(eigenvalues > 10000)
                        % All eigenvalues are less than 1.0, so reject the displacement estimate
                        du_s(x, y, z) = 0;
                        dv_s(x, y, z) = 0;
                        dw_s(x, y, z) = 0;
                    else
                        % At least one eigenvalue is 1.0 or greater, so calculate the displacement estimate
                        d_local = pinv(A_local) * b_local;
                        du_s(x, y, z) = d_local(1);
                        dv_s(x, y, z) = d_local(2);
                        dw_s(x, y, z) = d_local(3);
                    end
                end
            end
        end

        % Update the displacement estimates
        u = u_s + du_s;
        v = v_s + dv_s;
        w = w_s + dw_s;

        % Discard displacements larger than 2^(scale+1)
        displacement_magnitude = sqrt(u.^2 + v.^2 + w.^2);
        max_displacement = 2^(s+1);
        out_of_range = displacement_magnitude > max_displacement;
        u(out_of_range) = 0;
        v(out_of_range) = 0;
        w(out_of_range) = 0;
    end
    u = u .* non_zero_mask;
    v = v .* non_zero_mask;
    w = w .* non_zero_mask;
end
