function [Ix, Iy, Iz, It] = LK3D_derivatives(I1, I2)

    % Calculate the spatial gradients using a Sobel filter
    Ix = convn(I1, (1/8) * [-1 0 1; -2 0 2; -1 0 1], 'same');
    Iy = convn(I1, (1/8) * [-1 -2 -1; 0 0 0; 1 2 1], 'same');
    Iz = convn(I1, (1/8) * [-1 -2 -1; -2 4 2; -1 -2 -1], 'same');

    % Calculate the temporal gradient
    It = I2 - I1;

end
