%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optical flow analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restoredefaultpath
clearvars
clc
warning('off','all');
addpath(genpath('/data/user/rodolphe/Scripts/Origin/Myfunctions'));
addpath(genpath('/data/user/rodolphe/Toolbox/spm12'));
rmpath(genpath('/data/user/rodolphe/Toolbox/spm12/external/fieldtrip/compat/'));
addpath(genpath('/data/user/rodolphe/Toolbox/bspmview'));

cd('/data/project/amaralab/McKnight_HBC/MREG_data/HBC02/MREG_recon_GRE/Preprocessed');
myrun = 'run1';
mysave_folder = 'Optical_flow';
nproc=8;

%% Step 1: Load and preprocess the 4D fMRI image data
fprintf('Load the 4D fMRI image data\n');
fMRI_file = strcat('swrrrRecon_full_',myrun,'_masked.nii');
fMRI_nii = spm_vol(fMRI_file);
fMRI_data = spm_read_vols(fMRI_nii);

% Create a mask of non-zero voxels
non_zero_mask = std(fMRI_data, 0, 4) > 0;


%% Step 2: Apply a band-pass filter
fprintf('Apply a band-pass filter\n');
fs = 10;
low_cutoff = 0.7;
high_cutoff = 1.5;
filter_order = 64; % order of the filter

% Normalize the frequencies by the Nyquist frequency
low_cutoff_norm = low_cutoff / (fs/2);
high_cutoff_norm = high_cutoff / (fs/2);

% Design the bandpass filter
bp_filter = fir1(filter_order, [low_cutoff_norm, high_cutoff_norm]);

filtered_fMRI_data = zeros(size(fMRI_data));
for i = 1:size(fMRI_data, 1)
    for j = 1:size(fMRI_data, 2)
        for k = 1:size(fMRI_data, 3)
            % Skip the processing for zero voxels
            if ~non_zero_mask(i, j, k)
                continue
            end
            voxel_time_series = squeeze(fMRI_data(i, j, k, :));
            filtered_fMRI_data(i, j, k, :) = filtfilt(bp_filter, 1,voxel_time_series);
        end
    end
end

%% Optional step, check filter quality



% % Select a voxel's time series
% voxel_time_series = squeeze(fMRI_data(57, 58, 50, :));
% 
% % Filter the voxel's time series
% filtered_voxel_time_series = filtfilt(bp_filter2, 1,voxel_time_series);
% 
% % Compute the power spectral density for the original and filtered signal
% [psd_original, freq_original] = pwelch(voxel_time_series, [], [], [], fs);
% [psd_filtered, freq_filtered] = pwelch(filtered_voxel_time_series, [], [], [], fs);
% 
% % Plot the power spectral density for the original and filtered signal on the same plot
% figure;
% plot(freq_original, 10*log10(psd_original), 'b');
% hold on;
% plot(freq_filtered, 10*log10(psd_filtered), 'r');
% xlabel('Frequency (Hz)');
% ylabel('Power/Frequency (dB/Hz)');
% title('PSD of Original (blue) and Filtered (red) Signals');
% xlim([0 fs/2]);
% legend('Original', 'Filtered');



%% Step 3: Extract cardiovascular pulse wavefronts
sz = size(filtered_fMRI_data);
pulse_wavefront = zeros(sz);
min_distance = round(0.3 * fs); % At least 0.3 seconds distance between extrema

total_voxels = sz(1) * sz(2) * sz(3);

% Create a text waiting bar
fprintf('Extract pulse wavefronts:  0%%\n');

% Create a temporary array to store voxel time series
temp_voxel_time_series = zeros(sz(4), total_voxels);

% Use for loop instead of parfor
for idx = 1:total_voxels
    [i, j, k] = ind2sub(sz(1:3), idx);
    % Skip the processing for zero voxels
            if ~non_zero_mask(i, j, k)
                continue
            end
    % Get the time series for this voxel
    voxel_time_series = squeeze(filtered_fMRI_data(i, j, k, :));
    
    % Find local maxima and minima
    [maxima, max_locs] = findpeaks(voxel_time_series, 'MinPeakDistance', min_distance);
    [minima, min_locs] = findpeaks(-voxel_time_series, 'MinPeakDistance', min_distance);
    
    % Only keep maxima that are positive and minima that are negative
    maxima = maxima(maxima > 0);
    minima = -minima(minima < 0);
    
    % Replace local maxima with the range of the signal in that period
    for loc_idx = 1:numel(max_locs)
        if loc_idx < numel(max_locs)
            % If it's not the last maximum, the period is until the next maximum
            period = voxel_time_series(max_locs(loc_idx):max_locs(loc_idx+1));
        else
            % If it's the last maximum, the period is until the end of the time series
            period = voxel_time_series(max_locs(loc_idx):end);
        end
        
        voxel_time_series(max_locs(loc_idx)) = max(period) - min(period);
    end
    
    % Replace all other values with zero
    for inner_idx = 1:length(voxel_time_series)
        if ~ismember(inner_idx, max_locs)
            voxel_time_series(inner_idx) = 0;
        end
    end

    
    % Store the processed time series in the temporary array
    temp_voxel_time_series(:, idx) = voxel_time_series;

    % Update the text waiting bar
    if mod(idx, round(total_voxels/100)) == 0
        fprintf('Extract pulse wavefronts: %3.0f%%\n', (idx / total_voxels) * 100);
    end
end

% Update the pulse_wavefront variable with the processed time series
for idx = 1:total_voxels
    [i, j, k] = ind2sub(sz(1:3), idx);
    pulse_wavefront(i, j, k, :) = temp_voxel_time_series(:, idx);
end



%% Step 4: Estimate optical flow using multi_resolution_LK function
r = 2; % Radius of the neighbourhood for LK3D function
scale = 4;

% Initialize optical flow cell array
num_volumes = size(pulse_wavefront, 4);
optical_flow = cell(1, num_volumes - 1);

% Start a parallel pool if it's not already running
if isempty(gcp('nocreate'))
    parpool(nproc);
end

% Construct the parfor_wait object
WaitMessage = parfor_wait(num_volumes - 1, 'Waitbar', true);

clear temp_voxel_time_series voxel_time_series

% Loop through the volumes and estimate the optical flow field for each pair of consecutive volumes using parfor
parfor j = 1:(num_volumes - 1)
    % Estimate the optical flow field using the multi_resolution_LK function  
    [ux, uy, uz] = multi_resolution_LK(pulse_wavefront(:,:,:,j), pulse_wavefront(:,:,:,j+1), scale, r,non_zero_mask);

    % Store the computed flow components in a structure
    optical_flow{j}.Vx = ux;
    optical_flow{j}.Vy = uy;
    optical_flow{j}.Vz = uz;
    
    % Send a message to the parfor_wait object
    WaitMessage.Send;
end

% Destroy the parfor_wait object
WaitMessage.Destroy;

% Step 5: Calculate the average optical flow for each voxel
num_of_time_points = numel(optical_flow);
average_flow = zeros(size(optical_flow{1}.Vx));
for t = 1:num_of_time_points
    average_flow = average_flow + sqrt(optical_flow{t}.Vx.^2 + optical_flow{t}.Vy.^2 + optical_flow{t}.Vz.^2);
end
average_flow = average_flow / num_of_time_points;

% Step 6: Save the average optical flow values as a new NIfTI file

if ~exist(mysave_folder,'dir')
    mkdir(mysave_folder)
end
cd(mysave_folder)

output_file = strcat('Card_flow_',myrun,'.nii');
output_nii = fMRI_nii(1);
output_nii.fname = output_file;
output_nii.dim = size(average_flow);
spm_write_vol(output_nii, single(average_flow));

save_mat = strcat('Optical_flow_',myrun,'.mat');
save(save_mat,'optical_flow','-v7.3');

