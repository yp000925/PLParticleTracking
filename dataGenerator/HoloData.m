close all; clear; clc;

% addpath(genpath('./function/'));  % Add funtion path with sub-folders
data_dir = '../data/LLParticle/';


%% Modify
data_num = 500;       % number of train data_single
noise_level =0;

sr = 20e-6; % pixel size of particles
Nz = 15;  
dz = (60e-3-12e-3)/Nz;
lambda = 660e-9;     % Illumination wavelength
z0     = 12e-3;       % Distance between the hologram and the center plane of the 3D object
z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object


Nxy = 1024;  pps  = 3.45e-6;   % pixel pitch of CCD camera
Nxy_network = 256;  pps_network =pps*Nxy/Nxy_network;      % rescaled pixel pitch for network computational cost

%%
objType = 'sim';  
lambda = 660e-9;     % Illumination wavelength
pps    = 20e-6;      % pixel pitch of CCD camera
z0     = 5e-3;       % Distance between the hologram and the center plane of the 3D object

z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object

NA = pps*Nxy/2/z0;

delta_x = lambda/(NA)
delta_z = 2*lambda/(NA^2)

params.lambda = lambda;
params.pps = pps_network;
params.z = z_range;
params.Ny = Nxy_network;
params.Nx = Nxy_network;
ori_otf3d = ProjKernel(params);
np_min = 50;
np_max = 50;
% ppv_min = 1e-3;
% ppv_max = 5e-3;
% d = Nxy_network*pps_network;
ppv_min = np_min/(params.Nx*params.Nx*Nz);
ppv_max = np_max/(params.Nx*params.Nx*Nz);
if ppv_min == ppv_max
    ppv_text = [num2str(ppv_min,'%.1e')];
else
    ppv_text = [num2str(ppv_min,'%.1e') '~' num2str(ppv_max,'%.1e')];
end

data_dir = [data_dir, 'Nxy', num2str(Nxy_network), '_Nz', num2str(Nz),'_ppv', ppv_text,'_dz', num2str(dz*1e3,'%.1f'),'mm', ...
    '_pps',num2str(pps_network*1e6,'%.1f'),'um','_lambda',num2str(lambda*1e9),'nm'];

if not(exist(data_dir,'dir'))
    mkdir(data_dir)
end


%% Generate training data


for idx = 1:data_num
    data = zeros(Nxy_network,Nxy_network);
    label = zeros(Nz,params.Nx,Nxy_network);
    N_random = randi([np_min np_max], 1, 1); % particle concentration
    obj = randomParticle(Nxy_network, Nz, sr/pps_network, N_random);   % randomly located particles
%     imagesc(plotdatacube(obj)); title(['3D object with particle of ' num2str(N_random) ]); axis image; drawnow; colormap(hot);
    t_o = (1-obj);
    [data_single] = gaborHolo(t_o, ori_otf3d, noise_level);
    otf3d = permute(ori_otf3d,[3,1,2]);% just permute for saving 
    data(:,:) = data_single;
    data = (data-min(min(data)))/(max(max(data))-min(min(data)));
    label(:,:,:) = permute(obj,[3,1,2]);% [Nz,Nxy,Nxy] 
    save([data_dir,'/',num2str(idx),'.mat'], 'data', 'label', 'otf3d');



    disp(idx)
   
end


AT = @(plane) (BackwardProjection(plane, ori_otf3d));


figure;
imagesc(data); title('Hologram'); axis image; drawnow; colormap(gray); colorbar; 
figure;
subplot(311); imagesc(plotdatacube(obj)); title('Last object'); axis image; drawnow; colormap(hot); colorbar; axis off;
subplot(312); imagesc(data); title('Hologram'); axis image; drawnow; colormap(gray); colorbar; axis off;
temp = abs(real(AT(data)));  %temp = (temp- min(temp(:)))./(max(temp(:))-min(temp(:)));
subplot(313); imagesc(plotdatacube(temp)); title('Gabor reconstruction'); axis image; drawnow; colormap(hot); colorbar; axis off;