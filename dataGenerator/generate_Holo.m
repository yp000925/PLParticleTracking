%% different density dataset
% generate different density for test the network 
%%
close all; clear; clc;
cd
%%
data_dir = '../data/diff_densityP/';
data_num = 5;
noise_level =0;

sr = 20e-6; % pixel size of particles

Nz = 7;  dz = (60e-3-12e-3)/Nz;
lambda = 660e-9;     % Illumination wavelength
z0     = 12e-3;       % Distance between the hologram and the center plane of the 3D object
z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object

data_mode = 0; % 0 for store kmap data 

Nxy = 1024;  pps  = 3.45e-6;   % pixel pitch of CCD camera
Nxy_qis = 1024; pps_qis = 2.2e-6; % pixel pitch of QIS camera

Nxy_network = 256;  pps_network =pps*Nxy/Nxy_network;      % rescaled pixel pitch for network computational cost

% for generating the projection kernel, the rescaled version is used as the params 
params.lambda = lambda;
params.pps = pps_network;
params.z = z_range;
params.Ny = Nxy_network;
params.Nx = Nxy_network;
ori_otf3d = ProjKernel(params);

% params.lambda = lambda;
% params.pps = pps;
% params.z = z_range;
% params.Ny = Nxy;
% params.Nx = Nxy;
% ori_otf3d2 = ProjKernel(params);


params.K                =  round(Nxy_qis/Nxy_network);                % Spatial  Overasampling Factor
params.T                =  100;  % Temporal Overasampling Factor
params.Qmax             =  2;               % Maximum Threshold
params.alpha = 1;
%%
np_min = 50;
np_max = 50;
% ppv_min = 1e-3;
% ppv_max = 5e-3;
% d = Nxy_network*pps_network;
ppv_min = 1e+9;
ppv_max = 3e+9;
% np_min = round(ppv_min*d*d*Nz*dz); 
% np_max = round(ppv_max*d*d*Nz*dz);

% if ppv_min == ppv_max
%     ppv_text = [num2str(ppv_min,'%.e')];
% else
%     ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
% end
% name = ['Nz', num2str(Nz), '_Nxy', num2str(Nxy_network),'_kt',num2str(params.T),'_ks',num2str(params.K),'_ppv',ppv_text,'_r',num2str(sr,'%.e'),'_pps',num2str(params.pps,'%.e')];
% 
if data_mode == 0
    data_dir = [data_dir,'Kmap_', num2str(np_min),'_particle'];
else
    data_dir = [data_dir,'Full_', num2str(np_min),'_particle'];
end
if not(exist(data_dir,'dir'))
    mkdir(data_dir)
end
%%
 

for idx = 1:data_num
    data = zeros(Nxy_network,Nxy_network);
    label = zeros(Nz,params.Nx,Nxy_network);
    y = zeros(params.T,Nxy_network*params.K,Nxy_network*params.K);
    N_random = randi([np_min np_max], 1, 1); % particle concentration
    obj = randomParticle(Nxy_network, Nz, sr/pps_network, N_random);   % randomly located particles
%     imagesc(plotdatacube(obj)); title(['3D object with particle of ' num2str(N_random) ]); axis image; drawnow; colormap(hot);
    t_o = (1-obj);
    [data_single] = gaborHolo(t_o, ori_otf3d, noise_level);
    otf3d = permute(ori_otf3d,[3,1,2]);% just permute for saving 
    data(:,:) = data_single;
    data = (data-min(min(data)))/(max(max(data))-min(min(data)));
    label(:,:,:) = permute(obj,[3,1,2]);% [Nz,Nxy,Nxy]
    y(:,:,:) = generateQIS(params,data);    
    k1 = generate_K1_map(y,params);
% 
%     if data_mode == 0
%         save([data_dir,'/',num2str(idx),'.mat'],'data','label','otf3d','k1');
%     else
%         save([data_dir,'/',num2str(idx),'.mat'],'data','label','otf3d','y');
%     end
break
       
    disp(idx)
   
end
%%
    figure();imagesc(k1);colormap(gray);
    figure();imagesc(data);colormap(gray);
