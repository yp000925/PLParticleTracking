%{
Simulate forward formation of a hologram from a 3D optical field

Inputs:
  phasemap  : phasemap from [-pi,pi]
  params  : Input parameters structure. Must contain the following
      - Nx, Ny, nz  : Size of the volume (in voxels) in x, y, and z
      - z_list      : List of reconstruction planes (um)
      - pp_holo     : Pixel size (um) of the image
      - wavelength  : Illumination wavelength (um)

Outputs:
  holo    : Ny-by-Nx 2D real hologram (estimated image)
%}

function holo = ForwardProjection2D(phasemap, otf)  
    
    Fholo = (fft2(phasemap).* otf);
    
    holo = ifft2(Fholo);
    
end