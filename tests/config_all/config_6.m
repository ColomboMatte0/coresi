% -*- matlab -*- (enables emacs matlab mode)

%% Input parameters
DATA_FILE = 'tests/test_mlem.dat';

PRECALCULATED_EV = 0;                  % The preprocessed events were stored =1 or not =0

%% Output parameters
%% This will be changed dynamically by the testing code
RESULTS_DIR = RESULT_PLACEHOLDER
%% Data type
DATA_TYPE = 'MACACO' ;%'MEGALIB';       % 'MEGALIB' or 'IPNL' or 'WP3'
% DATA_TYPE = 'IPNL';       % 'MEGALIB' or 'IPNL' or 'WP3'

PSF = 'OFF';
%% Sensitivity 
SENSITIVITY = 'ON' ;       % 'ON', 'OFF'. Recommended MODEL 'cos0rho0' when sensitivity is 'OFF'
                           % and 'cos1rho2' when the SENSITIVITY is 'ON'
SENSITIVITY_FILE = 'config_6.raw';
%%   Data selection
SAMPLES =1;                            % for statistical studies
COUNTS_PER_SAMPLE = 1;
%20000       % nb of events for 7 point source
PRESELECT=1;                           % COUNTS_PER_SAMPLE is:
                                       % if 0, nb of detected events
                                       % if 1, nb of useful events
if (PRECALCULATED_EV == 0)             % First event to be analysed from the .tra file
  FIRST = 1;                            % Should be >= 1
end
STORE_EV = 0;                          % Store preprocessed =1 or not =0

%%   Reconstructed volume

VOXELS_FILTER  = [5, 5, 5];



% 
%  VOLUME_DIMENSIONS = [10.25, 10.25, 10.25]; %in cm for 7 point source
%  VOXELS =[41,41,41] ; 
%  VOLUME_CENTRE = [0, 0, 0];
%  

% VOLUME_DIMENSIONS = [30.25,10.25,8.25] ; %in cm for Radon source
% VOXELS = [121, 41, 33];  
% VOLUME_CENTRE = [0, 0, 0];

%VOLUME_DIMENSIONS = [30.25,8.25,10.25] ; %in cm for Radon source, vertical
%VOXELS = [121, 33, 41];
%MACACO
VOLUME_DIMENSIONS = [20,20,0.4] ;
VOXELS = [50, 50, 1];  
VOLUME_CENTRE = [0, 0, 0];
% 

%   Iterations
DATA_TYPE_MEGA=0;   %0=sivan, 1=revan, 0 is default value
FIRST_ITERATION=0;                     % continue iterations;
                                       % 1 means from the beginning
ITERATIONS = 0;
LAST_ITERATION=FIRST_ITERATION+ITERATIONS;
ALGORITHM = 7;          % 0=CV, 1=CV_eu (energy uncertainty, sigma on angle), 
                        % 2=CV_su (spatial uncertainty), 3=CV_esu (energy and spatial uncertainty)
                        % 4=spectral continuous, 5=spectral discrete with CVs, 6=spectral discrete with EU
			% 7=spectral discrete in 4D, 8==spectral discrete in 4D with EU 9=spectral continuous in 4D
                        % -1=RTS, -2=RTV (not implemented)

SENSITIVITY_MODEL = 'with_attenuation_exp'; 
MODEL = 'cos1rho2';       % cos0rho0 que le K et la gaussienne,
                          % cos1rho1 cos/rho, cos1rho2 cos/rho^2, cos0rho2
                          % 1/rho^2

WIDTH_FACTOR=1;         % RADON 
NB_SIGMA = 2;


ALPHA_TV=0;%normalized true image: methode A, alpha_usual=1/s_j0*efficacity_j0
TV_ITERATION=20;

%%   Energy a priori
ENERGY_FLAG = 4;               % ANY=0, RANGE=1, KNOWN=2, LINEAR SPECTRAL=3, FIXED SPECTRAL=4
if (ENERGY_FLAG==1)            % total energy in some range
  ENERGY_MIN = 1450.5;         % lower bound, keV
  ENERGY_MAX = 1551.5;         % upper bound, keV
end
if (ENERGY_FLAG==2)            % known total energy
  ENERGY_TOTAL=140;           % the total energy, keV
end
if (ENERGY_FLAG==3)            % spectral reconstruction
  ENERGY_MIN = 65;            % lower bound, keV
  ENERGY_MAX = 665;           % upper bound, keV
  nE = 41;		       % number of energy bins
end
if (ENERGY_FLAG==4)            % spectral reconstruction
  nE = 4;		       % number of energy bins
  ENERGIES = [140,245,364,511] % fixed energies to try
end

%%   Camera properties
% Coordinates w.r. to the absolute frame


% %Gate
NB_LAYERS = 7;
LAY_CENTRES = [-10, -11, -12, -13, -14, -15, -16];   % supposed to be on a line || Oz,
ABS_CENTRE = -31;          % -29+0.115   ArrayDistance_absorber+LaBr3CrystalBlock.Position
LAY_SIZE = [9.0,9.0,0.2];      % Wafer.Shape *2
ABS_SIZE = [28, 21, 3];  % PixelDetectorCal.Shape *2 (for x and y)
ABS_VOXELS= [8, 6, 1];  
%Camera materials: 0 = Si, 1 = BGO, 2 = LaBr3
MATERIAL_SCA = 0
MATERIAL_ABS = 0
% 

% SPATIAL_UNCERTAINTY = 1;         % 1=yes, 0=no
if (ALGORITHM==2)
    ABS_VOXEL_SAMPLING = 4;          % nb points to sample a voxel vertically;
    WEIGHTS = [0, 0, 0, 1];           % must have ABS_VOXEL_SAMPLING elements; first is on bottom
%     ABS_VOXEL_SAMPLING = 1;          % nb points to sample a voxel vertically;
%     WEIGHTS = [1];           % must have ABS_VOXEL_SAMPLING elements
end
 


NB_CAMERAS = 1;
% 
FRAME_0_ORIGIN = [0, 0, 0];
Ox_0 = [1, 0, 0];     % parallel to scatterer edge
Oy_0 = [0, 1, 0];     % parallel to scatterer edge
Oz_0 = [0, 0, 1];    % orthogonal to the camera, tw the source


%% Hodoscope a priori
HODOSCOPE_FLAG=0;                           % ON=1, OFF =0;
if (HODOSCOPE_FLAG > 0)
  BEAM_SIGMA = 2 ;                  % Gaussian hodoscope, cm
  BEAM_WIDTH_FACTOR =3 ;            % number of sigma
  BEAM_ENTRY_POINT =[-10, 0, 0];         % a point on the beam line
  BEAM_DIRECTION =[1, 0, 0] ;            % direction of the beam
  shift=0;
  BEAM_FIRST_ITERATION=FIRST_ITERATION+shift;
  BEAM_ITERATIONS=ITERATIONS-shift;
  BEAM_INCLUSION='ONLYHODO';         % CONSTANT, LINEAR, FORCE, FORCEINV, ALTERNATE
end
