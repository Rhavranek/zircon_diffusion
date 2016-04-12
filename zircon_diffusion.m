%% An attempt at zircon diffusion along the c-axis
%Strongly influenced by the cellular automotan code sent by GT 
% REH April, 2016

% This model shows how zircon diffusion may not be reponsive to low levels
% of damage (we get a plateau in retentivity) - only one path of diffusion is created, and the pathway is
% very tortuous 
%                  -

omega=1.0;    % = 1/tau where tau=relaxation time scale
density=4;  % initial density

% These are the weights for equilibrium distribution
t1=4/9;   % Rest particle
t2=1/9;   % "Straight" directions
t3=1/36;  % Diagonal directions

% This sneakily incorporates the factor of of 3 that shows up in the
% equilibrium distributions, so c_squ actually equals c^2/3.
c_squ=1/3; 

% Lattice size
nx=28; ny=28;

% Set up matrix. The matrix is nx by ny by 9, where 9 represents the 9
% discrete lattice velocities. F is the current distribution for each
% velocity, and FEQ is the equilibrium distribution.
F=repmat(density/9,[nx ny 9]); FEQ=F;

% ON stands for Occupied Nodes, and the ON matrix encodes the (x,y)
% locations of occupied (boundary) nodes.
msize=nx*ny; CI=[0:msize:msize*7];
    %this has to be *7 to fit with the rules below

%BOUND=zeros(nx,ny);BOUND(1:nx,1)=1;%open channel

load zircon_spaces.txt;
BOUND=zircon_spaces;
ON=find(BOUND); %matrix offset of each Occupied Node
TO_REFLECT=[ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4) ...
            ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8)];
REFLECTED= [ON+CI(5) ON+CI(6) ON+CI(7) ON+CI(8) ...
            ON+CI(1) ON+CI(2) ON+CI(3) ON+CI(4)];


avu=1;         % Average x-directed velocity
prevavu=1;     % Previous "
ts=0;          % Time-step counter
deltaU=1e-7;   % Velocity added to first row (left side)
numactivenodes=sum(sum(1-BOUND));

%%RUN

%if a row is 010 then it becomes 100


    
while (ts<4000 & 1e-10<abs((prevavu-avu)/avu)) | ts<100
    
    % Propagate (be careful about directions here ...)
    F(:,:,4)=F([2:nx 1],[ny 1:ny-1],4);     % up and right
    F(:,:,3)=F(:,[ny 1:ny-1],3);            % right
    F(:,:,2)=F([nx 1:nx-1],[ny 1:ny-1],2);  % down and right
    F(:,:,5)=F([2:nx 1],:,5);               % up
    F(:,:,1)=F([nx 1:nx-1],:,1);            % propagate down
    F(:,:,6)=F([2:nx 1],[2:ny 1],6);        % up and left
    F(:,:,7)=F(:,[2:ny 1],7);               % left
    F(:,:,8)=F([nx 1:nx-1],[2:ny 1],8);     % down and left
    
    
    BOUNCEDBACK=F(TO_REFLECT); %Densities bouncing back at next timestep
    
    % Calculate local densities as the sum of all velocity components
    DENSITY=sum(F,3);
    
    % Calculate velocities:
    % UX = sum of "rightward" minus sum of "leftward"
    % UY = sum of "upward" minus sum of "downward"
    UX=(sum(F(:,:,[1 2 8]),3)-sum(F(:,:,[4 5 6]),3))./DENSITY;
    UY=(sum(F(:,:,[2 3 4]),3)-sum(F(:,:,[6 7 8]),3))./DENSITY;
    
    % Apply boundary conditions:
    % Add velocity to left side (row 1).
    % Set velocity and density at boundary points to zero.
    % Pre-compute squares, etc.
    UX(1,1:ny)=UX(1,1:ny)+deltaU; %Increase inlet pressure
    UX(ON)=0; UY(ON)=0; DENSITY(ON)=0;
    U_SQU=UX.^2+UY.^2; U_C2=UX+UY; U_C4=-UX+UY; U_C6=-U_C2; U_C8=-U_C4;
    
    % Calculate equilibrium distribution: stationary
    FEQ(:,:,9)=t1*DENSITY.*(1-U_SQU/(2*c_squ));
    % nearest-neighbours
    FEQ(:,:,1)=t2*DENSITY.*(1+UX/c_squ+0.5*(UX/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,3)=t2*DENSITY.*(1+UY/c_squ+0.5*(UY/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,5)=t2*DENSITY.*(1-UX/c_squ+0.5*(UX/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,7)=t2*DENSITY.*(1-UY/c_squ+0.5*(UY/c_squ).^2-U_SQU/(2*c_squ));
    % next-nearest neighbours
    FEQ(:,:,2)=t3*DENSITY.*(1+U_C2/c_squ+0.5*(U_C2/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,4)=t3*DENSITY.*(1+U_C4/c_squ+0.5*(U_C4/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,6)=t3*DENSITY.*(1+U_C6/c_squ+0.5*(U_C6/c_squ).^2-U_SQU/(2*c_squ));
    FEQ(:,:,8)=t3*DENSITY.*(1+U_C8/c_squ+0.5*(U_C8/c_squ).^2-U_SQU/(2*c_squ));
    
    % Compute new particle velocities
    F=omega*FEQ+(1-omega)*F;
    
    % Bounce-back from boundaries
    F(REFLECTED)=BOUNCEDBACK;
    
    % Update time step and check for equilibrium
    prevavu=avu;avu=sum(sum(UX))/numactivenodes; ts=ts+1;
end
figure;colormap(gray(2));image(2-BOUND');hold on;
quiver(2:nx,1:ny,UX(2:nx,:)',UY(2:nx,:)');
title(['Flow field after ',num2str(ts),'\deltat']);xlabel('x');ylabel('y');
