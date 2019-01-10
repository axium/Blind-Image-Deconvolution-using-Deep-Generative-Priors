function TrajCurve = createTrajectory(TrajSize, anxiety, numT, MaxTotalLength, do_show)
%
% function TrajCurve = createTrajectory(TrajSize, anxiety, numT, MaxTotalLength, do_show)
%
% Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012].
% Each trajectory consists of a complex-valued vector determining the discrete positions of 
%  a particle following a 2-D random motion in continuous domain. 
% The particle has an initial velocity vector which, at each iteration, is affected by a Gaussian perturbation and
% by a deterministic inertial component, directed toward the previous particle position. In addition, 
% with a small probability, an impulsive (abrupt) perturbation aiming at inverting the particle velocity may arises,
% mimicking a sudden movement that occurs when the user presses the camera button or tries to compensate the camera shake.
% At each step, the velocity is normalized to guarantee that trajectories corresponding to equal exposures have the same length. 
% Each perturbation (Gaussian, inertial, and impulsive) is ruled by its own parameter. 
% Rectilinear Blur as in [Boracchi and Foi 2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs)
%
% input description
% TrajSize                   size (in pixels) of the square support of the Trajectory curve
% anxiety                    parameter determining the amount of shake (in the range [0,1] 0 corresponds to rectilinear
%                                 trajectories). This term scales the random vector that is added at each sample.
% numT                      number of samples where the Trajectory is sampled
% MaxTotalLength    maximum length of the trajectory computed as the sum of all distanced between consecuive points
% do_show                show a figure illustrating the trajectory
%
% output description
% TrajCurve                                  ouput structure having the following fields
%   TrajCurve.x                             complex-valued vector. Each point determines a position in the complex plane to be considered as
%                                                           the image plane
% TrajCurve.TotLenght                length of the TrajCurve (measured as the sum of absolute distance between consecutive points )
% TrajCurve.Anxiety                     input parameter
% TrajCurve.MaxTotalLength      input parameter
% TrajCurve.nAbruptShakes       number of abrupt shakes occurred here
%
% Usage
% TrajCurve = createTrajectory() use default parameters
% TrajCurve = createTrajectory(TrajSize, anxiety, numT, MaxTotalLength, do_show) set all the parameters
% 
% it is possible to plot the  motion trajectory using plot(TrajCurve.x) 
% motion blur PSF can be obtained by sampling these trajectories using createPSFs function
%
%
% References
% [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
%  Image Processing, IEEE Transactions on. vol.21, no.8, pp. 3502 - 3517, Aug. 2012, doi:10.1109/TIP.2012.2192126 
% Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
%
% [Boracchi and Foi 2011] Giacomo Boracchi and Alessandro Foi, "Uniform motion blur in Poissonian noise: blur/noise trade-off"
%  Image Processing, IEEE Transactions on. vol. 20, no. 2, pp. 592-598, Feb. 2011 doi: 10.1109/TIP.2010.2062196 
% Preprint Available at http://home.dei.polimi.it/boracchi/publications.html
% 
%
% Revision History
% March 2009         - beta release (not available online)
% July  2011            - beta release made available on http://home.dei.polimi.it/boracchi/software
% December 2012  - first official release
%
% Giacomo Boracchi*, Alessandro Foi**
% giacomo.boracchi@polimi.it
% alessandro.foi@tut.fi
% * Politecnico di Milano
% **Tampere University of Technology

if nargin<1
    TrajSize=64;
end

if nargin<2
    anxiety=0.1*rand(1);
end

if nargin<3
    numT=2000;
end

if nargin<4
    MaxTotalLength = 60;
end

if nargin<4
    do_show = 1;
end

do_compute_curvature = 0;

TotCurvature = 0;
TotLength = 0;
abruptShakesCounter = 0;

% term determining, at each sample, the strengh of the component leating towards the previous position
centripetal = 0.7 * rand(1);

% term determining, at each sample, the random component of the new direction
gaussianTerm =10 * rand(1);

% probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements 
freqBigShakes = 0.2 * rand(1);

%% Generate x(t), Discrete Random Motion Trajectory  in Continuous Domain 

% v is the initial velocity vector, initialized at random direction
init_angle = 360*rand(1);
% initial velocity vector having norm 1
v0 = cosd(init_angle) + i * sind(init_angle);
% the speed of the initial velocity vector
v = v0* MaxTotalLength/(numT-1);

if anxiety>0
    v = v0 * anxiety;
end

% initialize the trajectory vector
x = zeros(numT , 1);

for t = 1 : numT - 1
   % determine if there is an abrupt (impulsive) shake
    if rand < freqBigShakes * anxiety
        % if yes, determine the next direction which is likely to be opposite to the previous one
        nextDirection = 2 * v * (exp( i * (pi + (rand(1) - 0.5))));
        abruptShakesCounter = abruptShakesCounter + 1;
    else
        nextDirection=0;
    end
    
    % determine the random component motion vector at the next step
    dv = nextDirection + anxiety * (gaussianTerm * (randn + i * randn) - centripetal * x(t)) * (MaxTotalLength / (numT - 1));
    v=v+dv;

    % velocity vector normalization 
    v = (v / abs(v)) * MaxTotalLength / (numT - 1);

    % update particle position
    x(t + 1) = x(t) + v;

    % compute total length
    TotLength=TotLength+abs(x(t+1)-x(t));

    if do_compute_curvature
        % compute total curvature
        if t>1
            TotCurvature = TotCurvature+ ComputeCurvature([real(x(t+1)),imag(x(t+1))],[real(x(t)),imag(x(t))],[real(x(t-1)),imag(x(t-1))]);
        end
    end


    %     close figure 2
    %     figure(2),plot(x(1:t)),title(num2str(t));
    %     hold on
    %     plot(x(t),'r*')
    %     pause(0.001)

end

%% compute Trajectory Statistics
if do_compute_curvature
    window_size=1+2*ceil(MaxTotalLength/(numT-1)/2);
    xrsmooth=conv2([repmat(x(1),[(window_size-1)/2 1]);x;repmat(x(end),[(window_size-1)/2 1])],ones(window_size,1)/window_size,'valid');

    TotCurvatureSmooth=0;
    CumultiveCurvatureSmooth=[];

    for t=2:floor(numel(xrsmooth)/window_size-1);
        TotCurvatureSmooth = TotCurvatureSmooth+ ComputeCurvature([real(xrsmooth(window_size*(t+1))),imag(xrsmooth(window_size*(t+1)))],[real(xrsmooth(window_size*(t))),imag(xrsmooth(window_size*(t)))],[real(xrsmooth(window_size*(t-1))),imag(xrsmooth(window_size*(t-1)))]);
        CumultiveCurvatureSmooth(t)=TotCurvatureSmooth;
    end

    TotCurvatureFitted=0;
    nFittingSamples=10; % samples a dx e a sx da fittare
    CumulativeCurvatureFitted=[];

    %calcola la curvatura con il fitting della parabola su  2*nFittingSamples+1
    %samples, utilizzando il suggerimento degli israeliti
    for t=nFittingSamples + 1 :numT-nFittingSamples;
        pts=[real(x(t-nFittingSamples:t+nFittingSamples))';imag(x(t-nFittingSamples: t+nFittingSamples))'];
        TotCurvatureFitted = TotCurvatureFitted+ ComputeCurvatureFitting(pts);
        CumulativeCurvatureFitted(t)=TotCurvatureFitted;
    end
end

%% Center the Trajectory 

% Set the lowest position in zero 
x = x - 1i * min(imag(x))-min(real(x));

% Center the Trajectory
x = x - 1i * rem(imag(x(1)), 1) - rem(real(x(1)), 1) + 1 + 1i;
x = x + 1i * ceil((TrajSize - max(imag(x))) / 2) + ceil((TrajSize - max(real(x))) / 2);

if do_show
    figure(455),
    %plot(real(x),1+max(imag(x))-imag(x)),
    plot(x),hold on
    plot(x(1),'rx');
    plot(x(end),'ro');
    axis([0 TrajSize 0 TrajSize])
    legend('Traj Curve', 'init' , 'end');
    %         square tight
    hold off
    title(['anxiety:' , num2str(anxiety) , ' number of abrupt shakes: ', num2str(abruptShakesCounter)]);
    drawnow
    pause(eps)
end

% build structure
TrajCurve.x=x;
TrajCurve.TotLenght = TotLength;
TrajCurve.TotCurvature = TotCurvature;
TrajCurve.Anxiety = anxiety;
TrajCurve.MaxTotalLength = MaxTotalLength;
TrajCurve.nAbruptShakes = abruptShakesCounter;

if do_compute_curvature
    TrajCurve.TotCurvatureSmooth=TotCurvatureSmooth/numT;
    TrajCurve.CumultiveCurvatureSmooth=CumultiveCurvatureSmooth/numT;
    TrajCurve.TotCurvatureFitted=TotCurvatureFitted/numT;
    TrajCurve.CumultiveCurvatureFitted=CumulativeCurvatureFitted/numT;
end

