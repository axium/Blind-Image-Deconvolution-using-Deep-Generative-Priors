function PSFS = createPSFs(TrajCurve , PSFsize , T , do_show , do_center)
%
% function PSFS = createPSFs(TrajCurve , PSFsize , T , do_show , do_center)
%
% PSFs are obtained by sampling the continuous trajectory TrajCurve on a regular pixel grid using linear interpolation
% at subpixel level
%
% input description
% TrajCurve                 Motion Blur trajectory cuve, provided by createTrajectory function
% PSFsize                     Size of the PFS where the TrajCurve is sampled
% T                                Vector of exposure times: for each of them a PSF will be generated
% do_show
% do_center
%
% output description
% PSFS                       cell array containing PSFS sampling TrajCurve for each exposure time  in T
%                                   numel(PSFS) = length(T).
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
% Revision History
% March 2009         - beta release (not available online)
% July  2011            - beta release made available on http://home.dei.polimi.it/boracchi/software
% December 2012  - first official release
% January 2014  - minor fix to allow trajectories to extend outside of the PSF support
%
% Giacomo Boracchi*, Alessandro Foi**
% giacomo.boracchi@polimi.it
% alessandro.foi@tut.fi
% * Politecnico di Milano
% **Tampere University of Technology

if ~exist('TrajCurve','var')||isempty(TrajCurve)
    TrajCurve = createTrajectory();
end

if  ~exist('PSFsize','var')||isempty(PSFsize)
    PSFsize = 64;
end

if numel(PSFsize)==1
    PSFsize=[PSFsize PSFsize];
end

if ~exist('T','var')||isempty(T)
    T = [1/1000 1/10 1/2 1];
end

if ~exist('do_show','var')||isempty(do_show)
    do_show = 0;
end

if ~exist('do_center','var')||isempty(do_center)
    do_center = 0;
end

PSFnumber = length(T);
numt = length(TrajCurve.x);   % number of samples of motion trajectory
x = TrajCurve.x;

if do_center
    % center with respect to baricenter
    x=x-mean(x)+(PSFsize(2)+1i*PSFsize(1)+1+1i)/2;
end


% % uncomment to restrict trajectory to support size
% x=max(1,min(PSFsize(2),real(x)))+1i*max(1,min(PSFsize(1),imag(x)));


%% generate le PSFS
PSFS = cell(1 , PSFnumber);

PSF = zeros(PSFsize);

triangle_fun=@(d) max(0,(1-abs(d)));
triangle_fun_prod=@(d1,d2) triangle_fun(d1) .* triangle_fun(d2);

        
% set the exposure time
for jj = 1:numel(T)
    if jj == 1
        prevT = 0;
    else
        prevT = T(jj - 1);
    end
    % sample the trajectory until time T
    for t = 1 : numel(x); %floor(prevT):ceil(T(jj))
        if (T(jj) * numt >= t) && (prevT * numt < t - 1);
            t_proportion = 1;
        elseif (T(jj) * numt >= t - 1) && (prevT * numt < t - 1);
            t_proportion = (T(jj) * numt)-(t - 1);
        elseif  (T(jj) * numt >= t) && (prevT * numt < t);
            t_proportion = t -(prevT * numt);
        elseif  (T(jj) * numt >= t - 1)&&(prevT * numt < t);
            t_proportion = (T(jj) - prevT) * numt;
        else
            t_proportion = 0;
        end
        
        m2 = min(PSFsize(2)-1,max(1,floor(real(x(t)))));
        M2 = m2+1;
        m1 = min(PSFsize(1)-1,max(1,floor(imag(x(t)))));
        M1 = m1+1;
        
        
        %% linear interp. (separable)
        
        PSF(m1 , m2) = PSF(m1 , m2)  +  t_proportion * triangle_fun_prod( real(x(t)) - m2 , imag(x(t)) - m1 );
        PSF(m1 , M2) = PSF(m1 , M2)  +  t_proportion * triangle_fun_prod( real(x(t)) - M2 , imag(x(t)) - m1 );
        PSF(M1 , m2) = PSF(M1 , m2)  +  t_proportion * triangle_fun_prod( real(x(t)) - m2 , imag(x(t)) - M1 );
        PSF(M1 , M2) = PSF(M1 , M2)  +  t_proportion * triangle_fun_prod( real(x(t)) - M2 , imag(x(t)) - M1 );
        
        
    end
    
    PSFS{jj} = PSF/numel(x);
    
end


%% show results
if do_show
    C = [];  D = [];
    for jj = 1:numel(T)
        
        C = [C;PSFS{jj}];
        D = [D;PSFS{jj}/max(PSFS{jj}(:))];
        
    end
    figure(456)
    subplot(1 , 2 , 1)
    imshow(C , []);  title('all PSF normalized w.r.t. the same maximum')
    subplot(1 , 2 , 2)
    imshow(D) ,  title('each PSFs normalized w.r.t. its own maximum')
    colormap(hot)
end

