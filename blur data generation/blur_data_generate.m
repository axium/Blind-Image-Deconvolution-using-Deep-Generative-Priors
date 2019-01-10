% Generates Motion Using Code From Paper
% [Boracchi and Foi, 2012] Modeling the Performance of Image Restoration from Motion Blur
% link: http://home.deib.polimi.it/boracchi/Projects/PSFGeneration.html
% Run this file in matlab to generate motion blurs similar to the paper

clc;clear all;close all;

TrainingSet =   60000;
ValidationSet = 20000;
BlurSize = 28;
MaxBlurLength = 28;
MinBlurLength = 5;
%% Training Set
X_Train = {};
BlurLength = randi([MinBlurLength,MaxBlurLength], TrainingSet,1);
for i = 1:TrainingSet
    traj = createTrajectory(BlurSize, 0.01, 100, BlurLength(i),0);
    psf = createPSFs(traj, BlurSize, 1, 0,1);
    X_Train{i} = psf{1};
end
%% Validation Set
X_Test = {};
BlurLength = randi([MinBlurLength,MaxBlurLength], ValidationSet,1);
for i = 1:ValidationSet
    traj = createTrajectory(BlurSize, 0.01, 100, BlurLength(i),0);
    psf = createPSFs(traj, BlurSize, 1, 0,1);
    X_Test{i} = psf{1};
end
%% Saving Blurs
save('blur_data.mat', 'X_Train', 'X_Test')
%% Showing 100 Samples from the DataSet 
for i = 1:100
    t = randi([1,ValidationSet]);
    subplot(10,10,i)
    imshow(X_Test{t}/max(max(X_Test{t})))
end
