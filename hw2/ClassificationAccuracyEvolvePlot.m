%% Classification Accuracy Evolve
% LearnRate = 0.1;
clear all
close all
clc
%--------------------------------------------------------------------------
fileID = fopen('ClassificationAccuracyEvolve.txt');
scannedTensor = textscan(fileID,'%s %s %f %s %f',...
    'Delimiter','=');
fclose(fileID);

TrainAccuracy = scannedTensor{1,3};
ValidationAccuracy = scannedTensor{1,5};

figure('Position',[40,80,1200,800])
plot(100*TrainAccuracy,'.-')
hold on
plot(100*ValidationAccuracy,'.-')
% plot(10:10:50,100*TrainAccuracy(10:10:50),'--')
% plot(10:10:50,100*ValidationAccuracy(10:10:50),'--')
hold off
xlabel('Step')
ylabel('Validation accuracy (%)')
legend('Training accuracy','Validation accuracy','Location','Southeast')
box off;
set(gca,'FontSize',18)
title('How the classification accuracy in the training and validation sets evolved')