%% Summery of hyperparameter tuning
% LearnRate = 0.1;
clear all
close all
clc
%--------------------------------------------------------------------------
fileID = fopen('HyperparameterTuniningInfo.txt');
scannedTensor = textscan(fileID,'%s %f %s %f %s %f %s %f %s %f %s %f',...
    'Delimiter',{'=',','});
fclose(fileID);

varName = {scannedTensor{3}{1}(1:end-1),...
    scannedTensor{5}{1}(1:end-1),...
    scannedTensor{7}{1}(1:end-1),...
    scannedTensor{9}{1}(1:end-1),...
    scannedTensor{11}{1}(~isspace(scannedTensor{11}{1})),...
    };
disp('Variables:')
disp(varName);

validScore2L = table(cell2mat(scannedTensor(4)),...
    cell2mat(scannedTensor(6)),...
    cell2mat(scannedTensor(8)),...
    cell2mat(scannedTensor(10)),...
    cell2mat(scannedTensor(12)),...
    'VariableNames',varName);

numHL1Range = unique(validScore2L.numHL1);
numHL2Range = unique(validScore2L.numHL2);
l2Factor1Range = unique(validScore2L.l2FactorHL1);
l2Factor2Range = unique(validScore2L.l2FactorHL2);

%% Highest score
[~,ind] = max(validScore2L.Testaccuracy);
disp('Optimal Hyper-parameter configuration:')
disp(validScore2L(ind,:));

%% Without L2-regularization
noL2ind=((validScore2L.l2FactorHL1 == 0)&(validScore2L.l2FactorHL2 == 0));
NoL2Score = validScore2L(noL2ind,:);

figure('Position',[40,80,1200,800])
% colormap(jet)
% surf(log2(reshape(NoL2Score.numHL1,[6,5])),...
%     log2(reshape(NoL2Score.numHL2,[6,5])),...
%     100*reshape(NoL2Score.Testaccuracy,[6,5]))
% xticks(log2(numHL1Range))
% xticklabels(numHL1Range)
% xlabel('Size of hidden layer 1')
% yticks(log2(numHL2Range))
% yticklabels(numHL2Range)
% ylabel('Size of hidden layer 2')
% zlabel('Test accuracy (%)')

bh = bar3(100*reshape(NoL2Score.Testaccuracy,[6,5]));
for k = 1:length(bh)
    zdata = bh(k).ZData;
    bh(k).CData = zdata;
    bh(k).FaceColor = 'interp';
end
xticks(log2(numHL1Range)-3)
xticklabels(numHL1Range)
xlabel('Size of hidden layer 1')
yticks(log2(numHL2Range)-3)
yticklabels(numHL2Range)
ylabel('Size of hidden layer 2')
zlim(100*[min(NoL2Score.Testaccuracy),max(NoL2Score.Testaccuracy)+0.01])
caxis(100*[min(NoL2Score.Testaccuracy),max(NoL2Score.Testaccuracy)])
zlabel('Validation accuracy (%)')

[~,ind] = max(NoL2Score.Testaccuracy);
optiConfig = NoL2Score(ind,:);
disp('Optimal Hyper-parameter configuration (without L2):')
disp(optiConfig);
hold on
scatter3(log2(optiConfig.numHL1)-3,log2(optiConfig.numHL2)-3,...
    100*optiConfig.Testaccuracy+1,128,'r','filled')
text(log2(optiConfig.numHL1)-3,log2(optiConfig.numHL2)-3,...
    100*optiConfig.Testaccuracy+4,...
    sprintf('Highest accuracy = %.1f%%',100*optiConfig.Testaccuracy),...
    'FontSize',18)
hold off
set(gca,'FontSize',18)
view([73,55])
title('Tuning hidden layer size (without L2-regularization)')

%% Hidden layer 1 and 2 both with size of 64
HL64ind=((validScore2L.numHL1 == 64)&(validScore2L.numHL2 == 64));
HL64Score = validScore2L(HL64ind,:);

figure('Position',[40,80,1200,800])
bh = bar3(100*reshape(HL64Score.Testaccuracy,[5,5]));
for k = 1:length(bh)
    zdata = bh(k).ZData;
    bh(k).CData = zdata;
    bh(k).FaceColor = 'interp';
end
xticklabels(l2Factor1Range)
xlabel('Hidden layer 1 L2-reg. ratio')
yticklabels(l2Factor1Range)
ylabel('Hidden layer 2 L2-reg. ratio')
zlim(100*[min(HL64Score.Testaccuracy),max(HL64Score.Testaccuracy)+0.01])
caxis(100*[min(HL64Score.Testaccuracy),max(HL64Score.Testaccuracy)])
zlabel('Validation accuracy (%)')

[~,ind] = max(HL64Score.Testaccuracy);
optiConfig = HL64Score(ind,:);
xInd = find(l2Factor1Range == optiConfig.l2FactorHL1);
yInd = find(l2Factor2Range == optiConfig.l2FactorHL2);
disp('Optimal L2-regularization configuration:')
disp(optiConfig);
hold on
scatter3(xInd,yInd,...
    100*optiConfig.Testaccuracy+0.5,128,'r','filled')
text(xInd,yInd,...
    100*optiConfig.Testaccuracy+4,...
    sprintf('Highest accuracy = %.1f%%',100*optiConfig.Testaccuracy),...
    'FontSize',18)
hold off
set(gca,'FontSize',18)
view([57,67])
title('Tuning L2-regularization ratio (Size of both hidden layers = 64)')
