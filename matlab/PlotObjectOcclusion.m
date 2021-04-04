clear all; close all; clc;

%see pytorch/cifar10/cifar_cnn versus cifar_readwrite. jupyter notebook for
%results
OccludeRatio = [0 0.1 0.2 0.4 0.8 1]*100;
NTM_Accuracy = [0.58 0.46 0.33 0.22 0.15 0.1];
CNN_Accuracy = [0.57 0.38 0.28 0.19 0.1 0.1];

hb = figure;
plot(OccludeRatio, NTM_Accuracy, 'r', 'LineWidth',2);
hold on;
plot(OccludeRatio, CNN_Accuracy, 'b', 'LineWidth',2);
plot(OccludeRatio, ones(length(CNN_Accuracy),1)*0.1, 'k--', 'LineWidth',2);

ylim([0 0.6]);
xlim([0 1]*100);
xlabel('Occlusion Ratio (%)','FontSize',14);
ylabel('Top-1 Accuracy','FontSize',14);

legend({'NTM (ours)','CNN'},'FontSize',14);
printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/OcclusionPlot.png'],printmode,printoption);
