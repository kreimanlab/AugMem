clear all; close all; clc;

hb = figure;
EpochNum = 20;
memBound = 10; %range to normalize memory
counter = 1;
for i =  1 %EpochNum %[1:5:EpochNum]
    %load(['../pytorch/cifar/plot/mem_' num2str(i) '.mat']);
    %load(['/home/mengmi/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/visualization/memory.mat']);
    load(['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/cifar/cifar_memory.mat']);
    %subplot(2,2,counter);
    counter = counter + 1;
    image(double(memory),'CDataMapping','scaled');
    colormap('hot');
    caxis manual
    caxis([-memBound memBound]);
    colorbar;
end
%set(hb, 'Position',[675   895   570   194]);
%yticklabels();
xlabel('Memory Content');
%yticks([1 2 3]);
%yticklabels({'obj','context','comb'});
ylabel('Index for Memory Slots');
%title( [xticklabelstring(selectedType,:) '; bin = ' num2str(selectedBin)]);

printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/memory.png'],printmode,printoption);

classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
NumClasses = length(classes);
for selected = [1:NumClasses]
    %load(['../pytorch/cifar/plot/att_' classes{selected} '.mat']);
    %load(['/home/mengmi/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/visualization/att_read_' num2str(selected-1) '.mat']);
    load(['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/cifar/cifar_att_read_' num2str(selected-1) '.mat']);
    att = att_read;
    %subplot(2,2,counter);
    counter = counter + 1;
    image(mat2gray(double(att(1:50,:))),'CDataMapping','scaled');
    colormap('hot');
    caxis manual
    caxis([0 1]);
    colorbar;
    xlabel('Attention');
    ylabel('Example Number');
    title(classes{selected});
    
    printpostfix = '.png';
    printmode = '-dpng'; %-depsc
    printoption = '-r200'; %'-fillpage'
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,['Figure/att_' classes{selected} '.png'],printmode,printoption);
end

