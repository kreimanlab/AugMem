clear all; close all; clc;

ExpTypeList = {'iid','class_iid','class_instance','instance'};
Type = 3;
Run = 1;
Epoch= 3;
TaskTotal = 5;
NumMemSlots = 100;

hb = figure;

memBound = 10; %range to normalize memory
counter = 1;
for Task =  [1:TaskTotal] 
    %load(['../pytorch/cifar/plot/mem_' num2str(i) '.mat']);
    %load(['/home/mengmi/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/visualization/memory.mat']);
    load(['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/' ExpTypeList{Type} '/' ExpTypeList{Type} ...
        '_run_' num2str(Run-1) '_task_' num2str(Task-1) '_epoch_' num2str(Epoch-1) '_memory.mat']);
        
    
    subplot(2,3,counter);
    counter = counter + 1;
    image(double(memory),'CDataMapping','scaled');
    colormap('hot');
    caxis manual
    title(['task = ' num2str(Task)]);
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

set(hb, 'Position',[675           2        1029        1087]);
printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/memory.png'],printmode,printoption);

NumClasses = 10;
hb = figure;
for Task = [1:TaskTotal]
    %selectedfiles = dir();
    
    for selected = [1:2] % NumClasses]
        matfilename = ['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/' ExpTypeList{Type} '/' ExpTypeList{Type} ...
        '_run_' num2str(Run-1) '_task_' num2str(Task-1) '_epoch_' num2str(Epoch-1) '_att_read_' num2str(selected-1) '.mat'];
    
        if exist(matfilename, 'file') ~= 2
            continue;
        end
        load(matfilename);

        att = att_read(:,1:8);
        unique(att)'
        uniquememslots = cellfun(@(c) unique(c), num2cell(att, 2), 'UniformOutput', false);
        mat = zeros(size(att,1), NumMemSlots);
        for p = 1:length(uniquememslots)
            mat(p, uniquememslots{p}) = 1;
        end
        att = mat;
        
        subplot(TaskTotal,NumClasses, (Task-1)*NumClasses + selected);
        
        image(mat2gray(double(att)),'CDataMapping','scaled');
        colormap('hot');
        caxis manual
        caxis([0 1]);
    end

end

% xlabel('Attention');
% ylabel('Example Number');
%title(['class = ' num2str(selected)]);
%colorbar;

printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/att_' ExpTypeList{Type} '.png'],printmode,printoption);

