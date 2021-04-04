clear all; close all; clc;
addpath('tsne');
printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = './Figure/';
printflag = 1;
labelsize=12;

ExpTypeList = {'iid','class_iid','class_instance','instance'};
Type = 3;
Run = 1;

tasklist = [1 3 5];
no_dims = 2;
initial_dims = 10;
classlist=[2:2:10];
nsamples = 50;
counter = 1;
classes = {'scissor', 'cup', 'remote', 'can', 'cellphone', 'ball', 'marker', 'lightbulb', 'glass','plug'};

% colorcode = {[1 0.2 0.2], ... %red
%     [0.6350, 0.0780, 0.1840], ... %date    
%     [0.4660, 0.6740, 0.1880], ... %green
%     [0, 0.75, 0.75], ... %cyan
%     [1.0000    0.0745    0.6510], ... %magenta
%     [0.75, 0.75, 0], ... %dark yellow
%     [0.8500, 0.3250, 0.0980], ... %orange
%     [0.9290, 0.6940, 0.1250], ... %yellow
%     [0.4940, 0.1840, 0.5560], ... %purple    
%     [0, 0.5, 0], ... %dark green
%     [0.3010, 0.7450, 0.9330], ... %light blue    
%     [0.25, 0.25, 0.25]}; %dark grey
%colorcode = colorcode(1:10);
       
% colorcode = {[0.8500, 0.3250, 0.0980], ... %orange
%     [1 0.2 0.2], ... %red    
%     [0.9290, 0.6940, 0.1250], ... %yellow
%     [0.4940, 0.1840, 0.5560], ... %purple
%     [0.4660, 0.6740, 0.1880], ... %green
%     [0.3010, 0.7450, 0.9330], ... %light blue
%     [0.6350, 0.0780, 0.1840], ... %date
%     [0, 0.5, 0], ... %dark green
%     [0, 0.75, 0.75], ... %cyan
%     [0.75, 0, 0.75], ... %magenta
%     [0, 0.4470, 0.7410], ... %blue
%     [0.75, 0.75, 0], ... %dark yellow
%     [0.25, 0.25, 0.25]}; %dark grey 


% colorcode = cell2mat(colorcode);
% colorcode = reshape(colorcode, length(colorcode)/3,3);
colorcode = [1.0000    0.0745    0.6510;...
    0    0.4471    0.7412;...
0.9294    0.6941    0.1255;...
0.4941    0.1843    0.5569;...
0.4667    0.6745    0.1882;...
 0.0588    1.0000    1.0000;...
 1.0000    1.0000    0.0667;...
 0.7882    0.7882    0.7882;...
 0    0    0;...
 0     1     0];

    
colorcode = colorcode(1:10,:);

hb = figure;
for i = tasklist
    
    train_X = [];
    Task = i;
    
    if Task == 1
        Epoch = 3;
    else
        Epoch = 1;
    end
    
    for selected = [1:classlist(i)]
        matfilename = ['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/'...
            ExpTypeList{Type} '/' ExpTypeList{Type} ...
        '_run_' num2str(Run-1) '_task_' num2str(Task-1) '_epoch_' num2str(Epoch-1) ...
        '_att_read_' num2str(selected-1) '.mat'];

        if exist(matfilename, 'file') ~= 2
            continue;
        end
        load(matfilename);
        att_read = reshape(direct, [nsamples,size(direct,2)]);
        train_X = [train_X; att_read];
    end
    %train_X = reshape(represents,classlist(i)*nsamples,initial_dims);
    
    labels = [1:classlist(i)]';
    labels = repmat(labels,1,nsamples)';
    train_labels = reshape(labels, classlist(i)*nsamples,1);
    
    train_X = double(train_X);
    % Run t−SNE
    %mappedX = tsne(train_X,train_labels,no_dims );
    mappedX = tsne(train_X,[],no_dims );
    szX = size(mappedX,1);
    
    % Plot results
    subplot(1,3,counter);
    
    if counter < 3
        gscatter(mappedX(szX:-1:1,1), mappedX(szX:-1:1,2), train_labels(szX:-1:1),colorcode(classlist(i):-1:1,:),'.',15,'off');
    else
        gscatter(mappedX(szX:-1:1,1), mappedX(szX:-1:1,2), train_labels(szX:-1:1),colorcode(classlist(i):-1:1,:),'.',15);
        legend(classes(10:-1:1),'Location','northeast');
    end
    %scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3));
    %
    title(['Task = ' num2str(i)],'FontSize',labelsize);
    counter = counter + 1;
end

if printflag == 1
    printfilename = 'visModelCluster';
    set(hb,'Position',[569         674        1312         290]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end