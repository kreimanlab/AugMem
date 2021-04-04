clear all; close all; clc;

ExpTypeList = {'iid','class_iid','class_instance','instance'};
Type = 3;
Run = 1;

tasklist = [1 3 5];
classlist=[2:2:10];
classes = {'scissor', 'cup', 'remote', 'can', 'cellphone', 'ball', 'marker', 'lightbulb', 'glass','plug'};

for i = tasklist    
    
    Task = i;
    
    if Task == 1
        Epoch = 3;
    else
        Epoch = 1;
    end
    
    
    matfilename = ['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/'...
        ExpTypeList{Type} '/' ExpTypeList{Type} ...
    '_run_' num2str(Run-1) '_task_' num2str(Task-1) '_epoch_' num2str(Epoch-1) ...
    '_confusemat.mat'];

    if exist(matfilename, 'file') ~= 2
        continue;
    end
    load(matfilename);
    ConfusionMat = confusemat;    
    for selected = [1:classlist(i)]
        c = selected;
        sumr = sum(ConfusionMat(c,:));
        ConfusionMat(c,:) = ConfusionMat(c,:)/sumr;
    end
    
    confmat = ConfusionMat; % sample data
    % plotting
    %plotConfMat(confmat, LabelList);

    hb = figure;
    numlabelsX = classlist(Task)+1;
    numlabelsY = classlist(Task);

    LabelListX = classes(1:classlist(Task));
    LabelList = LabelListX;
    
    imagesc(confmat);
    colormap(jet);
    caxis([0 1]);
    xlabel('Predicted Labels','FontSize',16','FontWeight','Bold');
    ylabel('Actual Labels','FontSize',16','FontWeight','Bold');
    hc=colorbar();
    set(hc,'YTick',[0:0.2:1]);

    set(gca,'XTick',1:numlabelsX,...
        'XTickLabel',LabelListX,...
        'YTick',1:numlabelsY,...
        'YTickLabel',LabelList,...
        'FontSize',16','FontWeight','Bold');

    %xticklabel_rotate;
    set(gca,'XTickLabelRotation',45)

    printpostfix = '.pdf';
    printmode = '-dpdf'; %-depsc
    printoption = '-r200'; %'-fillpage'

    set(hb,'Position',[1361           7        1246        1082]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,['Figure/Confusionmat_task_' num2str(Task) printpostfix],printmode,printoption);
    
    
    
end




