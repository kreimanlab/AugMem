clear all; close all; clc;

%resultrootdir = '/home/mengmi/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/outputs/';
resultrootdir = '/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/outputs_';
%resultrootdir = '/home/mengmi/HMS/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/outputs/';
ExpTypeList = {'iid','class_iid','class_instance','instance'};
modellist = {'original','memslot50','memsz100','Sparse','NoLogits','NoReg','NoReplay'};
legendname={'AugMem (ours)','NumMemSlot','NumIndexReplays','MemSparseness','DistillationLoss','MemUsageLoss','ReplayLoss'};
colorlist = jet(length(modellist));
FirstAccuThres = 80;

for type =  [3] %[1:length(ExpTypeList)]
    display('start');
    lastval = [];
    lastval_std = [];
    typecase = ExpTypeList{type};
    counter = 1;
    hb = figure; 
    hold on;
    selectedModel = [1:length(modellist)];
    for model = modellist(selectedModel)
        model = model{1};
        if strcmp(model, 'original')
            testresult = readtable([resultrootdir(1:end-1) '/' typecase '/AugMem_ResNet18/' 'test.csv']);
        else
%             if any(strcmp(gridsearchlist,model))
%                 testresult = readtable([resultrootdir typecase '/' model '_gridsearch/' 'summary.csv']);
%                 testresult = testresult{2,1};
%                 testresult = testresult{1};
%                 testresult = readtable([resultrootdir typecase '/' model '_gridsearch/' testresult '/test.csv']);
%                 
%             else
                
                testresult = readtable([resultrootdir model '/' typecase '/AugMem_ResNet18/' 'test.csv']);
%             end
        end
                
        testresult = testresult{:,:};         
        %testresult = testresult(1,:);
        % only evaluate those runs where first task performance is equal
        selectedrows = find(testresult(:,1)>70);
        testresult = testresult(selectedrows,:);
        testresult_mean = nanmean(testresult,1);
        lastval = [lastval testresult_mean(end)];
        lastval_std = [lastval_std nanstd(testresult(:,end))/sqrt(length(testresult(:,end)))];
        testresult_std = nanstd(testresult,0,1)/sqrt(size(testresult,1));
        %plot([1:size(testresult,2)], testresult_mean, 'Color',colorlist(counter,:),'LineWidth',2);
        errorbar([1:size(testresult,2)], testresult_mean,testresult_std,'Color',colorlist(counter,:),'LineWidth',2)
        counter = counter + 1;
    end
    
    lastval
    lastval_std
    
    plot([1:5], 0.5./[1:5]*100,'k--','LineWidth',2);
    
    yticks([0:20:100]);
    xticks([1:size(testresult,2)]);
    xlim([1 size(testresult,2)]);
    ylim([0 100]);
    ylabel('Top-1 Accuracy (%)','FontSize',12,'FontWeight','bold');
    xlabel('Task Number','FontSize',12,'FontWeight','bold');
    title(['Paradigm: ' typecase],'FontSize',12,'FontWeight','bold');
    legend([legendname(selectedModel) 'Chance'], 'FontSize',12,'FontWeight','bold','Location','southeastoutside');
    set(hb,'Position',[680   654   788   435]);
    
    hb = figure; hold on;
    ngroups = 1;
    nbars = size(lastval, 2);
    NumTypes = nbars;
    NumVisualBin = 1;
    xaxis = [1:NumTypes]; 

    xticklabelstring =str2mat(legendname); %,'Material'
    %legendstring = {'[0.5 1]','[1.75 2.25]','[3.5 4.5]','[7 9]'};
    H = bar(lastval);
    barcolor = [1     1     1;...
    0.8314    0.8157    0.7843;...
    0.5020    0.5020    0.5020;...
    0     0     0];

    plot([0.5 2:nbars-1 nbars+0.5],lastval(1)*ones(1,nbars),'k:');
    plot([0.5 2:nbars-1 nbars+0.5],10*ones(1,nbars),'k--');
    
    for b = 1:NumVisualBin
        set(H(b),'FaceColor',barcolor(3,:));
    end
    xlim([0.5 NumTypes+0.5]);
    ylim([0 0.5]*100);
    for x = 1:NumTypes
        errorbar(x, lastval(x), lastval_std(x), 'k.','HandleVisibility','off');
    end
    plot([0.5 2:nbars-1 nbars+0.5],lastval(1)*ones(1,nbars),'k:');
    plot([0.5 2:nbars-1 nbars+0.5],10*ones(1,nbars),'k--');
    hold off
    set(gca,'YTick',[0:0.1:0.5]*100);
    set(gca,'XTick',(xaxis));
    xtickangle(45);
    set(gca,'TickLength',[0 0]);
    set(gca,'XTickLabel',xticklabelstring);
    ylabel('Top-1 Accuracy (last task)','FontSize', 12);
    legend('AugMem (ours)','chance');
    
    printpostfix = '.png';
    printmode = '-dpng'; %-depsc
    printoption = '-r200'; %'-fillpage'
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,['Figure/ablated_task_'  typecase printpostfix],printmode,printoption);
    
end