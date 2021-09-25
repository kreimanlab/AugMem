clear all; close all; clc;

resultrootdir = '/home/mengmi/Desktop/aaai/aaai_results/outputs';
ExpTypeList = {'iid','class_iid','class_instance','instance'};
modellist = {'AugMem_MobileNetV2', 'AugMem_ResNet18_PretrainMemKmeans','AugMem_ResNet18_herding'};
legendname={'AugMemMobileNetV2','AugMemPretrainMemKmeans','AugMem_herding'};
colorlist = jet(length(modellist));
FirstAccuThres = 80;

for type =  [2] %[1:length(ExpTypeList)]
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
        testresult = readtable([resultrootdir(1:end) '/' typecase '/' model '/' 'test.csv']);  
                
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
    
    
    
end