clear all; close all; clc;

% (220*(13*13*64+10)+2000*8)/ (224*224*3 + 10)

%resultrootdir = '/home/mengmi/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/outputs/';
%resultrootdir = '/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/outputs/';
resultrootdir = '/home/mengmi/Desktop/aaai/aaai_results/finalResults_March17/core50_outputs/';
%resultrootdir = '/home/mengmi/HMS/Dropbox/Mengmi/Proj_CL_NTM/pytorch/core50/outputs/';
ExpTypeList = {'iid','class_iid','class_instance','instance'};
modellist = {'AugMem','EWC','GEM','AGEM','iCARL','L2','MAS','NaiveRehearsal','SI','Stable','GSS','BIC','COPE','lwf','NormalNN','NormalNNupper'};
legendname={'HAMN (ours)','EWC','GEM','AGEM','iCARL','L2','MAS','NaiveRehearsal','SI','Stable','GSS','BIC','CoPE','LwF','lowerBound','upperBound'};
colorlist = jet(length(modellist));
FirstAccuThres = 80;

writemode = 'supp'; %choose among 'supp' and 'main' figure in main paper; different output figure size

cond = [];
cond_std = [];

for type =  [2 3] %[1:length(ExpTypeList)]
    typecase = ExpTypeList{type};
    counter = 1;
    temp = [];
    temp_std = [];
    hb = figure; 
    hold on;
    selectedModel = [1:length(modellist)];
    for model = modellist(selectedModel)
        model = model{1};
        if strcmp(model, 'NormalNNupper')
            testresult = readtable([resultrootdir typecase '/NormalNN_ResNet18_offline/' 'test.csv']);
            testresult = testresult{:,:}; 
            
        elseif strcmp(model, 'Stable') || strcmp(model, 'BIC') || strcmp(model, 'GSS') || strcmp(model, 'COPE') || strcmp(model,'lwf')
%             if any(strcmp(gridsearchlist,model))
%                 testresult = readtable([resultrootdir typecase '/' model '_gridsearch/' 'summary.csv']);
%                 testresult = testresult{2,1};
%                 testresult = testresult{1};
%                 testresult = readtable([resultrootdir typecase '/' model '_gridsearch/' testresult '/test.csv']);
            if strcmp(typecase, 'class_iid')
                temptypecase = 'classiid';
            else
                temptypecase = 'classinstance';
            end
            testresult = readtable(['/home/mengmi/Desktop/aaai/Rushikesh_results/' model '_core50_' temptypecase '.csv']);    
            
            if sum(sum(testresult{:,:}>1)) == 0
            %if strcmp(model,'BIC') || strcmp(model,'GSS')
                testresult = testresult{:,:}*100; 
            else
                testresult = testresult{:,:};  
            end
        else
            testresult = readtable([resultrootdir typecase '/' model '_ResNet18/' 'test.csv']);
            testresult = testresult{:,:}; 
%             end
        end
        
        %testresult = testresult(1,:);
        % only evaluate those runs where first task performance is equal
        % for some reasons, GSS and stable cannot even perform well in the
        % first tasks
        if strcmp(model,'Stable') || strcmp(model,'GSS') || strcmp(model,'COPE')  || strcmp(model,'lwf')
            selectedrows = [1:10];
        else
            selectedrows = find(testresult(:,1)>FirstAccuThres);            
        end
        testresult = testresult(selectedrows,:);
        testresult_mean = nanmean(testresult,1);
        temp = [temp nanmean(nanmean(testresult,2))];
        temp_std = [temp_std nanstd(nanmean(testresult,2))/sqrt(length(nanmean(testresult,2)))];
        
        testresult_std = nanstd(testresult,0,1)/sqrt(size(testresult,1));
        %plot([1:size(testresult,2)], testresult_mean, 'Color',colorlist(counter,:),'LineWidth',2);
        errorbar([1:size(testresult,2)], testresult_mean,testresult_std,'Color',colorlist(counter,:),'LineWidth',2)
        counter = counter + 1;
    end
    
    plot([1:5], 0.5./[1:5]*100,'k--','LineWidth',2);
    cond = [cond; temp];
    cond_std = [cond_std; temp_std];
    yticks([0:20:100]);
    xticks([1:size(testresult,2)]);
    ax=gca;
    ax.FontSize = 16;
    xlim([1 size(testresult,2)]);
    ylim([0 100]);
    ylabel('Top-1 Accuracy (%)','FontSize',16,'FontWeight','bold');
    xlabel('Task Number','FontSize',16,'FontWeight','bold');
    %title(['Paradigm: ' typecase],'FontSize',12,'FontWeight','bold');
    if type ~= 3
        if strcmp(writemode, 'main')
            %set(hb,'Position',[1250         112         582         435]);
            set(hb,'Position',[1245         107         444         435]); 
        else
            set(hb,'Position',[1250         112         582         435]);
        end
    else
        
        if strcmp(writemode, 'main')
            legend([legendname(selectedModel) 'Chance'], 'FontSize',12,'FontWeight','bold','Location','southeastoutside');
            %set(hb,'Position',[680   654   788   435]);
            %set(hb,'Position',[1250         112         582         435]);
            set(hb,'Position',[675   649   688   435]);
            legend boxoff 
        else
            set(hb,'Position',[1250         112         582         435]);
        end
    end
    %legend boxoff 
    
    printpostfix = '.pdf';
    printmode = '-dpdf'; %-depsc
    printoption = '-r200'; %'-fillpage'
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    
    if strcmp(writemode, 'main')
        print(hb,['Figure/performance_' typecase '_core50' printpostfix],printmode,printoption);
    else
        print(hb,['Figure/performance_' typecase '_core50_supp' printpostfix],printmode,printoption);
    end
    
end

startcolor = [0.8706    0.9216    0.9804];
overcolor = [0.3922    0.4745    0.6353];
%NumConds = 2-2;
%barcolor = [linspace(startcolor(1),overcolor(1),NumConds)', linspace(startcolor(2),overcolor(2),NumConds)', linspace(startcolor(3),overcolor(3),NumConds)'];
bboxcolor = [1 1 1];
fccolor = [0 0 0];

%barcolor = [bboxcolor; barcolor; fccolor];
barcolor = [bboxcolor; fccolor];


hb = figure('units','pixels');
hold on;
% mturk_mean = subjplot_mean; %all elements in a row belong to the same group; size(A, 1) is the number of groups
% mturk_std = subjplot_std;

mturk_mean = cond';
mturk_std = cond_std';

ngroups = size(mturk_mean, 1);
nbars = size(mturk_mean, 2);
H = bar(mturk_mean);
for b = 1:nbars
    set(H(b),'FaceColor',barcolor(b,:));
    %set(H(b),'EdgeColor',edgecolor(b,:));
    set(H(b),'LineWidth',2);
end

plot([0 ngroups+0.5],[nanmean(0.5./[1:5]*100) nanmean(0.5./[1:5]*100)],'k--','LineWidth',2);

% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, mturk_mean(:,i), mturk_std(:,i), 'k.');
end

xlim([0.5 ngroups+0.5]);
ylim([0 100]);
hold off
%legend(LegName,'Location','Northwest','FontSize', 12);

LegName = {'Class-iid','Class-instance','Chance'}; 
legend(LegName,'Location','Northwest','FontSize', 12);
XtickLabelName = legendname;
xtickangle(90);
%xlabel('Context Object Ratio','FontSize',12);
xticks([1:ngroups]);
set(gca,'YTick',[0:20:100]);
set(gca, 'TickDir', 'out')
%set(gca,'XTickLength',[0 0]);
%set(gca,'XTick',[]);
set(gca,'XTickLabel',XtickLabelName,'FontSize',15);
ylabel('Avg Accuracy','FontSize', 15);
%title('expA What (mturk)','FontSize', 12);
legend('boxoff'); 

%get(hb,'Position')
%set(hb,'Position',[675   833   577   251]);
% printpostfix = '.eps';
% printmode = '-depsc'; %-depsc
% printpostfix = '.eps';
% printmode = '-depsc'; %-depsc
printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/fig_core50_avg_accu' printpostfix],printmode,printoption);

