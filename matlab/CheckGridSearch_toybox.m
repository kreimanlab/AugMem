clear all; close all; clc;

Folder   = '/home/mengmi/Desktop/ntm-replay/toybox_gridsearch_outputs/';
FileList = dir(fullfile(Folder, '**', 'test.csv'));

colorlist = jet(15);

settinglist = {'class_instance','class_iid'};
modellist = {'EWC','GEM','iCARL','L2','MAS',...
    'SI','NaiveRehearsal','NormalNNlower','NormalNNupper'};

modelid = 1;
settingid = 1;

hb = figure; hold on;
legendlist = {};
subjmean = [];

for i = [1:length(FileList)]
    
    if contains(FileList(i).folder, settinglist{settingid}) ...
            && contains(FileList(i).folder, modellist{modelid})
        
        testresult = readtable([FileList(i).folder '/test.csv']);
        testresult = testresult{:,:};
        subjmean = [subjmean; nanmean(testresult,1)];
        
        k = strfind(FileList(i).folder,modellist{modelid});
        legendtemp = FileList(i).folder;
        k = k +length(modellist{modelid})+1; 
        legendtemp = legendtemp(k:end);
        legendlist = [legendlist; legendtemp];
    end   
    
end

title([settinglist{settingid} '; ' modellist{modelid}]);
for i = 1:size(subjmean,1)    
    plot([1:6],subjmean(i,:),'Color',colorlist(i,:));
end    
legend(legendlist);  

[val ind] = max(subjmean(:,6));
legendlist{ind}
subjmean
legendlist'