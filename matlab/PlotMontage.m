clear all; close all; clc;

imgsz = 128;
numClass = 2;
Hypo = 34;
Task = 1; %choose among 1,3,5
load(['Mat/montage_' num2str(Task) '.mat']);
%montage(uint8(store), 'Size', [10 34]);
selected = [0,1,5,20]+1;

hb = figure;
ind = [];
for i = selected
    for j = [1:numClass]
    
        
        ind = [ind (j-1)*Hypo + i];
    end
end

store = store(:,:,:,ind);
montage(uint8(store), 'Size', [4  numClass]);

printpostfix = '.png';
printmode = '-dpng'; %-depsc
printoption = '-r200'; %'-fillpage'
set(hb,'Units','Inches');
pos = get(hb,'Position');
set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(hb,['Figure/Hypothesis_' num2str(Task) '.png'],printmode,printoption);