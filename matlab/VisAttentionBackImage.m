clear all; close all; clc;
NumClasses = 10;

common = [];
for c= 1:10
load(['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/class_instance/class_instance_run_0_task_4_epoch_0_att_read_' num2str(c-1) '.mat']);
common = [common; unique(att_read)];
end
uniquehypolist = [0; unique(common)];

save('Mat/uniquehypolist.mat','uniquehypolist');
load('Mat/uniquehypolist.mat');

ExpTypeList = {'iid','class_iid','class_instance','instance'};
Type = 3;
Run = 1;
Epoch= 3;
TaskTotal = 5;
NumMemSlots = 100;
mean=[0.485, 0.456, 0.406];
std=[0.229, 0.224, 0.225];
                         

hb = figure;

imgsz = 128;
%store = ones( imgsz,imgsz,3, length(uniquehypolist)*NumClasses);
store = ones( imgsz,imgsz,3, length(uniquehypolist)*2);
counter = 1;



for Task = 1 %5 %[1:TaskTotal]
    %selectedfiles = dir();

    for selected = [1:2] %[1:NumClasses]
        if Task == 1
            Epoch = 3;
        else
            Epoch = 1;
        end
        matfilename = ['/home/mengmi/Projects/Proj_CL_NTM/pytorch/core50/visualization/' ExpTypeList{Type} '/' ExpTypeList{Type} ...
        '_run_' num2str(Run-1) '_task_' num2str(Task-1) '_epoch_' num2str(Epoch-1) '_att_read_' num2str(selected-1) '.mat'];

        if exist(matfilename, 'file') ~= 2
            continue;
        end
        load(matfilename);
        %uniquehypolist = unique(att_read);
        inputs = reshape(inputs,[50, 3, 224, 224]);
        for q = 1

            for hypothesis = uniquehypolist'

                img = double(squeeze(inputs(q,:,:,:)));
                for c = [1:3]
                    img(c,:,:) = img(c,:,:)*std(c)+mean(c);
                end
                img = permute(img, [2 3 1]);
                img = uint8(img*256);
                img = imresize(img,[imgsz imgsz]);

                if hypothesis == 0
                    store(:,:,:,counter) = img;
                    counter = counter + 1;
                    continue;
                end
                
                att = squeeze(att_read(q,:,:,:));
                empty = zeros(size(att));
                empty(find(att == hypothesis)) = 1;
                empty = sum(empty,3);
                %empty = empty(:,:,1);
                empty = 1-mat2gray(empty);
                empty = imresize(empty, [imgsz imgsz]);
                %hypoatt = heatmap_overlay(img, empty);

                figure;
                h = imshow(img);
                set(h, 'AlphaData', 1-empty);
                hold on;
                g = imshow(1-empty);
                set(g, 'AlphaData', empty);
                %imshow(heat);
                %title(['step =' num2str(t) '; ' num2str(i) '; response:' nms{predicted_seq(t)+1}]);
                saveas(gca, ['temp.jpg'])
                temp = imread('temp.jpg');
                temp = temp(44:243,150:344,:);
                temp = imresize(temp,[imgsz imgsz]);
                
                store(:,:,:,counter) = temp;
                counter = counter + 1;
%                 imshow(temp);
%                 drawnow;
%                 pause(0.1);
                close all;

            end
        end

    end
end



%montage(uint8(store), 'Size', [ NumClasses length(uniquehypolist)]);
montage(uint8(store), 'Size', [ 2 length(uniquehypolist)]);
store = uint8(store);
save(['Mat/montage_' num2str(Task) '.mat'],'store');