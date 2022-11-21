% This code allows to unpack the mvnx file from Xsens

clear all; close all; clc;

Subject_name = 'JeCh_2';
file_dir = ['/home/fbailly/disk/Eye-tracking/XsensData/', Subject_name, '/exports_shoulder_height'];

dir_content = dir(file_dir);

for i = 1:length(dir_content)
    file_name = dir_content(i).name
    if length(file_name) > 5
        if strcmp(file_name(end-4:end), ".mvnx")
            mvnx = load_mvnx([dir_content(i).folder, '/', file_name]);
            Move_name = [file_name(1:end-5)];
            mvnx_converter_general_trampo(mvnx, file_dir, file_name, Subject_name, Move_name);
        end
    end
end

disp('Success')

















