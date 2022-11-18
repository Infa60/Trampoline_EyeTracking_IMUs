% this code allows to create a map of the gaze trajectory from Xsens +
% eye-tracker measurements on trampolinists


clear all; close all; clc;

% file_dir = '/home/fbailly/Documents/Programmation/gaze_trajectory_Pupil_Xsens/tests_error/Tests_5aout2022/Xsens';
% file_name = 'Test_17032021-007.mvnx';
% mvnx = load_mvnx([file_dir, '/exports/' file_name])
% mvnx = load_mvnx([file_dir, '/' file_name])
% Look at the comment to know which trial it is!
% Subject_name = 'GuSe';
% Move_name = ['4-o', '8-1o', '8--o'];
% mvnx_converter_general_trampo(mvnx, file_dir, file_name, Subject_name, Move_name);
            
%% OR do it in loop

Subject_name = 'SaMi';
file_dir = ['/home/fbailly/disk/Eye-tracking/XsensData/', Subject_name, '/exports_shoulder_height'];

dir_content = dir(file_dir);

for i = 1:length(dir_content)
    file_name = dir_content(i).name
    if length(file_name) > 5
        if strcmp(file_name(end-4:end), ".mvnx")
            mvnx = load_mvnx([dir_content(i).folder, '/', file_name]);
%             Subject_name = '';
%             Move_name = [mvnx.comment];
            Move_name = [file_name(1:end-5)];
            mvnx_converter_general_trampo(mvnx, file_dir, file_name, Subject_name, Move_name);
        end
    end
end

disp('Success')

















