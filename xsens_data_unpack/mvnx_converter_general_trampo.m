function [data] = mvnx_converter_general_trampo(mvnx, file_dir, file_name, Subject_name, Move_name)

    %trim unwanted information
    temp = mvnx.subject.frames.frame;
    global_JCS_positions = mvnx.subject.frames.frame(3).position;
    global_JCS_orientations = mvnx.subject.frames.frame(3).orientation;
    temp = temp(4:end); %get rid of unwanted fields
    frameRate = mvnx.subject.frameRate;
    % fields = { 'tc' , 'ms', 'type', 'index', 'footContacts', 'jointAngleXZY', 'jointAngleErgoXZY', 'jointAngleErgo', 'centerOfMass'}; %fields you would like to remove
    % temp = rmfield(temp, fields); 

    for j=1:length(mvnx.subject.segments.segment)
        segmentQLabels{1 + 4*(j-1)} = {[mvnx.subject.segments.segment(j).label, '_w']} ; %23 segmentsx4 =92
        segmentQLabels{1 + 4*(j-1) +1} = {[ mvnx.subject.segments.segment(j).label, '_i']};
        segmentQLabels{1 + 4*(j-1) +2} ={[ mvnx.subject.segments.segment(j).label, '_j']};
        segmentQLabels{1 + 4*(j-1) +3} ={[ mvnx.subject.segments.segment(j).label, '_k']};
    end
    segmentQLabels=[segmentQLabels{:}];

    for j=1:length(mvnx.subject.segments.segment)
        segmentLabels{1 + 3*(j-1)} = {[mvnx.subject.segments.segment(j).label, '_x']} ; %23 segmentsx3 =69
        segmentLabels{1 + 3*(j-1) + 1} = {[ mvnx.subject.segments.segment(j).label, '_y']};
        segmentLabels{1 + 3*(j-1) +2} ={[ mvnx.subject.segments.segment(j).label, '_z']};
    end
    segmentLabels=[segmentLabels{:}];

    for j=1:length(mvnx.subject.sensors.sensor)
        sensorLabels{1 + 3*(j-1)} = {[ mvnx.subject.sensors.sensor(j).label, '_x']}; %17 sensors x 3=51
        sensorLabels{1 + 3*(j-1)+1} = {[ mvnx.subject.sensors.sensor(j).label, '_y']};
        sensorLabels{1 + 3*(j-1)+2} = {[ mvnx.subject.sensors.sensor(j).label, '_z']};
    end
    sensorLabels=[sensorLabels{:}];

    for j=1:length(mvnx.subject.joints.joint) 
        jointLabels{1 + 3*(j-1)} = {[ mvnx.subject.joints.joint(j).label, '_x']};  %22 jointsx3=66
        jointLabels{1 + 3*(j-1)+1} = {[ mvnx.subject.joints.joint(j).label, '_y']};
        jointLabels{1 + 3*(j-1)+2} = {[ mvnx.subject.joints.joint(j).label, '_z']};
    end
    jointLabels=[jointLabels{:}];

    for j=1:length(mvnx.subject.sensors.sensor)
        sensorQLabels{1 + 4*(j-1)} = {[ mvnx.subject.sensors.sensor(j).label, '_w']}; %17 segmentsx4 =68
        sensorQLabels{1 + 4*(j-1) + 1} = {[ mvnx.subject.sensors.sensor(j).label, '_i']};
        sensorQLabels{1 + 4*(j-1) +2} ={[ mvnx.subject.sensors.sensor(j).label, '_j']};
        sensorQLabels{1 + 4*(j-1) +3} ={[ mvnx.subject.sensors.sensor(j).label, '_k']};
    end
    sensorQLabels=[sensorQLabels{:}];


    orientation = temp(1).orientation; %92 columns

    position = temp(1).position; %69 columns     
    velocity = temp(1).velocity;
    acceleration = temp(1).acceleration;
    angularVelocity = temp(1).angularVelocity;
    angularAcceleration = temp(1).angularAcceleration;

    sensorOrientation = temp(1).sensorOrientation; %68?

    sensorFreeAcceleration = temp(1).sensorFreeAcceleration;
    sensorMagneticField = temp(1).sensorMagneticField; %51

    jointAngle = temp(1).jointAngle; %66
    centerOfMass = temp(1).centerOfMass;
    
    time = temp(1).time;
    index = temp(1).index;
    ms = temp(1).ms;

    for j=2:length(temp)

         orientation=[orientation;  temp(j).orientation];

         position=[position;  temp(j).position];
         velocity=[velocity;  temp(j).velocity];
         acceleration=[acceleration;  temp(j).acceleration];
         angularVelocity=[angularVelocity;  temp(j).angularVelocity];
         angularAcceleration=[angularAcceleration;  temp(j).angularAcceleration];

         sensorFreeAcceleration=[sensorFreeAcceleration;  temp(j).sensorFreeAcceleration];
         sensorMagneticField=[sensorMagneticField;  temp(j).sensorMagneticField];

         sensorOrientation=[sensorOrientation;  temp(j).sensorOrientation];

         jointAngle=[jointAngle;  temp(j).jointAngle];
         centerOfMass=[centerOfMass;  temp(j).centerOfMass];
        
         time = [time; temp(j).time];
         index = [index; temp(j).index];
         ms = [ms; temp(j).ms];

    end
    

    %% Add metadata 
    
%     new_folder_name = [file_dir, '/', Subject_name, '_', mvnx.comment];
    new_folder_name = [file_dir, '/', Move_name];

    eval(sprintf("mkdir %s", new_folder_name))
    fileOut = ([file_dir '/' file_name(1:end-5)]);

    save( [new_folder_name, '/', 'Subject_name.mat'], 'Subject_name')
    save( [new_folder_name, '/', 'Move_name.mat'], 'Move_name')
    save( [new_folder_name, '/', 'frameRate.mat'], 'frameRate')
    save( [new_folder_name, '/', 'time.mat'], 'time')
    save( [new_folder_name, '/', 'index.mat'], 'index')
    save( [new_folder_name, '/', 'ms.mat'], 'ms')
    save( [new_folder_name, '/', 'position.mat'], 'position')
    save( [new_folder_name, '/', 'orientation.mat'], 'orientation')
    save( [new_folder_name, '/', 'velocity.mat'], 'velocity')
    save( [new_folder_name, '/', 'acceleration.mat'], 'acceleration')
    save( [new_folder_name, '/', 'angularVelocity.mat'], 'angularVelocity')
    save( [new_folder_name, '/', 'angularAcceleration.mat'], 'angularAcceleration')
    save( [new_folder_name, '/', 'sensorFreeAcceleration.mat'], 'sensorFreeAcceleration')
    save( [new_folder_name, '/', 'sensorOrientation.mat'], 'sensorOrientation')
    save( [new_folder_name, '/', 'jointAngle.mat'], 'jointAngle')
    save( [new_folder_name, '/', 'centerOfMass.mat'], 'centerOfMass')
    save( [new_folder_name, '/', 'global_JCS_positions.mat'], 'global_JCS_positions')
    save( [new_folder_name, '/', 'global_JCS_orientations.mat'], 'global_JCS_orientations')

    clear data mvnx temp 


end
