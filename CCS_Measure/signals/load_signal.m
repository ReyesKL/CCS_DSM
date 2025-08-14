function [time, sig] = load_signal(file_path, set_min, set_max)
    %% Load Siganl - Simple function for reading a CSV signal
    % INPUT ARGUMENTS:
    %   file_path (required) string - Path to the desired CSV file you wish
    %   to load
    %   set_min (optional) double - Minimum value to scale the signal to. 
    %   set_max (optional) double - Maximum value to scale the signal to.

    % Handle input argumens to this function
    arguments
        file_path           (1,1) string {mustBeFile}
        set_min             (1,1) double = NaN
        set_max             (1,1) double = NaN
    end
    
    %open the file as a CSV
    data = readtable(file_path);
    
    %now extract the relevant data
    time = data.Time; sig = data.Signal;
    
    %have the minimum and maximum values default to the minimum and maximum
    %of the signal
    if isnan(set_min); set_min = min(sig); end
    if isnan(set_max); set_max = max(sig); end
    
    %get the minimum and maximum of the signal
    x_min = min(sig); x_max = max(sig);

    %compute the scaling parameters
    K = (set_max - set_min) / (x_max - x_min);
    D = set_min - K*x_min;
    
    %now re-scale the signal
    sig = K.*sig + D;
end