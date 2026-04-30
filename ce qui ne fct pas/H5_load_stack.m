filename = 'L44E_Nov02_JCtest.h5';

% Get all station groups
info = h5info(filename, '/results/vectorData');
nStations = length(info.Groups);
nSamples = 1600;

% Pre-allocate - stackType has 'value' and 'error' fields
Vstack_value = zeros(nSamples, nStations);
Vstack_error = zeros(nSamples, nStations);
Istack_value = zeros(nSamples, nStations);
Istack_error = zeros(nSamples, nStations);
stationIDs   = zeros(nStations, 1);

% Load all stations
for k = 1:nStations
    stationPath = info.Groups(k).Name;
    parts = strsplit(stationPath, '/');
    stationIDs(k) = str2double(parts{end});
    
    V = h5read(filename, [stationPath '/Vstack']);
    I = h5read(filename, [stationPath '/Istack']);
    
    Vstack_value(:, k) = V.value;
    Vstack_error(:, k) = V.error;
    Istack_value(:, k) = I.value;
    Istack_error(:, k) = I.error;
    
    if mod(k, 200) == 0
        fprintf('Loaded %d / %d stations...\n', k, nStations);
    end
end

fprintf('Done! Arrays are %d x %d (samples x stations)\n', nSamples, nStations);

[stationIDs_sorted, sortIdx] = sort(stationIDs);
Vstack_value = Vstack_value(:, sortIdx);
Istack_value = Istack_value(:, sortIdx);

k = 393;
figure(1);
clf;
plot(Vstack_value(:, k));
title(sprintf('Vstack - Station %d (ID: %d)', k, stationIDs(k)));
xlabel('Sample');
ylabel('Value');
pause(0.5); % adjust speed, or press any key if you use pause() alone
