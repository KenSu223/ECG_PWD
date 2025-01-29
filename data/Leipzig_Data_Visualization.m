%Script for visualizing the Leipzig data 

clear 
close all
load('Version2/CB300482IIIV2.mat')
downsample_factor = 5; %from 20kHz to 4kHz
doppler = decimate(doppler, downsample_factor);

fs=1000;
fs_D=4000;
time =1/fs:1/fs:length(fECG)/fs;
time_D =1/fs_D:1/fs_D:length(doppler)/fs_D;


numChannels=7;
numSamples=length(fECG);
% time = 1:numSamples;
figure;
for i = 1:numChannels
    subplot(numChannels, 1, i);
    plot(time_D,doppler);
    hold on

    plot(time, 2*fECG(i,:),'LineWidth', 3);

    title(sprintf('Channel %d', i),'FontSize', 14);

    if i == numChannels
        xlabel('Time (S)', 'FontSize', 18);
    end
ax = gca;
ax.XAxis.FontSize = 18; % Set font size for x-axis ticks to 18
ax.YAxis.FontSize = 18; % Set font size for y-axis ticks to 18

end


linkaxes(findall(gcf, 'type', 'axes'), 'x');


