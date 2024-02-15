clear all
close all

load('AK101197_ECG_FetMatHRResults.mat')
load('AK101197.mat')

channel=3;
duration=60;
t = 0:1/fs:duration-1/fs; % 10 seconds signal
figure;
plot(t, abdmECG(channel,1:duration*fs));
hold on;
% Plot maternal R-peaks
plot(maternalRPeakTimes(maternalRPeakTimes < duration) , zeros(length(maternalRPeakTimes(maternalRPeakTimes < duration))), 'ro', 'MarkerSize', 10);
% Add labels and title

abdmECG=abdmECG(:,1:duration*fs);
maternalRPeakTimes=maternalRPeakTimes(maternalRPeakTimes < duration); 
%fetal ECG extraction
residual = FECGSYN_kf_extraction(maternalRPeakTimes*1000,abdmECG(channel,:),0);
plot(t, residual);

xlabel('Time (s)');
ylabel('Amplitude');
title('abdmECG Signal with Maternal R-peaks');

