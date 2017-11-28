clear all;
close all;

% read image - It will complete soon
I = imread('attack_client055_laptop_SD_iphone_video_scene01_050.jpg');

% blurriness features
blurriness = blurMetric(I);
nonRefBlurriness = noRefferencePerceptualBlurMetric(I);

% Chromatic Moment Features
chromatic = chromaticMomentFeatures(I);

% color diversity features
diversity = colorDiversity(I);

