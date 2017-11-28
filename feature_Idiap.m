function featuresIdiap = feature_Idiap(iterCntIdiaptrain, iterCntIdiaptest,  varargin )
% clear all;
% close all;
% varargin = {'distortion', 'glcm', 'visualRhythmHori', 'visualRhythm'};
% varargin = {'distortion', 'glcm', 'lbpU8'};
% varargin = {'lbpU8', 'diffusionWeightOverlapLBP'};
% varargin = {'distortion', 'glcm'};
% train: attack_highdef_client108_session01_highdef_photo_controlled 가 수상하다

%% init
setDir = fullfile(cd, 'imageDatabases', 'Idiap');
numFrames = 30;
numTotalFeatures = 0;
numCurrentFeatures = 0;
distortionFeatureFlag = 0;
glcmFeatureFlag = 0;
visualRhythmHoriFlag = 0;
overlapJPGLBPFlag = 0; % lbp uniform 8 Flag
mappingU8=getmapping(8,'u2');
lbpU16Flag = 0; % lbp uniform 16 Flag
mappingU16=getmapping(16,'u2');
lbpU8FourierFlag = 0; % lbp uniform 8 for fourier spectrum Flag

diffusionWeightLbpU8Flag = 0;
diffusionWeightOverlapLBPFlag = 0;

ligRmvPpmLBPFlag = 0;
ligRmvPpmDistortionFlag = 0;
ligRmvPpmGlcmFlag = 0;

cutTop = 0;
cutBottom = 0;

paraStr = '';

%% parameter setting
for i = 1 : length(varargin),
    paraStr = strcat(paraStr, '_', varargin{i});
end

if ~isempty(strfind(paraStr,'distortion'))
    distortionFeatureFlag = 1;
    numTotalFeatures = numTotalFeatures + 115;
end
if ~isempty(strfind(paraStr,'glcm'))
    glcmFeatureFlag = 1;
    numTotalFeatures = numTotalFeatures + 88;
end
if ~isempty(strfind(paraStr,'visualRhythmHori'))
    visualRhythmHoriFlag = 1;
    numTotalFeatures = numTotalFeatures + 3600;
end
if ~isempty(strfind(paraStr,'overlapJPGLBP'))
    overlapJPGLBPFlag = 1;
    numTotalFeatures = numTotalFeatures + 833;
end
if ~isempty(strfind(paraStr,'lbpU8Fourier'))
    lbpU8FourierFlag = 1;
    numTotalFeatures = numTotalFeatures + 59;
end
if ~isempty(strfind(paraStr,'lbpU16'))
    lbpU16Flag = 1;
    numTotalFeatures = numTotalFeatures + 243;
end
if ~isempty(strfind(paraStr,'diffusionWeightOverlapLBP'))
    diffusionWeightOverlapLBPFlag = 1;
    numTotalFeatures = numTotalFeatures + 833;
end

if ~isempty(strfind(paraStr,'ligRmvPpmLBP'))
    ligRmvPpmLBPFlag = 1;
    numTotalFeatures = numTotalFeatures + 833;
end

if ~isempty(strfind(paraStr,'ligRmvPpmDistortion'))
    ligRmvPpmDistortionFlag = 1;
    numTotalFeatures = numTotalFeatures + 115;
end
if ~isempty(strfind(paraStr,'ligRmvPpmGlcm'))
    ligRmvPpmGlcmFlag = 1;
    numTotalFeatures = numTotalFeatures + 88;
end



trainPath = fullfile('imageDatabases', 'Idiap', 'train');
testPath = fullfile('imageDatabases', 'Idiap', 'test');

% 60 (categories) * 50 (each video is selected only 50 frames)
trainGenuine = zeros( 60*numFrames, numTotalFeatures );
% fixed 60, hand 60 1video가 없음
trainReplay = zeros( (60+60-1)*numFrames, numTotalFeatures );
% fixed 90, hand 90 1 video가 없음
trainPrinted = zeros( (90+90-1)*numFrames, numTotalFeatures );

% 80 (categories) * 50 (each video is selected only 50 frames)
testGenuine = zeros( 80*numFrames, numTotalFeatures );
% fixed 80, hand 80
testReplay = zeros( (80+80)*numFrames, numTotalFeatures );
% fixed 120, hand 120
testPrinted = zeros( (120+120)*numFrames, numTotalFeatures );


trainDir= dir([trainPath '\*.jpg']);
% trainDir= dir([trainPath '\*_LigRmv.ppm']);
trainFileNames = {trainDir.name};
testDir= dir([testPath '\*.jpg']);
% testDir= dir([testPath '\*_LigRmv.ppm']);
testFileNames = {testDir.name};
numCurrentGenRow = 0;
numCurrentPriRow = 0;
numCurrentRepRow = 0;
%% extrcat train feature
if distortionFeatureFlag || glcmFeatureFlag || visualRhythmHoriFlag || overlapJPGLBPFlag || lbpU8FourierFlag || lbpU16Flag || diffusionWeightOverlapLBPFlag,
    for i = 1:size(trainFileNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(trainPath, '\', trainFileNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        ind=find(ismember(iterCntIdiaptrain(:, 1), strrep(trainFileNames{i}, '.jpg', '.ppm')));
        ligRmvCnt = iterCntIdiaptrain{ind,2};
        if ligRmvCnt <= 3,
            iterCntDW = 8;
        elseif ligRmvCnt <= 6,
            iterCntDW = 7;
        elseif ligRmvCnt <= 9,
            iterCntDW = 6;
        else
            iterCntDW = 5;
        end
        
        
        if ~isempty(strfind(trainFileNames{i},'authenticate')), % genuine
            numCurrentGenRow = numCurrentGenRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(blurriness))) = blurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(blurriness);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(nonRefBlurriness);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(chromatic))) = chromatic;
                numCurrentGenFeatures = numCurrentGenFeatures + length(chromatic);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(diversity))) = diversity;
                numCurrentGenFeatures = numCurrentGenFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(horiReshape))) = horiReshape;
                numCurrentGenFeatures = numCurrentGenFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(trainFileNames{i},'photo')), % photo
            numCurrentPriRow = numCurrentPriRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(blurriness))) = blurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(blurriness);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(nonRefBlurriness);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(chromatic))) = chromatic;
                numCurrentPriFeatures = numCurrentPriFeatures + length(chromatic);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(diversity))) = diversity;
                numCurrentPriFeatures = numCurrentPriFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(horiReshape))) = horiReshape;
                numCurrentPriFeatures = numCurrentPriFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(trainFileNames{i},'video')), % replay
            numCurrentRepRow = numCurrentRepRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(blurriness))) = blurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(blurriness);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(nonRefBlurriness);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(chromatic))) = chromatic;
                numCurrentRepFeatures = numCurrentRepFeatures + length(chromatic);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(diversity))) = diversity;
                numCurrentRepFeatures = numCurrentRepFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(horiReshape))) = horiReshape;
                numCurrentRepFeatures = numCurrentRepFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
        end
        
    end
end

%% ppm extract training feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    trainLigRmvDir= dir([trainPath '\*_LigRmv.ppm']);
    trainLigRmvNames = {trainLigRmvDir.name};
    
    for i = 1:size(trainLigRmvNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(trainPath, '\', trainLigRmvNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        if ~isempty(strfind(trainFileNames{i},'authenticate')), % genuine
            numCurrentGenRow = numCurrentGenRow + 1;
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(blurriness))) = blurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(blurriness);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(nonRefBlurriness);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(chromatic))) = chromatic;
                numCurrentGenFeatures = numCurrentGenFeatures + length(chromatic);
                trainGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(diversity))) = diversity;
                numCurrentGenFeatures = numCurrentGenFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(trainFileNames{i},'photo')), % photo
            numCurrentPriRow = numCurrentPriRow + 1;
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(blurriness))) = blurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(blurriness);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(nonRefBlurriness);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(chromatic))) = chromatic;
                numCurrentPriFeatures = numCurrentPriFeatures + length(chromatic);
                trainPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(diversity))) = diversity;
                numCurrentPriFeatures = numCurrentPriFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16);
                trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(trainFileNames{i},'video')), % replay
            numCurrentRepRow = numCurrentRepRow + 1;
            
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(blurriness))) = blurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(blurriness);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(nonRefBlurriness);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(chromatic))) = chromatic;
                numCurrentRepFeatures = numCurrentRepFeatures + length(chromatic);
                trainReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(diversity))) = diversity;
                numCurrentRepFeatures = numCurrentRepFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16);
                trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
            
            
        end
        
    end
end


numCurrentGenRow = 0;
numCurrentPriRow = 0;
numCurrentRepRow = 0;
%% extrcat test feature
if distortionFeatureFlag || glcmFeatureFlag || visualRhythmHoriFlag || overlapJPGLBPFlag || lbpU8FourierFlag || lbpU16Flag || diffusionWeightOverlapLBPFlag,
    for i = 1:size(testFileNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(testPath, '\', testFileNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        ind=find(ismember(iterCntIdiaptest(:, 1), strrep(testFileNames{i}, '.jpg', '.ppm')));
        ligRmvCnt = iterCntIdiaptest{ind,2};
        if ligRmvCnt <= 3,
            iterCntDW = 8;
        elseif ligRmvCnt <= 6,
            iterCntDW = 7;
        elseif ligRmvCnt <= 9,
            iterCntDW = 6;
        else
            iterCntDW = 5;
        end
                
        if ~isempty(strfind(testFileNames{i},'authenticate')), % genuine
            numCurrentGenRow = numCurrentGenRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(blurriness))) = blurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(blurriness);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(nonRefBlurriness);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(chromatic))) = chromatic;
                numCurrentGenFeatures = numCurrentGenFeatures + length(chromatic);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(diversity))) = diversity;
                numCurrentGenFeatures = numCurrentGenFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(horiReshape))) = horiReshape;
                numCurrentGenFeatures = numCurrentGenFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(testFileNames{i},'photo')), % photo
            numCurrentPriRow = numCurrentPriRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(blurriness))) = blurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(blurriness);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(nonRefBlurriness);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(chromatic))) = chromatic;
                numCurrentPriFeatures = numCurrentPriFeatures + length(chromatic);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(diversity))) = diversity;
                numCurrentPriFeatures = numCurrentPriFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(horiReshape))) = horiReshape;
                numCurrentPriFeatures = numCurrentPriFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(testFileNames{i},'video')), % replay
            numCurrentRepRow = numCurrentRepRow + 1;
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(blurriness))) = blurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(blurriness);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(nonRefBlurriness);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(chromatic))) = chromatic;
                numCurrentRepFeatures = numCurrentRepFeatures + length(chromatic);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(diversity))) = diversity;
                numCurrentRepFeatures = numCurrentRepFeatures + length(diversity);
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(horiReshape))) = horiReshape;
                numCurrentRepFeatures = numCurrentRepFeatures + length(horiReshape);
            end
            
            % lbp uniform 8 features
            if overlapJPGLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'nh');
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'nh');
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                %lbpFeatures = overlapLBP( I, mappingU8, mappingU16 );
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
        end
    end
end

%% ppm extract training feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    testLigRmvDir= dir([testPath '\*_LigRmv.ppm']);
    testLigRmvNames = {testLigRmvDir.name};
    
    for i = 1:size(testLigRmvNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(testPath, '\', testLigRmvNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        if ~isempty(strfind(testFileNames{i},'authenticate')), % genuine
            numCurrentGenRow = numCurrentGenRow + 1;
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(blurriness))) = blurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(blurriness);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentGenFeatures = numCurrentGenFeatures + length(nonRefBlurriness);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(chromatic))) = chromatic;
                numCurrentGenFeatures = numCurrentGenFeatures + length(chromatic);
                testGenuine( numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(diversity))) = diversity;
                numCurrentGenFeatures = numCurrentGenFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentGenFeatures = numCurrentGenFeatures + length(lbpFeatures);
            end
            
            
        elseif ~isempty(strfind(testFileNames{i},'photo')), % photo
            numCurrentPriRow = numCurrentPriRow + 1;
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(blurriness))) = blurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(blurriness);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentPriFeatures = numCurrentPriFeatures + length(nonRefBlurriness);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(chromatic))) = chromatic;
                numCurrentPriFeatures = numCurrentPriFeatures + length(chromatic);
                testPrinted( numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(diversity))) = diversity;
                numCurrentPriFeatures = numCurrentPriFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentPriFeatures = numCurrentPriFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
            end
            
        elseif ~isempty(strfind(testFileNames{i},'video')), % replay
            numCurrentRepRow = numCurrentRepRow + 1;
            
            
            if ligRmvPpmDistortionFlag,
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(blurriness))) = blurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(blurriness);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentRepFeatures = numCurrentRepFeatures + length(nonRefBlurriness);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(chromatic))) = chromatic;
                numCurrentRepFeatures = numCurrentRepFeatures + length(chromatic);
                testReplay( numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(diversity))) = diversity;
                numCurrentRepFeatures = numCurrentRepFeatures + length(diversity);
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentRepFeatures = numCurrentRepFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
            end
            
        end
        
    end
end



featuresIdiap.trainGenuine = trainGenuine;
featuresIdiap.trainReplay = trainReplay;
featuresIdiap.trainPrinted = trainPrinted;
featuresIdiap.testGenuine = testGenuine;
featuresIdiap.testReplay = testReplay;
featuresIdiap.testPrinted = testPrinted;






end




