function featuresCASIA = feature_CASIA(iterCntCASIAtrain, iterCntCASIAtest,  varargin )
% clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%c%%%%%%%%%%%%%%%%%
% CASIA Database
%
%             |    low quality    middle quality     high quality
% ------------------------------------------------------------------------------------
% genuine |        1                     2                     HR1
% photo    |       3, 5                 4, 6                 HR2, HR3
% replay    |        7                     8                     HR4
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% varargin = {'distortion', 'glcm', 'visualRhythm'};
% varargin = {'distortion', 'glcm'};
% varargin = { 'distortion', 'glcm', 'overlapJPGLBP'};
%% init
% numTotalTrain = 12000;
% numTotalTest = 18000;
% numObjTrain = 20;
% numObjTest = 30;
numFrames = 30;
numTotalFeatures = 0;
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
if ~isempty(strfind(paraStr,'diffusionWeightLbpU8'))
    diffusionWeightLbpU8Flag = 1;
    numTotalFeatures = numTotalFeatures + 59;
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

trainPath = fullfile('imageDatabases', 'CASIA', 'train_release');
testPath = fullfile('imageDatabases', 'CASIA', 'test_release');
imgName = {'1'; '2'; '3'; '4'; '5'; '6'; '7'; '8'; 'HR_1'; 'HR_2'; 'HR_3'; 'HR_4'};
imgGroup = {'001'; '002'; '003'; '004'; '005'; '006'; '007'; '008'; '009'; '010'; '011'; '012'; '013'; '014'; '015';...
    '016'; '017'; '018'; '019'; '020'; '021'; '022'; '023'; '024'; '025'; '026'; '027'; '028'; '029'; '030';};

% 3 * 20 (categories) * 50 (each video is selected only 50 frames)
trainGenuine = zeros( 3*20*numFrames, numTotalFeatures );
trainReplay = zeros( 3*20*numFrames, numTotalFeatures );
% 6 * 20 (categories) * 50 (each video is selected only 50 frames)
trainPrinted = zeros( 6*20*numFrames, numTotalFeatures );

% 3 * 30 (categories) * 50 (each video is selected only 50 frames)
testGenuine = zeros( 3*30*numFrames, numTotalFeatures );
testReplay = zeros( 3*30*numFrames, numTotalFeatures );
% 6 * 30 (categories) * 50 (each video is selected only 50 frames)
testPrinted = zeros( 3*60*numFrames, numTotalFeatures );

trainDir= dir([trainPath '\*.jpg']);
% trainDir= dir([trainPath '\*-LigRmv.ppm']);
trainFileNames = {trainDir.name};
testDir= dir([testPath '\*.jpg']);
% testDir= dir([testPath '\*-LigRmv.ppm']);
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
        
        ind=find(ismember(iterCntCASIAtrain(:, 1), strrep(trainFileNames{i}, '.jpg', '.ppm')));
        ligRmvCnt = iterCntCASIAtrain{ind,2};        
        if ligRmvCnt <= 3,
            iterCntDW = 8;
        elseif ligRmvCnt <= 6,
            iterCntDW = 7;
        elseif ligRmvCnt <= 9,
            iterCntDW = 6;
        else
            iterCntDW = 5;
        end
        
        %% _HR_
        if ~isempty(strfind(trainFileNames{i},'_HR_')),
            if ~isempty(strfind(trainFileNames{i},'_1_')), % genuine
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
                    
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fContrast))) = glcmFeatures.fContrast;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fContrast);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fCorrelation))) = glcmFeatures.fCorrelation;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fCorrelation);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fDiffEntr))) = glcmFeatures.fDiffEntr;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fDiffEntr);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fDiffVari))) = glcmFeatures.fDiffVari;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fDiffVari);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fEnergy))) = glcmFeatures.fEnergy;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fEnergy);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fEntropy))) = glcmFeatures.fEntropy;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fEntropy);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fHomogeneity))) = glcmFeatures.fHomogeneity;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fHomogeneity);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fInMeaCor1))) = glcmFeatures.fInMeaCor1;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fInMeaCor1);
                    %                 %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fInMeaCor2))) = glcmFeatures.fInMeaCor2;
                    %                 %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fInMeaCor2);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fSumAver))) = glcmFeatures.fSumAver;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fSumAver);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fSumEntr))) = glcmFeatures.fSumEntr;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fSumEntr);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fSumVari))) = glcmFeatures.fSumVari;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fSumVari);
                    %                 trainGenuine(numCurrentGenRow, (numCurrentGenFeatures+1) : (numCurrentGenFeatures + length(glcmFeatures.fVariance))) = glcmFeatures.fVariance;
                    %                 numCurrentGenFeatures = numCurrentGenFeatures + length(glcmFeatures.fVariance);
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
                    %lbpFeatures = lbp(I,1,8,mappingU8,'nh');
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
                
            elseif ~isempty(strfind(trainFileNames{i},'_2_')) || ~isempty(strfind(trainFileNames{i},'_3_')), % photo
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
                
            elseif ~isempty(strfind(trainFileNames{i},'_4_')), % replay
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
            
        else
            %% no _HR_
            if ~isempty(strfind(trainFileNames{i},'_1_')) || ~isempty(strfind(trainFileNames{i},'_2_')), % genuine
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
                
            elseif ~isempty(strfind(trainFileNames{i},'_3_')) || ~isempty(strfind(trainFileNames{i},'_4_')) ...
                    || ~isempty(strfind(trainFileNames{i},'_5_')) || ~isempty(strfind(trainFileNames{i},'_6_')), % photo
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
                
            elseif ~isempty(strfind(trainFileNames{i},'_7_')) || ~isempty(strfind(trainFileNames{i},'_8_')), % replay
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
end

%% ppm extract training feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    
    trainLigRmvDir= dir([trainPath '\*-LigRmv.ppm']);
    trainLigRmvNames = {trainLigRmvDir.name};
    
    
    for i = 1:size(trainLigRmvNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(trainPath, '\', trainLigRmvNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        %% _HR_
        if ~isempty(strfind(trainLigRmvNames{i},'_HR_')),
            if ~isempty(strfind(trainLigRmvNames{i},'_1_')), % genuine
                numCurrentGenRow = numCurrentGenRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                
                
            elseif ~isempty(strfind(trainLigRmvNames{i},'_2_')) || ~isempty(strfind(trainLigRmvNames{i},'_3_')), % photo
                numCurrentPriRow = numCurrentPriRow + 1;
                
                
                if ligRmvPpmDistortionFlag,
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
                    lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                    trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
                end
                
            elseif ~isempty(strfind(trainLigRmvNames{i},'_4_')), % replay
                numCurrentRepRow = numCurrentRepRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                    lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                    trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
                end
            end
            
        else
            %% no _HR_
            if ~isempty(strfind(trainLigRmvNames{i},'_1_')) || ~isempty(strfind(trainLigRmvNames{i},'_2_')), % genuine
                numCurrentGenRow = numCurrentGenRow + 1;
                                
                if ligRmvPpmDistortionFlag,
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
                
            elseif ~isempty(strfind(trainLigRmvNames{i},'_3_')) || ~isempty(strfind(trainLigRmvNames{i},'_4_')) ...
                    || ~isempty(strfind(trainLigRmvNames{i},'_5_')) || ~isempty(strfind(trainLigRmvNames{i},'_6_')), % photo
                numCurrentPriRow = numCurrentPriRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                    lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                    trainPrinted(numCurrentPriRow, (numCurrentPriFeatures+1) : (numCurrentPriFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentPriFeatures = numCurrentPriFeatures + length(lbpFeatures);
                end
                                
            elseif ~isempty(strfind(trainLigRmvNames{i},'_7_')) || ~isempty(strfind(trainLigRmvNames{i},'_8_')), % replay
                numCurrentRepRow = numCurrentRepRow + 1;
                                
                if ligRmvPpmDistortionFlag,
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
                    lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                    trainReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
                end
                
            end
        end
    end    
end


numCurrentGenRow = 0;
numCurrentPriRow = 0;
numCurrentRepRow = 0;
if distortionFeatureFlag || glcmFeatureFlag || visualRhythmHoriFlag || overlapJPGLBPFlag || lbpU8FourierFlag || lbpU16Flag || diffusionWeightOverlapLBPFlag,
    %% extract test feature
    for i = 1:size(testFileNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(testPath, '\', testFileNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);        
        
        ind=find(ismember(iterCntCASIAtest(:, 1), strrep(testFileNames{i}, '.jpg', '.ppm')));
        ligRmvCnt = iterCntCASIAtest{ind,2};
        if ligRmvCnt <= 3,
            iterCntDW = 8;
        elseif ligRmvCnt <= 6,
            iterCntDW = 7;
        elseif ligRmvCnt <= 9,
            iterCntDW = 6;
        else
            iterCntDW = 5;
        end
        
        %% _HR_
        if ~isempty(strfind(testFileNames{i},'_HR_')),
            if ~isempty(strfind(testFileNames{i},'_1_')), % genuine
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
                
            elseif ~isempty(strfind(testFileNames{i},'_2_')) || ~isempty(strfind(testFileNames{i},'_3_')), % photo
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
                
            elseif ~isempty(strfind(testFileNames{i},'_4_')), % replay
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
                    testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
                end
                
            end
            
        else
            %% no _HR_
            if ~isempty(strfind(testFileNames{i},'_1_')) || ~isempty(strfind(testFileNames{i},'_2_')), % genuine
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
                
            elseif ~isempty(strfind(testFileNames{i},'_3_')) || ~isempty(strfind(testFileNames{i},'_4_')) ...
                    || ~isempty(strfind(testFileNames{i},'_5_')) || ~isempty(strfind(testFileNames{i},'_6_')), % photo
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
                
            elseif ~isempty(strfind(testFileNames{i},'_7_')) || ~isempty(strfind(testFileNames{i},'_8_')), % replay
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
                    testReplay(numCurrentRepRow, (numCurrentRepFeatures+1) : (numCurrentRepFeatures + length(lbpFeatures))) = lbpFeatures;
                    numCurrentRepFeatures = numCurrentRepFeatures + length(lbpFeatures);
                end
                
            end
        end
    end
end



%% ppm extract test feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    
    testLigRmvDir= dir([testPath '\*-LigRmv.ppm']);
    testLigRmvNames = {testLigRmvDir.name};
    
    for i = 1:size(testLigRmvNames, 2),
        numCurrentGenFeatures = 0;
        numCurrentPriFeatures = 0;
        numCurrentRepFeatures = 0;
        
        I = imread(strcat(testPath, '\', testLigRmvNames{i}));
        
        I = cutImage(I, cutTop, cutBottom);
        
        %% _HR_
        if ~isempty(strfind(testLigRmvNames{i},'_HR_')),
            if ~isempty(strfind(testLigRmvNames{i},'_1_')), % genuine
                numCurrentGenRow = numCurrentGenRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                
            elseif ~isempty(strfind(testLigRmvNames{i},'_2_')) || ~isempty(strfind(testLigRmvNames{i},'_3_')), % photo
                numCurrentPriRow = numCurrentPriRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                
                
            elseif ~isempty(strfind(testLigRmvNames{i},'_4_')), % replay
                numCurrentRepRow = numCurrentRepRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
            
        else
            %% no _HR_
            if ~isempty(strfind(testLigRmvNames{i},'_1_')) || ~isempty(strfind(testLigRmvNames{i},'_2_')), % genuine
                numCurrentGenRow = numCurrentGenRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                
            elseif ~isempty(strfind(testLigRmvNames{i},'_3_')) || ~isempty(strfind(testLigRmvNames{i},'_4_')) ...
                    || ~isempty(strfind(testLigRmvNames{i},'_5_')) || ~isempty(strfind(testLigRmvNames{i},'_6_')), % photo
                numCurrentPriRow = numCurrentPriRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
                
                
            elseif ~isempty(strfind(testLigRmvNames{i},'_7_')) || ~isempty(strfind(testLigRmvNames{i},'_8_')), % replay
                numCurrentRepRow = numCurrentRepRow + 1;
                
                if ligRmvPpmDistortionFlag,
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
end

featuresCASIA.trainGenuine = trainGenuine;
featuresCASIA.trainReplay = trainReplay;
featuresCASIA.trainPrinted = trainPrinted;
featuresCASIA.testGenuine = testGenuine;
featuresCASIA.testReplay = testReplay;
featuresCASIA.testPrinted = testPrinted;


