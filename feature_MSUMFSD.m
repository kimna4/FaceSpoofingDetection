function featuresMSUMFSD = feature_MSUMFSD(iterCntMSU, varargin )
% clear all;
% close all;
% varargin = {'distortion', 'glcm', 'visualRhythmHori', 'visualRhythm'};
% varargin = {'distortion', 'glcm', 'diffusionWeightLbpU8'};
% varargin = {'lbpU8'};
% I = imread('attack.jpg');
%% init
setDir = fullfile(cd, 'imageDatabases', 'MSUMFSD');
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


%% init according to the parameter
% testList = load(fullfile(setDir, 'test_sub_list.txt'));
testList = ['01'; '13'; '14'; '23'; '24'; '26'; '28'; '29'; '30'; '32'; '33'; '35'; '36'; '37'; '39'; '42'; '48'; '49'; '50'; '51' ];
trainList = ['02';  '03'; '05'; '06'; '07'; '08'; '09'; '11'; '12'; '21'; '22'; '34'; '53'; '54'; '55'];
% android * laptop * 50 (each video is selected only 50 frames)
trainGenuine = zeros( size(trainList, 1)*2*numFrames, numTotalFeatures );
% android * laptop * ipad * iphone * 50 (each video is selected only 50 frames)
trainReplay = zeros( size(trainList, 1)*4*numFrames, numTotalFeatures );
trainPrinted = zeros( size(trainList, 1)*2*numFrames, numTotalFeatures );

% android * laptop * 50 (each video is selected only 50 frames)
testGenuine = zeros( size(testList, 1)*2*numFrames, numTotalFeatures );
% android * laptop * ipad * iphone * 50 (each video is selected only 50 frames)
testReplay = zeros( size(testList, 1)*4*numFrames, numTotalFeatures );
testPrinted = zeros( size(testList, 1)*2*numFrames, numTotalFeatures );


if distortionFeatureFlag || glcmFeatureFlag || visualRhythmHoriFlag || overlapJPGLBPFlag || lbpU8FourierFlag || lbpU16Flag || diffusionWeightOverlapLBPFlag,
    %% train feature extraction
    for i = 1:size(trainList, 1),
        
        realAndroid = strcat('real_client0', trainList(i,1), trainList(i,2), '_android_SD_scene01');
        realLaptop = strcat('real_client0', trainList(i,1), trainList(i,2), '_laptop_SD_scene01');
        attackAndroidIpad = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_ipad_video_scene01');
        attackAndroidIphone = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_iphone_video_scene01');
        attackAndroidPrinted = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_printed_photo_scene01');
        attackLaptopIpad = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_ipad_video_scene01');
        attackLaptopIphone = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_iphone_video_scene01');
        attackLaptopPrinted = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_printed_photo_scene01');
        
        %% real android & laptop
        curDir1 = fullfile(setDir, realAndroid);
        imglist1 = dir([curDir1 '\*.jpg']);
        %     imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, realLaptop);
        imglist2 = dir([curDir2 '\*.jpg']);
        %     imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1)); % using 2-d image
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1)); % using 2-d image
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 8, features with fourier spectrum
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16, features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            %         if diffusionWeightLbpU8Flag,
            %             lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16);
            %             trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
            %             numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            %         end
            
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
        end
        
        %% attack replay
        curDir1 = fullfile(setDir, attackAndroidIpad);
        imglist1 = dir([curDir1 '\*.jpg']);
        curDir2 = fullfile(setDir, attackAndroidIphone);
        imglist2 = dir([curDir2 '\*.jpg']);
        curDir3 = fullfile(setDir, attackLaptopIpad);
        imglist3 = dir([curDir3 '\*.jpg']);
        curDir4 = fullfile(setDir, attackLaptopIphone);
        imglist4 = dir([curDir4 '\*.jpg']);
        totalImgList = [imglist1; imglist2; imglist3; imglist4];
        totalImgListSize = size(totalImgList, 1);
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir1, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir2, totalImgList(j).name));
                end
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir3, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir4, totalImgList(j).name));
                end
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
              
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
        end
        
        %% attack print
        curDir1 = fullfile(setDir, attackAndroidPrinted);
        imglist1 = dir([curDir1 '\*.jpg']);
        curDir2 = fullfile(setDir, attackLaptopPrinted);
        imglist2 = dir([curDir2 '\*.jpg']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
              
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
                        
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
    end
end

%% ppm extract training feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    for i = 1:size(trainList, 1),
        
        realAndroid = strcat('real_client0', trainList(i,1), trainList(i,2), '_android_SD_scene01');
        realLaptop = strcat('real_client0', trainList(i,1), trainList(i,2), '_laptop_SD_scene01');
        attackAndroidIpad = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_ipad_video_scene01');
        attackAndroidIphone = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_iphone_video_scene01');
        attackAndroidPrinted = strcat('attack_client0', trainList(i,1), trainList(i,2), '_android_SD_printed_photo_scene01');
        attackLaptopIpad = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_ipad_video_scene01');
        attackLaptopIphone = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_iphone_video_scene01');
        attackLaptopPrinted = strcat('attack_client0', trainList(i,1), trainList(i,2), '_laptop_SD_printed_photo_scene01');
        
        %% real android & laptop
        curDir1 = fullfile(setDir, realAndroid);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        %     imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, realLaptop);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        %     imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I, mappingU8, mappingU16 );
                trainGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
        end
        
        %% attack replay
        curDir1 = fullfile(setDir, attackAndroidIpad);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, attackAndroidIphone);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        curDir3 = fullfile(setDir, attackLaptopIpad);
        imglist3 = dir([curDir3 '\*_LigRmv.ppm']);
        curDir4 = fullfile(setDir, attackLaptopIphone);
        imglist4 = dir([curDir4 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2; imglist3; imglist4];
        totalImgListSize = size(totalImgList, 1);
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir1, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir2, totalImgList(j).name));
                end
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir3, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir4, totalImgList(j).name));
                end
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            
            
        end
        
        %% attack print
        curDir1 = fullfile(setDir, attackAndroidPrinted);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, attackLaptopPrinted);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                trainPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
    end
    
end


if distortionFeatureFlag || glcmFeatureFlag || visualRhythmHoriFlag || overlapJPGLBPFlag || lbpU8FourierFlag || lbpU16Flag || diffusionWeightOverlapLBPFlag,
    
    %% test feature extraction
    for i = 1:size(testList, 1),
        
        realAndroid = strcat('real_client0', testList(i,1), testList(i,2), '_android_SD_scene01');
        realLaptop = strcat('real_client0', testList(i,1), testList(i,2), '_laptop_SD_scene01');
        attackAndroidIpad = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_ipad_video_scene01');
        attackAndroidIphone = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_iphone_video_scene01');
        attackAndroidPrinted = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_printed_photo_scene01');
        attackLaptopIpad = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_ipad_video_scene01');
        attackLaptopIphone = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_iphone_video_scene01');
        attackLaptopPrinted = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_printed_photo_scene01');
        
        %% real android & laptop
        curDir1 = fullfile(setDir, realAndroid);
        imglist1 = dir([curDir1 '\*.jpg']);
        curDir2 = fullfile(setDir, realLaptop);
        imglist2 = dir([curDir2 '\*.jpg']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
                          
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
                                    
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
        
        %% attack replay
        curDir1 = fullfile(setDir, attackAndroidIpad);
        imglist1 = dir([curDir1 '\*.jpg']);
        curDir2 = fullfile(setDir, attackAndroidIphone);
        imglist2 = dir([curDir2 '\*.jpg']);
        curDir3 = fullfile(setDir, attackLaptopIpad);
        imglist3 = dir([curDir3 '\*.jpg']);
        curDir4 = fullfile(setDir, attackLaptopIphone);
        imglist4 = dir([curDir4 '\*.jpg']);
        totalImgList = [imglist1; imglist2; imglist3; imglist4];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir1, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir2, totalImgList(j).name));
                end
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir3, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir4, totalImgList(j).name));
                end
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
                          
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
        
        %% attack print
        curDir1 = fullfile(setDir, attackAndroidPrinted);
        imglist1 = dir([curDir1 '\*.jpg']);
        curDir2 = fullfile(setDir, attackLaptopPrinted);
        imglist2 = dir([curDir2 '\*.jpg']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
                          
            ind=find(ismember(iterCntMSU(:, 1), strrep(totalImgList(j).name, '.jpg', '.ppm')));
            ligRmvCnt = iterCntMSU{ind,2};
            if ligRmvCnt <= 3,
                iterCntDW = 8;
            elseif ligRmvCnt <= 6,
                iterCntDW = 7;
            elseif ligRmvCnt <= 9,
                iterCntDW = 6;
            else
                iterCntDW = 5;
            end
            
            if distortionFeatureFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            % glcm features
            if glcmFeatureFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
            end
            
            % visual rhythm features
            if visualRhythmHoriFlag,
                visualRhythmFeatures = visualRhythm(I(:, :, 1));
                horiReshape = reshape(visualRhythmFeatures.horizontalLine', 1, []);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(horiReshape))) = horiReshape;
                numCurrentFeatures = numCurrentFeatures + length(horiReshape);
            end
            
            % lbp overlap 883 features
            if overlapJPGLBPFlag,
                %lbpFeatures = lbp(I,1,8,mappingU8,'h');
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            if lbpU8FourierFlag,
                F = fourierSpectrum(I(:, :, 1), 'noiseMode', 'median', 7);
                lbpFeatures = lbp(F,1,8,mappingU8,'h');
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp uniform 16 features
            if lbpU16Flag,
                lbpFeatures = lbp(I,2,16,mappingU16,'h');
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            % lbp diffusion weight uniform 8 features
            if diffusionWeightOverlapLBPFlag,
                lbpFeatures = diffusionWeightLBP(I, 1, 8, mappingU8, mappingU16, iterCntDW);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
    end
end


%% ppm extract test feature
if ligRmvPpmLBPFlag || ligRmvPpmDistortionFlag  || ligRmvPpmGlcmFlag,
    
    %% test feature extraction
    for i = 1:size(testList, 1),
        
        realAndroid = strcat('real_client0', testList(i,1), testList(i,2), '_android_SD_scene01');
        realLaptop = strcat('real_client0', testList(i,1), testList(i,2), '_laptop_SD_scene01');
        attackAndroidIpad = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_ipad_video_scene01');
        attackAndroidIphone = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_iphone_video_scene01');
        attackAndroidPrinted = strcat('attack_client0', testList(i,1), testList(i,2), '_android_SD_printed_photo_scene01');
        attackLaptopIpad = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_ipad_video_scene01');
        attackLaptopIphone = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_iphone_video_scene01');
        attackLaptopPrinted = strcat('attack_client0', testList(i,1), testList(i,2), '_laptop_SD_printed_photo_scene01');
        
        %% real android & laptop
        curDir1 = fullfile(setDir, realAndroid);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, realLaptop);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testGenuine((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
        end
        
        %% attack replay
        curDir1 = fullfile(setDir, attackAndroidIpad);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, attackAndroidIphone);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        curDir3 = fullfile(setDir, attackLaptopIpad);
        imglist3 = dir([curDir3 '\*_LigRmv.ppm']);
        curDir4 = fullfile(setDir, attackLaptopIphone);
        imglist4 = dir([curDir4 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2; imglist3; imglist4];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir1, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir2, totalImgList(j).name));
                end
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                if ~isempty(strfind(totalImgList(j).name,'_ipad_'))
                    I = imread(fullfile(curDir3, totalImgList(j).name));
                elseif  ~isempty(strfind(totalImgList(j).name,'_iphone_'))
                    I = imread(fullfile(curDir4, totalImgList(j).name));
                end
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testReplay((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
        end
        
        %% attack print
        curDir1 = fullfile(setDir, attackAndroidPrinted);
        imglist1 = dir([curDir1 '\*_LigRmv.ppm']);
        curDir2 = fullfile(setDir, attackLaptopPrinted);
        imglist2 = dir([curDir2 '\*_LigRmv.ppm']);
        totalImgList = [imglist1; imglist2];
        totalImgListSize = size(totalImgList, 1);
        
        for j = 1: size(totalImgList, 1),
            numCurrentFeatures = 0; % init numCurrentFeatures;
            if ~isempty(strfind(totalImgList(j).name,'_android_'))
                I = imread(fullfile(curDir1, totalImgList(j).name));
            elseif ~isempty(strfind(totalImgList(j).name,'_laptop_'))
                I = imread(fullfile(curDir2, totalImgList(j).name));
            end
            
            I = cutImage(I, cutTop, cutBottom);
            % [ssr I msrcr ] = retinex(I);
            
            
            if ligRmvPpmDistortionFlag,
                % blurriness features
                blurriness = blurMetric(I);
                nonRefBlurriness = noRefferencePerceptualBlurMetric(I(:, :, 1));
                % Chromatic Moment Features
                chromatic = chromaticMomentFeatures(I);
                % color diversity features
                diversity = colorDiversity(I);
                
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(blurriness))) = blurriness;
                numCurrentFeatures = numCurrentFeatures + length(blurriness);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(nonRefBlurriness))) = nonRefBlurriness;
                numCurrentFeatures = numCurrentFeatures + length(nonRefBlurriness);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(chromatic'))) = chromatic';
                numCurrentFeatures = numCurrentFeatures + length(chromatic');
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(diversity'))) = diversity';
                numCurrentFeatures = numCurrentFeatures + length(diversity');
            end
            
            if ligRmvPpmGlcmFlag,
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [-1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [0 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 0] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
                
                glcm = graycomatrix( I(:, :, 1), 'NumLevels', 16, 'offset', [1 -1] );
                glcmFeatures = GLCMFeatures(glcm, 16);
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(glcmFeatures'))) = glcmFeatures';
                numCurrentFeatures = numCurrentFeatures + length(glcmFeatures');
            end
            
            if ligRmvPpmLBPFlag,
                lbpFeatures = overlapLBP( I(:, :, 1), mappingU8, mappingU16 );
                testPrinted((i-1)*totalImgListSize+j, (numCurrentFeatures+1) : (numCurrentFeatures + length(lbpFeatures))) = lbpFeatures;
                numCurrentFeatures = numCurrentFeatures + length(lbpFeatures);
            end
            
            
            
        end
    end
    
end


featuresMSUMFSD.trainGenuine = trainGenuine;
featuresMSUMFSD.trainReplay = trainReplay;
featuresMSUMFSD.trainPrinted = trainPrinted;
featuresMSUMFSD.testGenuine = testGenuine;
featuresMSUMFSD.testReplay = testReplay;
featuresMSUMFSD.testPrinted = testPrinted;






