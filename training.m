
% clear all;
% featuresCASIA = feature_CASIA( 'distortion', 'glcm', 'lbpU8', 'lbpU8Fourier', 'lbpU16' );
% featuresMSU = feature_MSUMFSD( 'distortion', 'glcm', 'lbpU8', 'lbpU8Fourier', 'lbpU16' );

% featuresCASIA = feature_CASIA( 'distortion', 'glcm', 'lbpU8', 'lbpU8Fourier', 'lbpU16' );
% featuresMSU = feature_MSUMFSD( 'distortion', 'glcm', 'lbpU8', 'lbpU8Fourier', 'lbpU16' );
% featuresIdiap = feature_Idiap( 'distortion', 'glcm', 'lbpU8', 'lbpU8Fourier', 'lbpU16' );
% 
% featuresCASIA = feature_CASIA( 'distortion', 'glcm');
% featuresIdiap = feature_Idiap( 'distortion', 'glcm');
% 
% featuresMSUPPM = feature_MSUMFSD( 'distortion', 'glcm', 'diffusionWeightLbpU8', 'lbpU8', 'lbpU16');
% featuresCASIAPPM = feature_CASIA( 'distortion', 'glcm', 'diffusionWeightLbpU8', 'lbpU8', 'lbpU16');
% featuresIdiapPPM = feature_Idiap( 'distortion', 'glcm', 'diffusionWeightLbpU8', 'lbpU8', 'lbpU16');


% txt 파일 읽기
%SpeDifTxtIdiap = readSpeDifTxtIdiap();

%save('SpeDifTxtIdiap.mat', 'SpeDifTxtIdiap', '-v7.3');

%% 2015 12 02, jpg, ppm 작업 후--
featuresMSU = feature_MSUMFSD(iterCntMSU, 'distortion', 'glcm', 'overlapJPGLBP');
featuresMSUDWLBP = feature_MSUMFSD(iterCntMSU, 'diffusionWeightOverlapLBP');
featuresMSUPPM = feature_MSUMFSD(iterCntMSU, 'ligRmvPpmDistortion','ligRmvPpmGlcm' ,'ligRmvPpmLBP');

featuresCASIA = feature_CASIA(iterCntCASIAtrain, iterCntCASIAtest, 'distortion', 'glcm', 'overlapJPGLBP');
featuresCASIADWLBP = feature_CASIA(iterCntCASIAtrain, iterCntCASIAtest, 'diffusionWeightOverlapLBP');
featuresCASIAPPM = feature_CASIA(iterCntCASIAtrain, iterCntCASIAtest, 'ligRmvPpmDistortion','ligRmvPpmGlcm' ,'ligRmvPpmLBP');

featuresIdiap = feature_Idiap(iterCntIdiaptrain, iterCntIdiaptest, 'distortion', 'glcm', 'overlapJPGLBP');
featuresIdiapDWLBP = feature_Idiap(iterCntIdiaptrain, iterCntIdiaptest, 'diffusionWeightOverlapLBP');
featuresIdiapPPM = feature_Idiap(iterCntIdiaptrain, iterCntIdiaptest, 'ligRmvPpmDistortion','ligRmvPpmGlcm' ,'ligRmvPpmLBP');

