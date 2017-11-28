%% feature train & test %%
% clear all;
% featuresCASIA = feature_CASIA( 'distortion', 'glcm', 'lbpU8' );
% featuresMSU = feature_MSU( 'distortion', 'glcm', 'lbpU8' );
% distortion: 115, glcm: 12, visualRhythmHori: 3600, lbpU8: 59,
% lbpU8Fourier: 59, lbpU16: 243
%


%% init
numFrames = 30;

%% CASE 1 - only MSU Features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY
% JPG replay
MSU_TrainData = [featuresMSU.trainGenuine; featuresMSU.trainReplay; ];
MSU_TestData = [featuresMSU.testGenuine; featuresMSU.testReplay; ];
% DW replay
MSU_TrainDataDW = [featuresMSUDWLBP.trainGenuine; featuresMSUDWLBP.trainReplay;];
MSU_TestDataDW = [featuresMSUDWLBP.testGenuine; featuresMSUDWLBP.testReplay;];
% ppm replay
MSU_TrainDataPPM = [featuresMSUPPM.trainGenuine; featuresMSUPPM.trainReplay;];
MSU_TestDataPPM = [featuresMSUPPM.testGenuine; featuresMSUPPM.testReplay;];
% Label replay
MSU_TrainLabel = [1*ones(size(featuresMSU.trainGenuine, 1), 1); -1*ones(size(featuresMSU.trainReplay, 1), 1)];
MSU_TestLabel = [1*ones(size(featuresMSU.testGenuine, 1), 1); -1*ones(size(featuresMSU.testReplay, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- PRINTED
% JPG print
MSU_TrainData = [featuresMSU.trainGenuine; featuresMSU.trainPrinted];
MSU_TestData = [featuresMSU.testGenuine; featuresMSU.testPrinted];
% DW print
MSU_TrainDataDW = [featuresMSUDWLBP.trainGenuine; featuresMSUDWLBP.trainPrinted];
MSU_TestDataDW = [featuresMSUDWLBP.testGenuine; featuresMSUDWLBP.testPrinted];
% ppm print
MSU_TrainDataPPM = [featuresMSUPPM.trainGenuine; featuresMSUPPM.trainPrinted];
MSU_TestDataPPM = [featuresMSUPPM.testGenuine; featuresMSUPPM.testPrinted];
% Label print
MSU_TrainLabel = [ones(size(featuresMSU.trainGenuine, 1), 1); -1*ones(size(featuresMSU.trainPrinted, 1), 1)];
MSU_TestLabel = [ones(size(featuresMSU.testGenuine, 1), 1); -1*ones(size(featuresMSU.testPrinted, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY + PRINTED
% JPG replay + print
MSU_TrainData = [featuresMSU.trainGenuine; featuresMSU.trainReplay; featuresMSU.trainPrinted];
MSU_TestData = [featuresMSU.testGenuine; featuresMSU.testReplay;  featuresMSU.testPrinted];
% DW replay + print
MSU_TrainDataDW = [featuresMSUDWLBP.trainGenuine; featuresMSUDWLBP.trainReplay; featuresMSUDWLBP.trainPrinted];
MSU_TestDataDW = [featuresMSUDWLBP.testGenuine; featuresMSUDWLBP.testReplay; featuresMSUDWLBP.testPrinted];
% ppm replay + print
MSU_TrainDataPPM = [featuresMSUPPM.trainGenuine; featuresMSUPPM.trainReplay; featuresMSUPPM.trainPrinted];
MSU_TestDataPPM = [featuresMSUPPM.testGenuine; featuresMSUPPM.testReplay; featuresMSUPPM.testPrinted];
% Label replay + print
MSU_TrainLabel = [ones(size(featuresMSU.trainGenuine, 1), 1); -1*ones(size(featuresMSU.trainReplay, 1), 1); -1*ones(size(featuresMSU.trainPrinted, 1), 1)];
MSU_TestLabel = [ones(size(featuresMSU.testGenuine, 1), 1); -1*ones(size(featuresMSU.testReplay, 1), 1); -1*ones(size(featuresMSU.testPrinted, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jpg (dis, glcm, lbp)
MSU_TrainDistortion = MSU_TrainData(:, 1:115);
MSU_TrainGlcm = MSU_TrainData(:, 116:203);
MSU_TrainLBP = MSU_TrainData(:, 204:1036);
MSU_TestDistortion = MSU_TestData(:, 1:115);
MSU_TestGlcm = MSU_TestData(:, 116:203);
MSU_TestLBP = MSU_TestData(:, 204:1036);

%dw (lbp)
MSU_TrainDW = MSU_TrainDataDW(:, 1:833);
MSU_TestDW = MSU_TestDataDW(:, 1:833);

%ppm (dis, glcm, lbp)
MSU_TrainDistortionPPM = MSU_TrainDataPPM(:, 1:115);
MSU_TrainGlcmPPM = MSU_TrainDataPPM(:, 116:203);
MSU_TrainLBPPPM = MSU_TrainDataPPM(:, 204:1036);
MSU_TestDistortionPPM = MSU_TestDataPPM(:, 1:115);
MSU_TestGlcmPPM = MSU_TestDataPPM(:, 116:203);
MSU_TestLBPPPM = MSU_TestDataPPM(:, 204:1036);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% distortion features normalization
[MSU_TrainDistortion MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainDistortion);
MSU_TestDistortion = featureNormalizeForTesting(MSU_TestDistortion, MSU_FeaturesMu, MSU_FeaturesStddev);
% glcm features normalization
[MSU_TrainGlcm MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainGlcm);
MSU_TestGlcm = featureNormalizeForTesting(MSU_TestGlcm, MSU_FeaturesMu, MSU_FeaturesStddev);
% LBP normalization
[MSU_TrainLBP MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainLBP);
MSU_TestLBP = featureNormalizeForTesting(MSU_TestLBP, MSU_FeaturesMu, MSU_FeaturesStddev);

% DW normaization
[MSU_TrainDW MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainDW);
MSU_TestDW = featureNormalizeForTesting(MSU_TestDW, MSU_FeaturesMu, MSU_FeaturesStddev);

% ppm normaization
[MSU_TrainDistortionPPM MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainDistortionPPM);
MSU_TestDistortionPPM = featureNormalizeForTesting(MSU_TestDistortionPPM, MSU_FeaturesMu, MSU_FeaturesStddev);
[MSU_TrainGlcmPPM MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainGlcmPPM);
MSU_TestGlcmPPM = featureNormalizeForTesting(MSU_TestGlcmPPM, MSU_FeaturesMu, MSU_FeaturesStddev);
[MSU_TrainLBPPPM MSU_FeaturesMu MSU_FeaturesStddev ]= featureNormalize(MSU_TrainLBPPPM);
MSU_TestLBPPPM = featureNormalizeForTesting(MSU_TestLBPPPM, MSU_FeaturesMu, MSU_FeaturesStddev);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. JPG: dis + glcm // 
MSU_TrainData = [MSU_TrainDistortion  MSU_TrainGlcm];
MSU_TestData = [MSU_TestDistortion  MSU_TestGlcm];

% 2. jpg: dis// ppm: LBP
MSU_TrainData = [MSU_TrainDistortion MSU_TrainLBPPPM];
MSU_TestData = [MSU_TestDistortion  MSU_TestLBPPPM];

% 3. jpg: dis // DW: LBP
MSU_TrainData = [MSU_TrainDistortion  MSU_TrainDW];
MSU_TestData = [MSU_TestDistortion  MSU_TestDW];

% 4. jpg: dis + glcm // ppm: LBP
MSU_TrainData = [MSU_TrainDistortion  MSU_TrainGlcm MSU_TrainLBPPPM];
MSU_TestData = [MSU_TestDistortion  MSU_TestGlcm MSU_TestLBPPPM];

% 5. jpg: dis + glcm // DW: LBP
MSU_TrainData = [MSU_TrainDistortion  MSU_TrainGlcm MSU_TrainDW];
MSU_TestData = [MSU_TestDistortion  MSU_TestGlcm MSU_TestDW];

% 6. DW: lbp // ppm: LBP
MSU_TrainData = [MSU_TrainDW MSU_TrainLBPPPM ];
MSU_TestData = [MSU_TestDW MSU_TestLBPPPM];

% 7. JPG: lbp // DW: LBP // PPM: LBP 
MSU_TrainData = [MSU_TrainLBP MSU_TrainDW MSU_TrainLBPPPM ];
MSU_TestData = [MSU_TestLBP MSU_TestDW MSU_TestLBPPPM];

% 8. JPG: dis + LBP // PPM: LBP // DW: LBP
MSU_TrainData = [MSU_TrainDistortion MSU_TrainLBP  MSU_TrainLBPPPM MSU_TrainDW];
MSU_TestData = [MSU_TestDistortion MSU_TestLBP MSU_TestLBPPPM MSU_TestDW];

% 9. JPG: dis + LBP
MSU_TrainData = [MSU_TrainDistortion MSU_TrainLBP];
MSU_TestData = [MSU_TestDistortion MSU_TestLBP];

% 10. JPG: LBP
MSU_TrainData = [MSU_TrainLBP];
MSU_TestData = [MSU_TestLBP];

% linear SVM
MSU_Model_1 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 0.5000 -g 0.0313 -t 2');
[MSU_predict_label_1, MSU_accuracy_1, MSU_dec_values1] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_1);

MSU_Model_2 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 2 -g 0.0313 -t 2');
[MSU_predict_label_2, MSU_accuracy_2, MSU_dec_values2] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_2);

MSU_Model_3 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 128 -g 0.0000991 -t 2');
[MSU_predict_label_3, MSU_accuracy_3, MSU_dec_values3] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_3);

MSU_Model_4 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 330 -g 0.00099 -t 2');
[MSU_predict_label_4, MSU_accuracy_4, MSU_dec_values4] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_4);

MSU_Model_5 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 220 -g 0.000097 -t 2');
[MSU_predict_label_5, MSU_accuracy_5, MSU_dec_values5] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_5);

MSU_Model_6 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 195 -g 0.00139-t 2');
[MSU_predict_label_6, MSU_accuracy_6, MSU_dec_values6] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_6);

% nogada
MSU_Model_5 = svmtrain(MSU_TrainLabel, MSU_TrainData, '-c 0.353553390593274 -g 0.000976562500000000 -t 2');
[MSU_predict_label_5, MSU_accuracy_5, MSU_dec_values5] = svmpredict(MSU_TestLabel, MSU_TestData, MSU_Model_5);

% HTER
MSU_predict_label = MSU_predict_label_5;
[acc HTER ] = getHTER( MSU_predict_label,  MSU_TestLabel, numFrames);

% ROC
MSU_dec_values = MSU_dec_values5;
[X, Y, T, AUC] = perfcurve(MSU_TestLabel, MSU_dec_values, 1);
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('True positive rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == 0.01,
        fprintf('Y: %f \n', Y(i));
    end
end

%ERR
MSU_dec_values = MSU_dec_values5;
[X, Y, T, AUC] = perfcurve(MSU_TestLabel, MSU_dec_values, 1, 'XCrit', 'fpr', 'Ycrit', 'fnr');
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('False negative rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == Y(i),
        fprintf('X: %f Y: %f \n', X(i), Y(i));
    end
end


% auto find parameter numFrames= 30
 [ resultAuto ] = findParameter( MSU_TrainLabel, MSU_TrainData, MSU_TestLabel, MSU_TestData, numFrames, 'A' ); 
% auto find parameter Fine
 [ resultAutoFine ] = findParameterFine( MSU_TrainLabel, MSU_TrainData, MSU_TestLabel, MSU_TestData, numFrames, 0, 0, 0, 0 ); 
 


%% case2 - only CASIA Features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY
% JPG replay
CASIA_TrainData = [featuresCASIA.trainGenuine; featuresCASIA.trainReplay; ];
CASIA_TestData = [featuresCASIA.testGenuine; featuresCASIA.testReplay; ];
% DW replay 
CASIA_TrainDataDW = [featuresCASIADWLBP.trainGenuine; featuresCASIADWLBP.trainReplay;];
CASIA_TestDataDW = [featuresCASIADWLBP.testGenuine; featuresCASIADWLBP.testReplay;];
% ppm replay
CASIA_TrainDataPPM = [featuresCASIAPPM.trainGenuine; featuresCASIAPPM.trainReplay;];
CASIA_TestDataPPM = [featuresCASIAPPM.testGenuine; featuresCASIAPPM.testReplay;];
% Label replay
CASIA_TrainLabel = [1*ones(size(featuresCASIA.trainGenuine, 1), 1); -1*ones(size(featuresCASIA.trainReplay, 1), 1)];
CASIA_TestLabel = [1*ones(size(featuresCASIA.testGenuine, 1), 1); -1*ones(size(featuresCASIA.testReplay, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- PRINTED
% JPG print
CASIA_TrainData = [featuresCASIA.trainGenuine; featuresCASIA.trainPrinted];
CASIA_TestData = [featuresCASIA.testGenuine; featuresCASIA.testPrinted];
% DW print
CASIA_TrainDataDW = [featuresCASIADWLBP.trainGenuine; featuresCASIADWLBP.trainPrinted];
CASIA_TestDataDW = [featuresCASIADWLBP.testGenuine; featuresCASIADWLBP.testPrinted];
% ppm print
CASIA_TrainDataPPM = [featuresCASIAPPM.trainGenuine; featuresCASIAPPM.trainPrinted];
CASIA_TestDataPPM = [featuresCASIAPPM.testGenuine; featuresCASIAPPM.testPrinted];
% Label print
CASIA_TrainLabel = [ones(size(featuresCASIA.trainGenuine, 1), 1); -1*ones(size(featuresCASIA.trainPrinted, 1), 1)];
CASIA_TestLabel = [ones(size(featuresCASIA.testGenuine, 1), 1); -1*ones(size(featuresCASIA.testPrinted, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY + PRINTED
% JPG replay + print
CASIA_TrainData = [featuresCASIA.trainGenuine; featuresCASIA.trainReplay; featuresCASIA.trainPrinted];
CASIA_TestData = [featuresCASIA.testGenuine; featuresCASIA.testReplay;  featuresCASIA.testPrinted];
% DW replay + print
CASIA_TrainDataDW = [featuresCASIADWLBP.trainGenuine; featuresCASIADWLBP.trainReplay; featuresCASIADWLBP.trainPrinted];
CASIA_TestDataDW = [featuresCASIADWLBP.testGenuine; featuresCASIADWLBP.testReplay; featuresCASIADWLBP.testPrinted];
% ppm replay + print
CASIA_TrainDataPPM = [featuresCASIAPPM.trainGenuine; featuresCASIAPPM.trainReplay; featuresCASIAPPM.trainPrinted];
CASIA_TestDataPPM = [featuresCASIAPPM.testGenuine; featuresCASIAPPM.testReplay; featuresCASIAPPM.testPrinted];
% Label replay + print
CASIA_TrainLabel = [ones(size(featuresCASIA.trainGenuine, 1), 1); -1*ones(size(featuresCASIA.trainReplay, 1), 1); -1*ones(size(featuresCASIA.trainPrinted, 1), 1)];
CASIA_TestLabel = [ones(size(featuresCASIA.testGenuine, 1), 1); -1*ones(size(featuresCASIA.testReplay, 1), 1); -1*ones(size(featuresCASIA.testPrinted, 1), 1)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jpg (dis, glcm, lbp)
CASIA_TrainDistortion = CASIA_TrainData(:, 1:115);
CASIA_TrainGlcm = CASIA_TrainData(:, 116:203);
CASIA_TrainLBP = CASIA_TrainData(:, 204:1036);
CASIA_TestDistortion = CASIA_TestData(:, 1:115);
CASIA_TestGlcm = CASIA_TestData(:, 116:203);
CASIA_TestLBP = CASIA_TestData(:, 204:1036);

%dw (lbp)
CASIA_TrainDW = CASIA_TrainDataDW(:, 1:833);
CASIA_TestDW = CASIA_TestDataDW(:, 1:833);

%ppm (dis, glcm, lbp)
CASIA_TrainDistortionPPM = CASIA_TrainDataPPM(:, 1:115);
CASIA_TrainGlcmPPM = CASIA_TrainDataPPM(:, 116:203);
CASIA_TrainLBPPPM = CASIA_TrainDataPPM(:, 204:1036);
CASIA_TestDistortionPPM = CASIA_TestDataPPM(:, 1:115);
CASIA_TestGlcmPPM = CASIA_TestDataPPM(:, 116:203);
CASIA_TestLBPPPM = CASIA_TestDataPPM(:, 204:1036);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% distortion features normalization
[CASIA_TrainDistortion CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainDistortion);
CASIA_TestDistortion = featureNormalizeForTesting(CASIA_TestDistortion, CASIA_FeaturesMu, CASIA_FeaturesStddev);
% glcm features normalization
[CASIA_TrainGlcm CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainGlcm);
CASIA_TestGlcm = featureNormalizeForTesting(CASIA_TestGlcm, CASIA_FeaturesMu, CASIA_FeaturesStddev);
% LBP normalization
[CASIA_TrainLBP CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainLBP);
CASIA_TestLBP = featureNormalizeForTesting(CASIA_TestLBP, CASIA_FeaturesMu, CASIA_FeaturesStddev);

% DW normaization
[CASIA_TrainDW CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainDW);
CASIA_TestDW = featureNormalizeForTesting(CASIA_TestDW, CASIA_FeaturesMu, CASIA_FeaturesStddev);

% ppm normaization
[CASIA_TrainDistortionPPM CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainDistortionPPM);
CASIA_TestDistortionPPM = featureNormalizeForTesting(CASIA_TestDistortionPPM, CASIA_FeaturesMu, CASIA_FeaturesStddev);
[CASIA_TrainGlcmPPM CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainGlcmPPM);
CASIA_TestGlcmPPM = featureNormalizeForTesting(CASIA_TestGlcmPPM, CASIA_FeaturesMu, CASIA_FeaturesStddev);
[CASIA_TrainLBPPPM CASIA_FeaturesMu CASIA_FeaturesStddev ]= featureNormalize(CASIA_TrainLBPPPM);
CASIA_TestLBPPPM = featureNormalizeForTesting(CASIA_TestLBPPPM, CASIA_FeaturesMu, CASIA_FeaturesStddev);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. JPG: dis + glcm // 
CASIA_TrainData = [CASIA_TrainDistortion  CASIA_TrainGlcm];
CASIA_TestData = [CASIA_TestDistortion  CASIA_TestGlcm];

% 2. jpg: dis// ppm: LBP
CASIA_TrainData = [CASIA_TrainDistortion CASIA_TrainLBPPPM];
CASIA_TestData = [CASIA_TestDistortion  CASIA_TestLBPPPM];

% 3. jpg: dis // DW: LBP
CASIA_TrainData = [CASIA_TrainDistortion  CASIA_TrainDW];
CASIA_TestData = [CASIA_TestDistortion  CASIA_TestDW];

% 4. jpg: dis + glcm // ppm: LBP
CASIA_TrainData = [CASIA_TrainDistortion  CASIA_TrainGlcm CASIA_TrainLBPPPM];
CASIA_TestData = [CASIA_TestDistortion  CASIA_TestGlcm CASIA_TestLBPPPM];

% 5. jpg: dis + glcm // DW: LBP
CASIA_TrainData = [CASIA_TrainDistortion  CASIA_TrainGlcm CASIA_TrainDW];
CASIA_TestData = [CASIA_TestDistortion  CASIA_TestGlcm CASIA_TestDW];

% 6. DW: lbp // ppm: LBP
CASIA_TrainData = [CASIA_TrainDW CASIA_TrainLBPPPM ];
CASIA_TestData = [CASIA_TestDW CASIA_TestLBPPPM];

% 7. JPG: lbp // DW: LBP // PPM: LBP 
CASIA_TrainData = [CASIA_TrainLBP CASIA_TrainDW CASIA_TrainLBPPPM ];
CASIA_TestData = [CASIA_TestLBP CASIA_TestDW CASIA_TestLBPPPM];

% 8. JPG: dis + LBP // PPM: LBP // DW: LBP
CASIA_TrainData = [CASIA_TrainDistortion CASIA_TrainLBP  CASIA_TrainLBPPPM CASIA_TrainDW];
CASIA_TestData = [CASIA_TestDistortion CASIA_TestLBP CASIA_TestLBPPPM CASIA_TestDW];

% 9. JPG: dis + LBP
CASIA_TrainData = [CASIA_TrainDistortion CASIA_TrainLBP];
CASIA_TestData = [CASIA_TestDistortion CASIA_TestLBP];


% linear SVM
% Accuracy = 79.5222% (7157/9000) (classification) [ cut 25´Â ´õ ¶³¾îÁü], [glcm normalization, 82.9778%]
CASIA_Model_1 = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 1 -g 0.09 -t 2');
[CASIA_predict_label_1, CASIA_accuracy_1, CASIA_dec_values1] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_1);

CASIA_Model_2 = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 50 -g 0.0028125 -t 2');
[CASIA_predict_label_2, CASIA_accuracy_2, CASIA_dec_values2] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_2);

CASIA_Model_3 = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 10 -g 0.0091 -t 2');
[CASIA_predict_label_3, CASIA_accuracy_3, CASIA_dec_values3] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_3);

CASIA_Model_4 = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 32 -g 0.1250 -t 2');
[CASIA_predict_label_4, CASIA_accuracy_4, CASIA_dec_values4] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_4);

CASIA_Model_5 = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 40 -g 0.0098425 -t 2');
[CASIA_predict_label_5, CASIA_accuracy_5, CASIA_dec_values5] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_5);

% nogada
CASIA_Model_5 = svmtrain(CASIA_TrainLabel, CASIA_TrainData);
[CASIA_predict_label_5, CASIA_accuracy_5, CASIA_dec_values5] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_5);



CASIA_Model_Print = svmtrain(CASIA_TrainLabel, CASIA_TrainData, '-c 16 -g 0.000244140625000000');
[CASIA_predict_label_Print, CASIA_accuracy_Print, CASIA_dec_valuesPrint] = svmpredict(CASIA_TestLabel, CASIA_TestData, CASIA_Model_Print);

% HTER    numFrames = 30
CASIA_predict_label = CASIA_predict_label_Print;
[acc HTER ]= getHTER( CASIA_predict_label,  CASIA_TestLabel, numFrames);

% ROC
CASIA_dec_values = CASIA_dec_values5;
[X, Y, T, AUC] = perfcurve(CASIA_TestLabel, CASIA_dec_values, 1);
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('True positive rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == 0.01,
        fprintf('Y: %f \n', Y(i));
    end
end

%ERR
CASIA_dec_values = CASIA_dec_values5;
[X, Y, T, AUC] = perfcurve(CASIA_TestLabel, CASIA_dec_values, 1, 'XCrit', 'fpr', 'Ycrit', 'fnr');
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('False negative rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == Y(i),
        fprintf('X: %f Y: %f \n', X(i), Y(i));
    end
end





% auto find parameter
 [ resultAuto ] = findParameter( CASIA_TrainLabel, CASIA_TrainData, CASIA_TestLabel, CASIA_TestData, numFrames, 'A' );
% auto find parameter Fine
 [ resultAutoFine ] = findParameterFine( CASIA_TrainLabel, CASIA_TrainData, CASIA_TestLabel, CASIA_TestData, numFrames, 0, 0, 0, 0 ); 

%% case 3 - only Idiap Features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY
% JPG replay
Idiap_TrainData = [featuresIdiap.trainGenuine; featuresIdiap.trainReplay; ];
Idiap_TestData = [featuresIdiap.testGenuine; featuresIdiap.testReplay; ];
% DW replay
Idiap_TrainDataDW = [featuresIdiapDWLBP.trainGenuine; featuresIdiapDWLBP.trainReplay;];
Idiap_TestDataDW = [featuresIdiapDWLBP.testGenuine; featuresIdiapDWLBP.testReplay;];
% ppm replay
Idiap_TrainDataPPM = [featuresIdiapPPM.trainGenuine; featuresIdiapPPM.trainReplay;];
Idiap_TestDataPPM = [featuresIdiapPPM.testGenuine; featuresIdiapPPM.testReplay;];
% Label replay
Idiap_TrainLabel = [1*ones(size(featuresIdiap.trainGenuine, 1), 1); -1*ones(size(featuresIdiap.trainReplay, 1), 1)];
Idiap_TestLabel = [1*ones(size(featuresIdiap.testGenuine, 1), 1); -1*ones(size(featuresIdiap.testReplay, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- PRINTED
% JPG print
Idiap_TrainData = [featuresIdiap.trainGenuine; featuresIdiap.trainPrinted];
Idiap_TestData = [featuresIdiap.testGenuine; featuresIdiap.testPrinted];
% DW print
Idiap_TrainDataDW = [featuresIdiapDWLBP.trainGenuine; featuresIdiapDWLBP.trainPrinted];
Idiap_TestDataDW = [featuresIdiapDWLBP.testGenuine; featuresIdiapDWLBP.testPrinted];
% ppm print
Idiap_TrainDataPPM = [featuresIdiapPPM.trainGenuine; featuresIdiapPPM.trainPrinted];
Idiap_TestDataPPM = [featuresIdiapPPM.testGenuine; featuresIdiapPPM.testPrinted];
% Label print
Idiap_TrainLabel = [ones(size(featuresIdiap.trainGenuine, 1), 1); -1*ones(size(featuresIdiap.trainPrinted, 1), 1)];
Idiap_TestLabel = [ones(size(featuresIdiap.testGenuine, 1), 1); -1*ones(size(featuresIdiap.testPrinted, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -- REPLAY + PRINTED
% JPG replay + print
Idiap_TrainData = [featuresIdiap.trainGenuine; featuresIdiap.trainReplay; featuresIdiap.trainPrinted];
Idiap_TestData = [featuresIdiap.testGenuine; featuresIdiap.testReplay;  featuresIdiap.testPrinted];
% DW replay + print
Idiap_TrainDataDW = [featuresIdiapDWLBP.trainGenuine; featuresIdiapDWLBP.trainReplay; featuresIdiapDWLBP.trainPrinted];
Idiap_TestDataDW = [featuresIdiapDWLBP.testGenuine; featuresIdiapDWLBP.testReplay; featuresIdiapDWLBP.testPrinted];
% ppm replay + print
Idiap_TrainDataPPM = [featuresIdiapPPM.trainGenuine; featuresIdiapPPM.trainReplay; featuresIdiapPPM.trainPrinted];
Idiap_TestDataPPM = [featuresIdiapPPM.testGenuine; featuresIdiapPPM.testReplay; featuresIdiapPPM.testPrinted];
% Label replay + print
Idiap_TrainLabel = [ones(size(featuresIdiap.trainGenuine, 1), 1); -1*ones(size(featuresIdiap.trainReplay, 1), 1); -1*ones(size(featuresIdiap.trainPrinted, 1), 1)];
Idiap_TestLabel = [ones(size(featuresIdiap.testGenuine, 1), 1); -1*ones(size(featuresIdiap.testReplay, 1), 1); -1*ones(size(featuresIdiap.testPrinted, 1), 1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jpg (dis, glcm, lbp)
Idiap_TrainDistortion = Idiap_TrainData(:, 1:115);
Idiap_TrainGlcm = Idiap_TrainData(:, 116:203);
Idiap_TrainLBP = Idiap_TrainData(:, 204:1036);
Idiap_TestDistortion = Idiap_TestData(:, 1:115);
Idiap_TestGlcm = Idiap_TestData(:, 116:203);
Idiap_TestLBP = Idiap_TestData(:, 204:1036);

%dw (lbp)
Idiap_TrainDW = Idiap_TrainDataDW(:, 1:833);
Idiap_TestDW = Idiap_TestDataDW(:, 1:833);

%ppm (dis, glcm, lbp)
Idiap_TrainDistortionPPM = Idiap_TrainDataPPM(:, 1:115);
Idiap_TrainGlcmPPM = Idiap_TrainDataPPM(:, 116:203);
Idiap_TrainLBPPPM = Idiap_TrainDataPPM(:, 204:1036);
Idiap_TestDistortionPPM = Idiap_TestDataPPM(:, 1:115);
Idiap_TestGlcmPPM = Idiap_TestDataPPM(:, 116:203);
Idiap_TestLBPPPM = Idiap_TestDataPPM(:, 204:1036);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% distortion features normalization
[Idiap_TrainDistortion Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainDistortion);
Idiap_TestDistortion = featureNormalizeForTesting(Idiap_TestDistortion, Idiap_FeaturesMu, Idiap_FeaturesStddev);
% glcm features normalization
[Idiap_TrainGlcm Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainGlcm);
Idiap_TestGlcm = featureNormalizeForTesting(Idiap_TestGlcm, Idiap_FeaturesMu, Idiap_FeaturesStddev);
% LBP normalization
[Idiap_TrainLBP Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainLBP);
Idiap_TestLBP = featureNormalizeForTesting(Idiap_TestLBP, Idiap_FeaturesMu, Idiap_FeaturesStddev);

% DW normaization
[Idiap_TrainDW Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainDW);
Idiap_TestDW = featureNormalizeForTesting(Idiap_TestDW, Idiap_FeaturesMu, Idiap_FeaturesStddev);

% ppm normaization
[Idiap_TrainDistortionPPM Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainDistortionPPM);
Idiap_TestDistortionPPM = featureNormalizeForTesting(Idiap_TestDistortionPPM, Idiap_FeaturesMu, Idiap_FeaturesStddev);
[Idiap_TrainGlcmPPM Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainGlcmPPM);
Idiap_TestGlcmPPM = featureNormalizeForTesting(Idiap_TestGlcmPPM, Idiap_FeaturesMu, Idiap_FeaturesStddev);
[Idiap_TrainLBPPPM Idiap_FeaturesMu Idiap_FeaturesStddev ]= featureNormalize(Idiap_TrainLBPPPM);
Idiap_TestLBPPPM = featureNormalizeForTesting(Idiap_TestLBPPPM, Idiap_FeaturesMu, Idiap_FeaturesStddev);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. JPG: dis + glcm // 
Idiap_TrainData = [Idiap_TrainDistortion  Idiap_TrainGlcm];
Idiap_TestData = [Idiap_TestDistortion  Idiap_TestGlcm];

% 2. jpg: dis// ppm: LBP
Idiap_TrainData = [Idiap_TrainDistortion Idiap_TrainLBPPPM];
Idiap_TestData = [Idiap_TestDistortion  Idiap_TestLBPPPM];

% 3. jpg: dis // DW: LBP
Idiap_TrainData = [Idiap_TrainDistortion  Idiap_TrainDW];
Idiap_TestData = [Idiap_TestDistortion  Idiap_TestDW];

% 4. jpg: dis + glcm // ppm: LBP
Idiap_TrainData = [Idiap_TrainDistortion  Idiap_TrainGlcm Idiap_TrainLBPPPM];
Idiap_TestData = [Idiap_TestDistortion  Idiap_TestGlcm Idiap_TestLBPPPM];

% 5. jpg: dis + glcm // DW: LBP
Idiap_TrainData = [Idiap_TrainDistortion  Idiap_TrainGlcm Idiap_TrainDW];
Idiap_TestData = [Idiap_TestDistortion  Idiap_TestGlcm Idiap_TestDW];

% 6. DW: lbp // ppm: LBP
Idiap_TrainData = [Idiap_TrainDW Idiap_TrainLBPPPM ];
Idiap_TestData = [Idiap_TestDW Idiap_TestLBPPPM];

% 7. JPG: lbp // DW: LBP // PPM: LBP 
Idiap_TrainData = [Idiap_TrainLBP Idiap_TrainDW Idiap_TrainLBPPPM ];
Idiap_TestData = [Idiap_TestLBP Idiap_TestDW Idiap_TestLBPPPM];

% 8. JPG: dis + LBP // PPM: LBP // DW: LBP
Idiap_TrainData = [Idiap_TrainDistortion Idiap_TrainLBP  Idiap_TrainLBPPPM Idiap_TrainDW];
Idiap_TestData = [Idiap_TestDistortion Idiap_TestLBP Idiap_TestLBPPPM Idiap_TestDW];

% 9. JPG: dis + LBP
Idiap_TrainData = [Idiap_TrainDistortion Idiap_TrainLBP];
Idiap_TestData = [Idiap_TestDistortion Idiap_TestLBP];


% linear SVM
Idiap_Model_1 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 2 -g 0.0313 -t 2');
[Idiap_predict_label_1, Idiap_accuracy_1, Idiap_dec_values1] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_1);

Idiap_Model_2 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 32 -g 0.0313 -t 2');
[Idiap_predict_label_2, Idiap_accuracy_2, Idiap_dec_values2] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_2);

Idiap_Model_3 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 8 -g 0.0078125 -t 2');
[Idiap_predict_label_3, Idiap_accuracy_3, Idiap_dec_values3] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_3);

Idiap_Model_4 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 0.125 -g 0.0078125 -t 2');
[Idiap_predict_label_4, Idiap_accuracy_4, Idiap_dec_values4] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_4);

Idiap_Model_5 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 0.125 -g 0.0038125 -t 2');
[Idiap_predict_label_5, Idiap_accuracy_5, Idiap_dec_values5] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_5);

Idiap_Model_6 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 0.225 -g 0.00048125 -t 2');
[Idiap_predict_label_6, Idiap_accuracy_6, Idiap_dec_values6] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_6);

% nogada
Idiap_Model_5 = svmtrain(Idiap_TrainLabel, Idiap_TrainData, '-c 0.250000000000000 -g 6.10351562500000e-05');
[Idiap_predict_label_5, Idiap_accuracy_5, Idiap_dec_values5] = svmpredict(Idiap_TestLabel, Idiap_TestData, Idiap_Model_5);

% HTER
Idiap_predict_label = Idiap_predict_label_5;
[acc HTER ] = getHTER( Idiap_predict_label,  Idiap_TestLabel, numFrames);

% ROC
Idiap_dec_values = Idiap_dec_values5;
[X, Y, T, AUC] = perfcurve(Idiap_TestLabel, Idiap_dec_values, 1);
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('True positive rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == 0.01,
        fprintf('Y: %f \n', Y(i));
    end
end

%ERR
Idiap_dec_values = Idiap_dec_values5;
[X, Y, T, AUC] = perfcurve(Idiap_TestLabel, Idiap_dec_values, 1, 'XCrit', 'fpr', 'Ycrit', 'fnr');
AUC
figure
plot(X,Y)
% set(gca, 'XTick', [0.0001 : 0.1 : 10])
% axis([0 10 0 1])
xlabel('False positive rate')
ylabel('False negative rate')
%title('ROC for Classification by Logistic Regression')
for i = 1 : size(X, 1),
    if X(i) == Y(i),
        fprintf('X: %f Y: %f \n', X(i), Y(i));
    end
end


% auto find parameter
 [ resultAuto ] = findParameter( Idiap_TrainLabel, Idiap_TrainData, Idiap_TestLabel, Idiap_TestData, numFrames, 'A' );
% auto find parameter Fine
 [ resultAutoFine ] = findParameterFine( Idiap_TrainLabel, Idiap_TrainData, Idiap_TestLabel, Idiap_TestData, numFrames, 0, 0, 0, 0 ); 


 
 
 

