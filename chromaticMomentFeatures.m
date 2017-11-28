function momentFeature = chromaticMomentFeatures(I)

% clear all;
% I is image
% I = imread('real.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hsv_image = rgb2hsv(I);

% calculate color moment
% another way to get the mean
% [sizeY, sizeX] = size(hsv_image( : , : , 1));
% sumH = sum(hsv_image(:, :, 1));
% sumH = sum(sumH, 2);
% meanH = sumH/(sizeY * sizeX);

reShapeH = reshape(hsv_image(:, :, 1), [], 1);
reShapeS = reshape(hsv_image(:, :, 2), [], 1);
reShapeV = reshape(hsv_image(:, :, 3), [], 1);
% reShapeR = reshape(I(:, :, 1), [], 1);
% reShapeG = reshape(I(:, :, 2), [], 1);
% reShapeB = reshape(I(:, :, 3), [], 1);

% calculate mean which is first moment
meanH = mean(reShapeH);
meanS = mean(reShapeS);
meanV = mean(reShapeV);

% calculate standard deviation which is second moment.
sdH = std(reShapeH);
sdS = std(reShapeS);
sdV = std(reShapeV);
% std2(hsv_image(:, :, 1))

% calculate skewness which is second moment.
skewH = skewness(reShapeH);
skewS = skewness(reShapeS);
skewV = skewness(reShapeV);

% maximal, minimal rate
arrCnt = size (reShapeH);
sortH = sort(reShapeH);
sortS = sort(reShapeS);
sortV = sort(reShapeV);

mostFreqCntH = 0;
leastFreqCntH = 0;
tmpMostFreqCntH = 0;
tmpMostFreqValueH = sortH(1);

mostFreqCntS = 0;
leastFreqCntS = 0;
tmpMostFreqCntS = 0;
tmpMostFreqValueS = sortS(1);

mostFreqCntV = 0;
leastFreqCntV = 0;
tmpMostFreqCntV = 0;
tmpMostFreqValueV = sortV(1);

for i = 1 : arrCnt(1),
    if tmpMostFreqValueH == sortH(i),
        tmpMostFreqCntH = tmpMostFreqCntH + 1;
    else
        if leastFreqCntH == 0,
            leastFreqCntH = tmpMostFreqCntH;
        else
            if leastFreqCntH > tmpMostFreqCntH,
                leastFreqCntH = tmpMostFreqCntH;
            end
        end
        tmpMostFreqValueH = sortH(i);
        tmpMostFreqCntH = 1;
    end    
    if mostFreqCntH < tmpMostFreqCntH,
        mostFreqCntH = tmpMostFreqCntH;
    end
    
    if tmpMostFreqValueS == sortS(i),
        tmpMostFreqCntS = tmpMostFreqCntS + 1;
    else
        if leastFreqCntS == 0,
            leastFreqCntS = tmpMostFreqCntS;
        else
            if leastFreqCntS > tmpMostFreqCntS,
                leastFreqCntS = tmpMostFreqCntS;
            end
        end
        tmpMostFreqValueS = sortS(i);
        tmpMostFreqCntS = 1;
    end    
    if mostFreqCntS < tmpMostFreqCntS,
        mostFreqCntS = tmpMostFreqCntS;
    end
    
    if tmpMostFreqValueV == sortV(i),
        tmpMostFreqCntV = tmpMostFreqCntV + 1;
    else
        if leastFreqCntV == 0,
            leastFreqCntV = tmpMostFreqCntV;
        else
            if leastFreqCntV > tmpMostFreqCntV,
                leastFreqCntV = tmpMostFreqCntV;
            end
        end
        tmpMostFreqValueV = sortV(i);
        tmpMostFreqCntV = 1;
    end    
    if mostFreqCntV < tmpMostFreqCntV,
        mostFreqCntV = tmpMostFreqCntV;
    end
    
end

if leastFreqCntH > tmpMostFreqCntH,
    leastFreqCntH = tmpMostFreqCntH;
end
if leastFreqCntS > tmpMostFreqCntS,
    leastFreqCntS = tmpMostFreqCntS;
end
if leastFreqCntV > tmpMostFreqCntV,
    leastFreqCntV = tmpMostFreqCntV;
end

% maximalRateH = mostFreqCntH * 100 / arrCnt(1);
% minimalRateH = leastFreqCntH * 100 / arrCnt(1);
% maximalRateS = mostFreqCntS * 100 / arrCnt(1);
% minimalRateS = leastFreqCntS * 100 / arrCnt(1);
% maximalRateV = mostFreqCntV * 100 / arrCnt(1);
% minimalRateV = leastFreqCntV * 100 / arrCnt(1);

maximalRateH = mostFreqCntH / arrCnt(1);
minimalRateH = leastFreqCntH / arrCnt(1);

maximalRateS = mostFreqCntS / arrCnt(1);
minimalRateS = leastFreqCntS / arrCnt(1);
maximalRateV = mostFreqCntV / arrCnt(1);
minimalRateV = leastFreqCntV / arrCnt(1);


% momentFeature = [meanH; meanS; meanV; sdH; sdS; sdV; skewH; skewS; skewV;
%     maximalRateH; minimalRateH; maximalRateS; minimalRateS; maximalRateV; minimalRateV];
% minimalRate는 거의 1을 가지기 때문에 비슷한 featrure가 된다. 그래서 제외해보자.
momentFeature = [meanH; meanS; meanV; sdH; sdS; sdV; skewH; skewS; skewV;
    maximalRateH; maximalRateS; maximalRateV; ];
end