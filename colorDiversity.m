function diversityFeature = colorDiversity(I)

% profile on;

% I = imread('real.jpg');

% profile report;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 32 level quantization, range is 1 to 32
quanti = (floor( double(I) ./ 8 ) + 1);
quantiR = reshape(quanti(:, :, 1), [], 1);
quantiG = reshape(quanti(:, :, 2), [], 1);
quantiB = reshape(quanti(:, :, 3), [], 1);
quantiBin = zeros(32, 32, 32);

for i = 1 : size(quantiR, 1),
    
    quantiBin(quantiR(i), quantiG(i), quantiB(i)) = quantiBin(quantiR(i), quantiG(i), quantiB(i)) + 1;
    
end

reQuantiBin = reshape(quantiBin(:, :, :), [], 1);
normBin = norm(reQuantiBin, 1);
reQuantiBin = sort( reQuantiBin ./ normBin, 'descend' );
% top 100 frequency feature ( 100 - d )
top100Freq =  reQuantiBin(1 : 100);

tmpValue = -1;
distCnt = 0;
for i = 1 : size(reQuantiBin, 1),    
    if tmpValue ~= reQuantiBin(i),
        distCnt = distCnt + 1;
        tmpValue = reQuantiBin(i);
    end    
    if tmpValue == 0,
        break;
    end
end

% not count 0. distCnt is distinct feature ( 1-d )
distCnt = distCnt - 1;
% normalization
distCnt = distCnt / size(reQuantiBin, 1);

diversityFeature = [top100Freq; distCnt];

end


