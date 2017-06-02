%CR = 分類結果，E = 錯誤率
function [CR E  P] = classFitness(data,individual)
gene = reshape(individual,size(individual,2)/14,14);%% gene length = 14
distance = zeros(size(data,1),size(gene,1));
[theta rho] = decode(gene);

for ii = 1:size(gene,1)
    coe_x = cos(theta(ii));
    coe_y = sin(theta(ii));
    distance(:,ii) = abs(data* [coe_x,coe_y]' + rho(ii))/ sqrt(coe_x*coe_x +coe_y*coe_y);
    P(ii,:) = [coe_x,coe_y,rho(ii)];
end
[mnA, ind] = min(distance, [], 2);

CR =ind';
E = sum(mnA)/size(data,1);
end

function [theta rho] = decode(gene)

theta = binary2Decimal (gene(:,1:9));  % 1~9 = theta
theta(find(theta>180)) =theta(find(theta>180))-360;
theta(find(theta < -180)) =theta(find(theta<-180))+180;
rho = binary2Decimal (gene(:,10:14));% 10~14 = rho
end


function D = binary2Decimal (B)
    b2d(1,1:size(B,2)-2) =2;
    b2d = [1,b2d];
    b2d = cumprod(b2d,2)';
    B(find(~B(:,end)),end)=-1;
    D = B(:,1:end-1)*b2d;
    D = D.*B(:,end);    
end