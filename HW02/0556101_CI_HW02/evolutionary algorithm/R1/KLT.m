%% kl轉換並回傳適應性 
% o 為點集平均
% v 為主要向量
function [o v e] = KLT(x)
        o = mean(x);
        x_shift =[ x(:,1)-o(1,1) , x(:,2)-o(1,2)];
        cov = x_shift'*x_shift;
        [V D] = eig(cov);
        max_eigenValue = max(abs( eig(cov)));
        ei =  find(abs(D/max_eigenValue)>=1);
        ei = 2 - mod(ei,2);
        v = V(:, ei );
        e=  sum(abs((x(:,1)-o(1))*v(2)-(x(:,2)-o(2))*v(1))/sqrt(v(1)*v(1)+v(2)*v(2)));
        
        
        
end