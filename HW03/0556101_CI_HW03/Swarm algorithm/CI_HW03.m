clear;
clc;
close all;
%% Load data
contents = importdata('data\P3ds3.txt','%d');
n = contents(1);
inputData = zeros(n,n);
for ii = 2:n*n+1
    inputData(ii-1) = contents(ii);
end
inputData = inputData';
start.x = contents(n*n+2)+1;
start.y = contents(n*n+3)+1;
goal.x = contents(n*n+4)+1;
goal.y = contents(n*n+5)+1;

start.row = n-start.y+1;
start.col = start.x;
goal.row= n-goal.y+1;
goal.col = goal.x;

%% Initial set
pheMap = zeros(n,n,4);  % 費洛蒙地圖
heightMap = zeros(n,n,4); % 高度差地圖
% 初始化高度差
% 上
heightMap(:,:,1) = [-ones(1,n); abs(inputData(1:n-1,:)-inputData(2:n,:))+1 ];
% 下
heightMap(:,:,2) =[abs(inputData(1:n-1,:)-inputData(2:n,:))+1 ;-ones(1,n) ];
% 左
heightMap(:,:,3) = [-ones(n,1) abs(inputData(:,1:n-1)-inputData(:,2:n))+1 ];
% 右
heightMap(:,:,4) =  [abs(inputData(:,1:n-1)-inputData(:,2:n))+1 -ones(n,1)];


total_Num = 20000; % 收斂數量
ant_Num=1000;       % 螞蟻數量
alpha =1.4; % 費洛蒙權重
beta = 0.6; % 高度權重
gamma = 2;% 機率參數
decay = 0.05; % 費洛蒙衰退量
overlap = 0.8; % 費洛蒙覆蓋效率
length_bunus = 1.6; % 距離獎勵
deposit = 1000; % 費洛蒙排放量
p = zeros(1,4); % 方向機率
bestMap = zeros(n,n) ; % 最佳地圖
bestDirection={}; % 最佳走法
bestPath = {};
bestLength = 100000000;% 最佳走法距離;
path_cost = zeros(1,total_Num);
ave_Num = 50;
correct_ave = zeros(1,total_Num/ave_Num);


ant = struct('direction',{},'path',{},'length',{},'row',{},'col',{},'status',{},'map',{});
for ai = 1:ant_Num
    ant(ai).direction = {};
    ant(ai).path = {[ start.row start.col]};
    ant(ai).length= 0;
    ant(ai).row = start.row;
    ant(ai).col = start.col;
    ant(ai).status = 0;
    ant(ai).map = zeros(n,n);
    ant(ai).map(start.row,start.col)=1;
end



count = 0; %到達終點數量
%% run
run = true;
while(run)
    
    pheDeltaMap = zeros(n,n,4);  % 費洛蒙變化地圖
    %% each ant
    for ai = 1:ant_Num
        if(ant(ai).status == 0 )
            %% forward pass
            % selection of next node
            % 選擇方向機率
            for pi = 1:4
                if(heightMap(ant(ai).row,ant(ai).col,pi)==-1)
                    p(pi) = 0;
                else
                    p(pi) = (alpha*pheMap(ant(ai).row,ant(ai).col,pi)+ beta*1/heightMap(ant(ai).row,ant(ai).col,pi))^gamma;
                end
            end
            
            % 不走重複的路
            % 上
            if(p(1) ~=0 && ant(ai).map(ant(ai).row-1,ant(ai).col)~=0)
                p(1) = 0;
            end
            % 下
            if(p(2) ~=0 && ant(ai).map(ant(ai).row+1,ant(ai).col)~=0)
                p(2) = 0;
            end
            % 左
            if(p(3) ~=0 && ant(ai).map(ant(ai).row,ant(ai).col-1)~=0)
                p(3) = 0;
            end
            % 右
            if(p(4) ~=0 && ant(ai).map(ant(ai).row,ant(ai).col+1)~=0)
                p(4) = 0;
            end
            p_sum = cumsum(p);
            % 無路可走  重置狀態
            if(p_sum(4) == 0)
                ant(ai).direction = {};                
                ant(ai).path = {};
                ant(ai).length =0;
                ant(ai).row = start.row;
                ant(ai).col = start.col;
                ant(ai).status = 0;
                ant(ai).map = zeros(n,n);
                ant(ai).map(start.row,start.col)=1;
                continue;
            end
            
            % 選擇方向
            while(true)
                rand_value = rand()*(p_sum(4));
                di = find(p_sum>=rand_value);
                if(p(di(1)) ~=0)
                    break;
                end
            end
            
                           
            ant(ai).length =   ant(ai).length+heightMap(ant(ai).row,ant(ai).col,di(1));
            switch  di(1)
                % 上
                case 1     
                    ant(ai).row = ant(ai).row-1;
                    % 下
                case 2
                    ant(ai).row = ant(ai).row+1;
                    % 左
                case 3
                    ant(ai).col = ant(ai).col-1;
                    % 右
                case 4
                    ant(ai).col = ant(ai).col+1;
            end
            
            % 儲存走過的路            
            ant(ai).map(ant(ai).row,ant(ai).col) =  size(  ant(ai).direction,2)+1;
            ant(ai).direction = [di(1) ant(ai).direction];
            ant(ai).path = [ant(ai).path [ant(ai).row ant(ai).col] ];
            
            %抵達終點
            if(ant(ai).row == goal.row && ant(ai).col == goal.col)
                count = count +1;
               path_cost(count) =  ant(ai).length ;
                
                ant(ai).status = 1;
                % 紀錄新的最佳解
                if( ant(ai).length<bestLength)
                    bestLength =  ant(ai).length;
                    bestMap =  ant(ai).map;
                    bestDirection =  ant(ai).direction;
                    bestPath = ant(ai).path;
                end
                % 統計抵達終點的螞蟻數量
                if(count >= total_Num)
                    run = false;
                    break;
                end
            end
        else
            %% backward pass
            % back-trace the remembered direction
            lastdirection = cell2mat(ant(ai).direction(1));
            switch  lastdirection
                % 上
                case 1
                    ant(ai).row = ant(ai).row+1;
                    % 下
                case 2
                    ant(ai).row = ant(ai).row-1;
                    % 左
                case 3
                    ant(ai).col = ant(ai).col+1;
                    % 右
                case 4
                    ant(ai).col = ant(ai).col-1;
            end
            ant(ai).direction = ant(ai).direction(2:end);
            
            % deposit pheromon
            pheDeltaMap(ant(ai).row,ant(ai).col,lastdirection) =  overlap*pheDeltaMap(ant(ai).row,ant(ai).col,lastdirection) + deposit/heightMap(ant(ai).row,ant(ai).col,lastdirection)*1/ant(ai).length^length_bunus;
            
            % 回到起點  重置再出發
            if(size(ant(ai).direction,2) == 0)
                ant(ai).direction = {};                
                ant(ai).path = {};
                ant(ai).length =0;
                ant(ai).row = start.row;
                ant(ai).col = start.col;
                ant(ai).status = 0;
                ant(ai).map = zeros(n,n);
                ant(ai).map(start.row,start.col)=1;
            end
            
        end
    end
    pheMap = (1-decay) *  pheMap + pheDeltaMap;
end

disp(count);

%% 顯示結果
figure('name','Map');
hold on;
[x,y]=meshgrid(1:1:n,1:1:n);          % the grid points
phe = sqrt(flipud(sum(pheMap,3)))*100;
contourf(x,y,phe,10);  % 地圖等高線圖
contour(x,y,inputData,10); % 費洛蒙濃度圖
temp = cell2mat(bestPath);
temp2 = reshape(temp,2,size(temp,2)/2);
lineCord.x = [start.x temp2(2,:)];
lineCord.y = [start.y n-temp2(1,:)+1];
% 標出起點、終點
 plot(start.x,start.y,'r*')     
 plot(goal.x,goal.y,'g*')
 % 標出最佳路線
line(lineCord.x', lineCord.y', 'color', 'r', 'linewidth', 2);
s = 'ants' 
% 長度曲線
figure('name','length curve'), plot(path_cost,'-b'), xlabel('ants'), ylabel('L');


% 長度優值曲線
correct = bestLength./path_cost(1:count);
correct_ave = reshape(correct,ave_Num,total_Num/ave_Num);
correct_curve = sum(correct_ave)./ave_Num*100;
figure('name','correct curve'), plot(correct_curve,'-b'), xlabel(['ants (' int2str( ave_Num) ')']), ylabel('C'),axis([ 1 total_Num/ave_Num 0 100]);

