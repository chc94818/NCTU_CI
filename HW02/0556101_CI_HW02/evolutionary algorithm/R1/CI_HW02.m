clear;
clc;
close all;
%% Load data
inputData=load('data\lineN100M4.txt');

%% Initial set
N=100;       % 點數量
M= 4;          % 類別數量
G = N;    % 基因數量
individualNum = 100; % 個體數量
parentNum = individualNum/2;% 祖代數量
childNum = individualNum/2; % 子代數量
epoch=1000;                 % 演化輪數
mutationRate = 5; % 突變機率
E = zeros(epoch); 


individuals = zeros(individualNum ,  G);  
parents = zeros(parentNum ,  G);
children = zeros(childNum ,  G);

xx = 2*round(min(inputData(:,1))+1):0.1:round(max(inputData(:,1))+1)*2;
minIndices = 0;
classLine = zeros(individualNum,M,size(xx,2));
individuals_fitness= zeros(individualNum,1);

%% 初始化個體
individuals = round(rand(individualNum , G)*(M-1)+1);


%% 演化回數
for iter = 1:epoch
    %% 計算各個體適應性
    minimum = 1000000000;% 找最好適應性
    for ii = 1:individualNum
            eTemp = 0;
        for ci = 1:M
            di = find(individuals(ii,:)==ci);
            x = inputData(di,:);
            [o v e] = KLT(x);    
            classLine(ii,ci,:) = (xx-o(1))*v(2)/v(1)+o(2);
            eTemp = eTemp+e;
        end
        individuals_fitness(ii) =  eTemp/N;
        if( individuals_fitness(ii) < minimum)
            minimum =  individuals_fitness(ii);
            minIndices = ii;
        end
    end
    E(iter) = minimum;
    %% 複製
    randIndices = randperm(individualNum);  % 隨機打亂訓練資料
    for ii = 1:parentNum
        e1 = individuals_fitness(randIndices(ii*2-1));
        e2 = individuals_fitness(randIndices(ii*2));
        %選擇錯誤率較低的留下，淘汰錯誤率高的
        if(e1 < e2)
            parents(ii,:) = individuals(randIndices(ii*2-1),:);
        else
            parents(ii,:) = individuals(randIndices(ii*2),:);
        end    
    end
    %% 交配
    % crossover 產生後代
    randIndices = randperm(parentNum);  % 隨機打亂訓練資料
    for ii = 1:childNum/2
        crossIndices = randperm(G);
        crossIndices = crossIndices(1:G/2);
        c1 = parents(randIndices(ii*2-1),:);
        c2 = parents(randIndices(ii*2),:);
        c1(1,crossIndices) = parents(randIndices(ii*2),crossIndices);
        c2(1,crossIndices) = parents(randIndices(ii*2-1),crossIndices);
        children(ii*2-1,:)  = c1;
        children(ii*2,:)  = c2;

    end
    %% 突變
    for ii = 1:childNum
        mutationPoint = round(rand(1,G)*1000);
        mutationIndices = find(mutationPoint<=mutationRate);
        children(ii,mutationIndices) = round(rand(1,size(mutationIndices,2))*(M-1)+1);
    end

    %% 更新
    individuals = [parents ; children];

end
%% 顯示結果
%顏色設置
colorSet =[ 1,0,0;
                      0,1,0;
                      0,0,1;
                      0.5,0.5,0;
                      0.5,0,0.5;
                      0,0.5,0.5];
                  
% 分類圖

figure('name','classify result');
axis([-round(max(inputData(:,1))+1) round(max(inputData(:,1))+1) -round(max(inputData(:,2))+1) round(max(inputData(:,2))+1)]);
hold on;
for mi = 1:M    
    di = find(individuals(minIndices,:)==mi);     
    plot(inputData(di,1),inputData(di,2),'o','Color',colorSet(mi,:))
    yy=  reshape(classLine(minIndices,mi,:),1,size(classLine,3) );
    plot(xx,yy,'Color',colorSet(mi,:));
end
% 錯誤率曲線
figure('name','error curve'), plot(E,'-b'), xlabel('epochs'), ylabel('E');
