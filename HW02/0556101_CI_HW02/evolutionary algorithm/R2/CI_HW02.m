clear;
clc;
close all;
%% Load data
inputData=load('data\lineN200M3.txt');

%% Initial set
N=200;       % 點數量
M= 3;          % 類別數量
G = 14*M;    % 基因數量(RHO需要用奇數個BIT)  THETA = 9, RHO =5, 9+5 = 14
individualNum = 100; % 個體數量
parentNum = individualNum/2;% 祖代數量
childNum = individualNum/2; % 子代數量
epoch=1000;                 % 演化輪數
mutationRate = 5; % 突變機率
E = zeros(epoch); 
ClassResult = zeros(epoch);

individuals = zeros(individualNum ,  G);  
parents = zeros(parentNum ,  G);
children = zeros(childNum ,  G);

xx = 2*round(min(inputData(:,1))+1):0.1:round(max(inputData(:,1))+1)*2;
minIndices = 0;
classLine = zeros(M,size(xx,2));
individuals_fitness= zeros(individualNum,1);

%% 初始化個體
individuals = round(rand(individualNum , G)*(1));


%% 演化回數
for iter = 1:epoch
    %% 計算各個體適應性
     E(iter) = 1000000000;% 找最好適應性
    for ii = 1:individualNum
        eTemp = 0;
         [CR Error P]   = classFitness(inputData,individuals(ii,:));
          individuals_fitness(ii) = Error;
         if(Error< E(iter))
             ClassResult = CR;
              E(iter) = Error;
              for mi = 1:M
                 classLine(mi,:)  = (-xx*P(mi,1)-P(mi,3))/P(mi,2);
              end
         end
          
    end
%     E(iter) = minimum;
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
% 
figure('name','classify result');
axis([round(min(inputData(:,1))+1) round(max(inputData(:,1))+1) round(min(inputData(:,2))+1) round(max(inputData(:,2))+1)]);
hold on;
for mi = 1:M    
    di = find(ClassResult(1,:)==mi);
    plot(inputData(di,1),inputData(di,2),'o','Color',colorSet(mi,:))
    yy=  reshape(classLine(mi,:),1,size(classLine,2) );
    plot(xx,yy,'Color',colorSet(mi,:));
end
% 錯誤率曲線
figure('name','error curve'), plot(E,'-b'), xlabel('epochs'), ylabel('E');
