clear;
clc;
close all;
%% Load data
inputData=load('data\cross200.txt');

%% Initial
trainNum=195;       % 每一輪訓練數量
testNum = 200;       % 每一輪測試數量
epoch=1000;                 % 訓練輪數
inputNodeNum=2;     % 輸入層神經元數
hiddenNodeNum=10;  % 隱藏層神經元數
outputNodeNum=1;  % 輸出層神經元數
rho=0.1;                         % 學習率

trainData=inputData(:,1:2);     % 訓練資料
trainLabel=inputData(:,3);        % 訓練LABEL

% LABEL正規化到 -1 1
labelNormalization=find(trainLabel==2);   
trainLabel(labelNormalization)=-1;

% 初始化權重
weightHidden=rand(inputNodeNum+1,hiddenNodeNum)*2-1;
weightOutput=rand(hiddenNodeNum+1,outputNodeNum)*2-1;

err=zeros(epoch,trainNum);  % 錯誤評估
J=zeros(epoch,trainNum);      % 總錯誤評估
J_ave = zeros(epoch,1);            % 平均總錯誤評估
outputOutput = zeros(epoch,trainNum);            % 輸出值
outputOutput_acti = zeros(epoch,trainNum);    % 輸出值(經過激發函數)
bias=1.0;   % 偏移量

%% Trainning
  % 每一輪訓練
for iter=1:epoch 
    trainIndices = randperm(size(trainData,1));  % 隨機打亂訓練資料
      % 每一次輸入
    for ii=1:trainNum
        %% 前饋階段
        % 讀取資料
        dataBias=[bias,trainData(trainIndices(ii),:)];
        label=trainLabel(trainIndices(ii));
        
        % 隱藏層
        hiddenOutput=dataBias*weightHidden;
        hiddenOutput_acti=actiFunc(hiddenOutput);
        % 輸出層
        hiddenOutput_bias=[bias,hiddenOutput_acti];
        outputOutput(iter,ii)=hiddenOutput_bias*weightOutput;
        outputOutput_acti(iter,ii)=actiFunc(outputOutput(iter,ii));
        % 計算error
        err(iter,ii)=label-outputOutput_acti(iter,ii);
        J(iter,ii)=(err(iter,ii)^2)/2;
        
        %% 倒傳遞階段    
        %計算輸出層修正向量
        dJdY_back_output = err(iter,ii);
        dJdW_back_output=-1*dJdY_back_output*deFunc(outputOutput(iter,ii))*hiddenOutput_bias;
        
        %計算隱藏層修正向量
        dJdY_back_hidden=-1*dJdY_back_output*deFunc(outputOutput(iter,ii))*weightOutput(2:hiddenNodeNum+1,:);
        dJdW_back_hidden=zeros(inputNodeNum+1,hiddenNodeNum);
        for k=1:hiddenNodeNum
            dJdW_back_hidden(:,k)=dJdY_back_hidden(k)*deFunc(hiddenOutput(k))*dataBias';
        end
        % 更新權重
        weightHidden=weightHidden-(dJdW_back_hidden*rho);
        weightOutput=weightOutput-(dJdW_back_output'*rho);
    end
    J_ave(iter,1) = sum(J(iter,:))/trainNum;
end
    
    

%% Test
outputLabel=zeros(testNum,1);
testLabel=zeros(testNum,1);
testOutput = zeros(1,testNum);
testOutput_acti=zeros(1,testNum);
for ii=1:testNum

        dataBias=[bias,trainData(ii,:)];
        testLabel(ii) =trainLabel(ii);
        % 隱藏層
        hiddenOutput=dataBias*weightHidden;
        hiddenOutput_acti=actiFunc(hiddenOutput);
        % 輸出層
        hiddenOutput_bias=[bias,hiddenOutput_acti];
        testOutput(ii)=hiddenOutput_bias*weightOutput;
        testOutput_acti(ii)=actiFunc(testOutput(ii));  
    % classification
    if   testOutput_acti(ii)<=0
        outputLabel(ii)=-1;
    else
        outputLabel(ii)=1;
    end
end
c=0; % c=正確次數
for i=1:testNum
    if outputLabel(i)==testLabel(i);
        c=c+1;
    end
end

%% Print results 
% 設定
[X,Y] = meshgrid(-1: 0.01: 1, -1: 0.01 :1);
Z = zeros(201,201); % 激發數值
V = [ 0 :10];               % 等高線數量
grid = zeros(hiddenNodeNum,1); % 隱藏層node

% 計算每個座標的輸出值
for x = 1:testNum
    for y = 1:testNum
        for node = 1:hiddenNodeNum
            sum = X(x,y)*weightHidden(2,node)+Y(x,y)*weightHidden(3,node)-weightHidden(1,node);
            grid(node,1) = actiFunc(sum);
        end
        sum= 0;
        for node= 1:hiddenNodeNum
            sum = sum+grid(node,1)*weightOutput(node+1,1);
        end
        sum = sum-weightOutput(1,1);
        Z(x,y) = actiFunc(sum);
    end
end


% 錯誤率曲線
figure('name','learning curve'), plot(J_ave(:,1),'-b'), xlabel('epochs'), ylabel('J');

%正確率以及分群邊界
acurracy = c/testNum*100;
figure('name','classify result'), contour(X,Y,Z,V,'Linecolor','k');
title(['Accuracy: ',num2str(acurracy),'%'] );
hold on;
plot(inputData(1:100,1),inputData(1:100,2),'og', inputData(101:200,1),inputData(101:200,2), 'ob');



