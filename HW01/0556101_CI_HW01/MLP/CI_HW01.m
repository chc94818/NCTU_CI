clear;
clc;
close all;
%% Load data
inputData=load('data\cross200.txt');

%% Initial
trainNum=195;       % �C�@���V�m�ƶq
testNum = 200;       % �C�@�����ռƶq
epoch=1000;                 % �V�m����
inputNodeNum=2;     % ��J�h���g����
hiddenNodeNum=10;  % ���üh���g����
outputNodeNum=1;  % ��X�h���g����
rho=0.1;                         % �ǲ߲v

trainData=inputData(:,1:2);     % �V�m���
trainLabel=inputData(:,3);        % �V�mLABEL

% LABEL���W�ƨ� -1 1
labelNormalization=find(trainLabel==2);   
trainLabel(labelNormalization)=-1;

% ��l���v��
weightHidden=rand(inputNodeNum+1,hiddenNodeNum)*2-1;
weightOutput=rand(hiddenNodeNum+1,outputNodeNum)*2-1;

err=zeros(epoch,trainNum);  % ���~����
J=zeros(epoch,trainNum);      % �`���~����
J_ave = zeros(epoch,1);            % �����`���~����
outputOutput = zeros(epoch,trainNum);            % ��X��
outputOutput_acti = zeros(epoch,trainNum);    % ��X��(�g�L�E�o���)
bias=1.0;   % �����q

%% Trainning
  % �C�@���V�m
for iter=1:epoch 
    trainIndices = randperm(size(trainData,1));  % �H�����ðV�m���
      % �C�@����J
    for ii=1:trainNum
        %% �e�X���q
        % Ū�����
        dataBias=[bias,trainData(trainIndices(ii),:)];
        label=trainLabel(trainIndices(ii));
        
        % ���üh
        hiddenOutput=dataBias*weightHidden;
        hiddenOutput_acti=actiFunc(hiddenOutput);
        % ��X�h
        hiddenOutput_bias=[bias,hiddenOutput_acti];
        outputOutput(iter,ii)=hiddenOutput_bias*weightOutput;
        outputOutput_acti(iter,ii)=actiFunc(outputOutput(iter,ii));
        % �p��error
        err(iter,ii)=label-outputOutput_acti(iter,ii);
        J(iter,ii)=(err(iter,ii)^2)/2;
        
        %% �˶ǻ����q    
        %�p���X�h�ץ��V�q
        dJdY_back_output = err(iter,ii);
        dJdW_back_output=-1*dJdY_back_output*deFunc(outputOutput(iter,ii))*hiddenOutput_bias;
        
        %�p�����üh�ץ��V�q
        dJdY_back_hidden=-1*dJdY_back_output*deFunc(outputOutput(iter,ii))*weightOutput(2:hiddenNodeNum+1,:);
        dJdW_back_hidden=zeros(inputNodeNum+1,hiddenNodeNum);
        for k=1:hiddenNodeNum
            dJdW_back_hidden(:,k)=dJdY_back_hidden(k)*deFunc(hiddenOutput(k))*dataBias';
        end
        % ��s�v��
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
        % ���üh
        hiddenOutput=dataBias*weightHidden;
        hiddenOutput_acti=actiFunc(hiddenOutput);
        % ��X�h
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
c=0; % c=���T����
for i=1:testNum
    if outputLabel(i)==testLabel(i);
        c=c+1;
    end
end

%% Print results 
% �]�w
[X,Y] = meshgrid(-1: 0.01: 1, -1: 0.01 :1);
Z = zeros(201,201); % �E�o�ƭ�
V = [ 0 :10];               % �����u�ƶq
grid = zeros(hiddenNodeNum,1); % ���ühnode

% �p��C�Ӯy�Ъ���X��
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


% ���~�v���u
figure('name','learning curve'), plot(J_ave(:,1),'-b'), xlabel('epochs'), ylabel('J');

%���T�v�H�Τ��s���
acurracy = c/testNum*100;
figure('name','classify result'), contour(X,Y,Z,V,'Linecolor','k');
title(['Accuracy: ',num2str(acurracy),'%'] );
hold on;
plot(inputData(1:100,1),inputData(1:100,2),'og', inputData(101:200,1),inputData(101:200,2), 'ob');



