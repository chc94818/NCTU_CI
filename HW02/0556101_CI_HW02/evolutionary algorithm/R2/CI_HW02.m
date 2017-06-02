clear;
clc;
close all;
%% Load data
inputData=load('data\lineN200M3.txt');

%% Initial set
N=200;       % �I�ƶq
M= 3;          % ���O�ƶq
G = 14*M;    % ��]�ƶq(RHO�ݭn�Ω_�ƭ�BIT)  THETA = 9, RHO =5, 9+5 = 14
individualNum = 100; % ����ƶq
parentNum = individualNum/2;% ���N�ƶq
childNum = individualNum/2; % �l�N�ƶq
epoch=1000;                 % �t�ƽ���
mutationRate = 5; % ���ܾ��v
E = zeros(epoch); 
ClassResult = zeros(epoch);

individuals = zeros(individualNum ,  G);  
parents = zeros(parentNum ,  G);
children = zeros(childNum ,  G);

xx = 2*round(min(inputData(:,1))+1):0.1:round(max(inputData(:,1))+1)*2;
minIndices = 0;
classLine = zeros(M,size(xx,2));
individuals_fitness= zeros(individualNum,1);

%% ��l�ƭ���
individuals = round(rand(individualNum , G)*(1));


%% �t�Ʀ^��
for iter = 1:epoch
    %% �p��U����A����
     E(iter) = 1000000000;% ��̦n�A����
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
    %% �ƻs
    randIndices = randperm(individualNum);  % �H�����ðV�m���
    for ii = 1:parentNum
        e1 = individuals_fitness(randIndices(ii*2-1));
        e2 = individuals_fitness(randIndices(ii*2));
        %��ܿ��~�v���C���d�U�A�^�O���~�v����
        if(e1 < e2)
            parents(ii,:) = individuals(randIndices(ii*2-1),:);
        else
            parents(ii,:) = individuals(randIndices(ii*2),:);
        end    
    end
    %% ��t
    % crossover ���ͫ�N
    randIndices = randperm(parentNum);  % �H�����ðV�m���
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
    %% ����
    for ii = 1:childNum
        mutationPoint = round(rand(1,G)*1000);
        mutationIndices = find(mutationPoint<=mutationRate);
        children(ii,mutationIndices) = round(rand(1,size(mutationIndices,2))*(M-1)+1);
    end

    %% ��s
    individuals = [parents ; children];

end
%% ��ܵ��G
%�C��]�m
colorSet =[ 1,0,0;
                      0,1,0;
                      0,0,1;
                      0.5,0.5,0;
                      0.5,0,0.5;
                      0,0.5,0.5];
                  
% ������
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
% ���~�v���u
figure('name','error curve'), plot(E,'-b'), xlabel('epochs'), ylabel('E');
