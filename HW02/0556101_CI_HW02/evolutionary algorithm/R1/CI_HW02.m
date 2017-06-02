clear;
clc;
close all;
%% Load data
inputData=load('data\lineN100M4.txt');

%% Initial set
N=100;       % �I�ƶq
M= 4;          % ���O�ƶq
G = N;    % ��]�ƶq
individualNum = 100; % ����ƶq
parentNum = individualNum/2;% ���N�ƶq
childNum = individualNum/2; % �l�N�ƶq
epoch=1000;                 % �t�ƽ���
mutationRate = 5; % ���ܾ��v
E = zeros(epoch); 


individuals = zeros(individualNum ,  G);  
parents = zeros(parentNum ,  G);
children = zeros(childNum ,  G);

xx = 2*round(min(inputData(:,1))+1):0.1:round(max(inputData(:,1))+1)*2;
minIndices = 0;
classLine = zeros(individualNum,M,size(xx,2));
individuals_fitness= zeros(individualNum,1);

%% ��l�ƭ���
individuals = round(rand(individualNum , G)*(M-1)+1);


%% �t�Ʀ^��
for iter = 1:epoch
    %% �p��U����A����
    minimum = 1000000000;% ��̦n�A����
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

figure('name','classify result');
axis([-round(max(inputData(:,1))+1) round(max(inputData(:,1))+1) -round(max(inputData(:,2))+1) round(max(inputData(:,2))+1)]);
hold on;
for mi = 1:M    
    di = find(individuals(minIndices,:)==mi);     
    plot(inputData(di,1),inputData(di,2),'o','Color',colorSet(mi,:))
    yy=  reshape(classLine(minIndices,mi,:),1,size(classLine,3) );
    plot(xx,yy,'Color',colorSet(mi,:));
end
% ���~�v���u
figure('name','error curve'), plot(E,'-b'), xlabel('epochs'), ylabel('E');
