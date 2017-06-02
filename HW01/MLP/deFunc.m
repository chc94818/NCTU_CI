%% derivative function of tanh
function [YY]=deFunc(X)
alpha=0.5;
YY=alpha*(1-actiFunc(X)^2);
end