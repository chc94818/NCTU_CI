%% Activation function tanh
function [Y]=actiFunc(X)
alpha=0.5;
Y=tanh(alpha.*X);
end