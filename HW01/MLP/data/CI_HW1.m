clc;
clear;

% import data
v = importdata('cross200.txt');
%v = importdata('elliptic200.txt');
%v = importdata('iris.txt',',');
%v = importdata('cross200.txt');
v = v(randperm(size(v,1)),:);

L = 2;                                      % 糷计
k_r = [10,1];                               % neuron计秖
num_feature = length(v(1,:)) - 1;           % 疭紉计秖
num_pattern = length(v(:,1));               % 戈计秖
num_hidden_neuron = 10;
x = v(:,1:num_feature).';                   % input data
target_output = v(:,num_feature+1).';       % data class
hidden_weight = randn(num_feature,num_hidden_neuron);
output_weight = randn(num_hidden_neuron,1);
y = zeros(10,num_pattern);
out = zeros(1,num_pattern);
error = zeros(1,num_pattern);

a = 3;
error_sum = 0;
% class = 2 , f > 0.5 , class = 1 , f < 0.5
for i = 1:num_pattern
    y(:,i) = hidden_weight.' * x(:,i);
    f = 1 ./ (1 + exp(-a .* y(:,i)));
    out(:,i) = f.' * output_weight;
    f = 1 ./ (1 + exp(-a .* out(:,i)));
    %fprintf('%f \n',f);
    if (f > 0.5 && target_output(:,i) == 1) || (f < 0.5 && target_output(:,i) == 2)
        error(i) = f - target_output(:,i);
    end
    error_sum = error_sum + power(error(i),2);
end

v_rj = 0;
lo = 0.2;

for time = 1:30
    for i = L:-1:1
        if i == L
            for j = 1:num_pattern
                v_rj = y(:,j).' * output_weight;
                J_W = -error(1,j) * (-power((1+exp(a * v_rj)),-2) * (a * exp(a * v_rj))) * y(1,j);
                output_weight = output_weight - lo * J_W;
            end
        else
            for j = 1:num_pattern
                v_rj = y(:,j).' * output_weight;
                J_W = -error(1,j) * (-power((1+exp(a * v_rj)),-2) * (a * exp(a * v_rj))) * y(1,j);
                for k = 1:k_r(i)
                    v_rj = x(:,j).' * hidden_weight(:,k);
                    J_W = J_W + -error(1,j) * (-power((1+exp(a * v_rj)),-2) * (a * exp(a * v_rj))) * x(1,j);
                end
                hidden_weight = hidden_weight - lo * J_W;
            end
        end
    end

    error_times = 0;
    for i = 1:num_pattern
        y(:,i) = hidden_weight.' * x(:,i);
        f = 1 ./ (1 + exp(-a .* y(:,i)));
        out(:,i) = f.' * output_weight;
        f = 1 ./ (1 + exp(-a .* out(:,i)));
        if (f>0.5 && target_output(:,i) == 1) || (f<0.5 && target_output(:,i) == 2)
            error(i) = f - target_output(:,i);
            error_times = error_times + 1;
        else
            error(i) = 0;
        end
        error_sum = error_sum + power(error(i),2);
    end
    fprintf('%f \n',error_times/num_pattern);
end