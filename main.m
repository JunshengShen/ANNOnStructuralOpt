clear ; close all; clc


%% =========== Part 1: Loading and Visualizing Data =============
input_layer_size  = 400;
hidden_layer_size = 400;
X=load('trainingSetsX.txt');
Y=load('trainingSetsY.txt');
X=X'(:);
Y=Y'(:);
%imagesc(Y),colorbar,colormap gray;
%imagesc(X),colorbar,colormap gray;
m = size(X, 1)/400;%the number of traning examples
X=reshape(X,m,400);


%% ================ Part 2: Loading Parameters ================
theta1=randn(400,401)./100;
theta2=randn(400,401)./100;
nn_params = [theta1(:) ; theta2(:)];% Unroll parameters 

%% ================ Part 3: Compute Cost (Feedforward) ================
lambda = 0;
num_labels=400;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda);