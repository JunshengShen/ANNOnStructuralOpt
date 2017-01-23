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
Y=reshape(Y,m,400);

%% ================ Part 2: Loading Parameters ================
theta1=randn(400,401)./100;
theta2=randn(400,401)./100;
nn_params= [theta1(:) ; theta2(:)];% Unroll parameters 

%% ================ Part 3: Compute Cost (Feedforward) ================
lambda = 0;
num_labels=400;
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda);
fprintf(['J= %f \n'], J);
nn_params=nn_params-grad.*0.01;
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda);
fprintf(['J= %f \n'], J); 
for i=1:50
  nn_params=nn_params-grad.*0.3;
  [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, Y, lambda);
  %fprintf(['J= %f \n'], J); 
end
fprintf(['J= %f \n'], J);
  
  
  
  
  
TESTX=load('trainingSetsX.txt');
TESTX=TESTX'(:);
TESTX=reshape(TESTX,m,400);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
a1 = [ones(m, 1) TESTX];  
z2 = a1 * Theta1';  
a2 = sigmoid(z2);  
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2'; 
h = sigmoid(z3);
h=reshape(h,20,20);
h=h';
imagesc(h),colorbar,colormap gray;