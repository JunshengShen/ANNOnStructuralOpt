function [j grand]=nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, Y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
a1 = [ones(m, 1) X];  
z2 = a1 * Theta1';  
a2 = sigmoid(z2);  
a2 = [ones(m, 1) a2];  
z3 = a2 * Theta2';  
h = sigmoid(z3);  
grand=0;
j=0;