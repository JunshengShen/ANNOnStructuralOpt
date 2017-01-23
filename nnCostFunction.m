function [J grad]=nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, Y, lambda)
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
%Implement forward propagation

J = (1/m)* sum(sum(((-Y) .* log(h) - (1 - Y) .* log(1 - h))));
Theta1_new=Theta1(:,2:size(Theta1,2));  
Theta2_new=Theta2(:,2:size(Theta2,2));  
J=J+lambda/2/m*(Theta1_new(:)'*Theta1_new(:)+Theta2_new(:)'*Theta2_new(:));  
%Implement the cost function



for i=1:m  
    y_new=Y(i,:);  
    a1=[1;X(i,:)'];  
    a2=[1;sigmoid(Theta1*a1)];  
    a3=sigmoid(Theta2*a2);  
    det3=a3-y_new';  
    det2=Theta2'*det3.*sigmoidGradient([1;Theta1*a1]);  
    det2 = det2(2:end);   
    Theta1_grad=Theta1_grad+det2*a1';  
    Theta2_grad=Theta2_grad+det3*a2';  
end  
%step 3 and 4  
Theta1_grad(:,1)=Theta1_grad(:,1)/m;  
Theta1_grad(:,2:size(Theta1_grad,2))=Theta1_grad(:,2:size(Theta1_grad,2))/m+...  
    lambda*Theta1(:,2:size(Theta1,2))/m;  
Theta2_grad(:,1)=Theta2_grad(:,1)/m;  
Theta2_grad(:,2:size(Theta2_grad,2))=Theta2_grad(:,2:size(Theta2_grad,2))/m+...  
    lambda*Theta2(:,2:size(Theta2,2))/m;  
%Implement backpropagation



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


