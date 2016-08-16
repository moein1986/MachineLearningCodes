function [J,grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
                                   
% nnCostFunction Implements the neural network cost function for a two layer
% neural network which performs classification

%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0; % initialize cost function

Theta1_grad = zeros(size(Theta1)); % initialize gradients
Theta2_grad = zeros(size(Theta2)); % initialize gradients

% ====================== Feedforward the neural network ======================


X = [ones(m, 1) X];
a1=X;
z2=X*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1),a2];
z3=a2*Theta2';
h=sigmoid(z3);
a3=h;

for ii=1:num_labels
    J=J+(-1/m)*((y==ii)'*log(h(:,ii))+(1-(y==ii))'*log(1-h(:,ii)));
end

% vectorized implementation of computing the cost function with regularization
J=J+(lambda/2/m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))); 

% ====================== backpropagation algorithm ======================

% the backpropagation algorithm is implemented to compute the gradients
%         Theta1_grad and Theta2_grad. 
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.


K=num_labels;
Logical_y=zeros(m,K); % mapping from output label vector to a binary vector
for ii=1:m
Logical_y(ii,y(ii))=1;
end

% computing the errors
delta3=a3-Logical_y;

delta2=(Theta2'*delta3').*sigmoidGradient([ones(m,1) z2]');
delta2(1,:)=[];

Delta1=delta2*a1;
Delta2=delta3'*a2;

Theta1_grad_TEMP=Delta1/m;
Theta2_grad_TEMP=Delta2/m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement regularization with the gradients.

Theta1_grad=Theta1_grad_TEMP+(lambda/m)*Theta1;
Theta1_grad(:,1)=Theta1_grad_TEMP(:,1);

Theta2_grad=Theta2_grad_TEMP+(lambda/m)*Theta2;
Theta2_grad(:,1)=Theta2_grad_TEMP(:,1);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
