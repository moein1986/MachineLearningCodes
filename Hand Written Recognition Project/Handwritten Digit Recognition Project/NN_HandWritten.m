%% Machine Learning - Neural Network Learning - Coursera Project

% This code implements one-vs-all logistic regression and neural
% netwroek to recognize hand-written digits.

%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % ("0" is mapped to label 10)

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


%% ================ Initializing Pameters ================
%  a two layer neural network that classifies digits is implemented. 
%  First, the neural network parameters will be randomely initialized (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Training NN ===================

% advanced optimizer "fmincg" will be used to train 
% our cost function "nnCostFunction"


fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 200);

lambda = 1; % regularization parameter 

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));




%% ================= Visualize Weights =================
%  Visualize what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

figure
displayData(Theta1(:, 2:end));



%% ================= Implement Predict =================
%  the neural network is used to predict the labels of the training set. This lets
%  us compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


