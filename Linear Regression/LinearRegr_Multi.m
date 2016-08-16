%% Machine Learning - Linear regression with multiple variables 

% a linear regression model with multiple variables is implemented to predict
% the prices of houses in terms of their sizes and bedroom numbers.


%% ================ Initialization and Feature Normalization ================

clear ; close all; clc

fprintf('Loading data ...\n');

data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu ,sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Gradient Descent ================

% ====================== YOUR CODE HERE ======================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house

price = [1 (1650-mu(1))/sigma(1) (3-mu(2))/sigma(2) ]*theta; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

