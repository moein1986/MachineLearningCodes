%% Machine Learning - Logistic Regression

% a logistic regression model is implemented to predict 
% whether a student gets admitted into a university.

%% Initialization
clear ; close all; clc

%% Load Data

%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Visualize the data ====================

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;


%% ============ Compute Cost Function and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


%% ============= Optimizing using fminunc  =============

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;


%% ============== Predict and Accuracies ==============

%  here the predict function will compute the training and test set accuracies of 
%  our model.



% Compute accuracy on the training set data1
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


