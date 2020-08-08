function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
       
    % https://www.coursera.org/learn/machine-learning/supplement/U90DX/gradient-descent-for-linear-regression
    fprintf('**********\nDebug purposes:\nTheta computed from gradient descent:%f,%f',theta(1),theta(2))
    
    theta_temp_1 = theta(1) -  alpha * (1/m) * sum( X(:,1) .* (X * theta - y ) ) ;
    theta_temp_2 = theta(2) -  alpha * (1/m) * sum( X(:,2) .* (X * theta - y ) ) ;
    
    theta(1) = theta_temp_1;
    theta(2) = theta_temp_2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('\nIteration: %d \nCost: %f\n **********', iter, J_history(iter))

end

end
