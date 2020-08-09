function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
features_number = size(X,2);
theta_temp = theta;
for iter = 1:num_iters
    
    % https://www.coursera.org/learn/machine-learning/supplement/U90DX/gradient-descent-for-linear-regression
    %fprintf('**********\nDebug purposes:\nTheta computed from gradient descent:%f,%f',theta(1),theta(2))

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    for column = 1:features_number
        if column == 1
            % This way, we avoid one scalar multiplication even the result
            % is the same
            theta_temp(column) = theta(column) -  alpha * (1/m) * sum( (X * theta - y) );
        end
        theta_temp(column) = theta(column) -  alpha * (1/m) * sum( (X * theta - y) .* X(:,column) );        
    end
    theta = theta_temp;








    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %fprintf('\nIteration: %d \nCost: %f\n **********', iter, J_history(iter))

end

end
