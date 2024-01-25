% 1.1 A Monte Carlo exercise
%% 1 
M = 5000; % number of simulations
T = 280; % length
Phi = 0.9; % AR(1) coefficient
y_mean = 3; % unconditional mean of y
std = 3; % unconditional std_dev of y

mu = y_mean * (1 - Phi);
gammasqr = (std^2) * (1 - Phi^2); % variance of epsilon
y = zeros(M, T);

rng(42) % set the seed for MATLAB's random number generator to the specific value of 42

for i = 1:M
        epsilon = 0 + sqrt(gammasqr) * randn(1, T); % generate a sequence of N(0,sigma^2)
    for j = 1:T-1 % generate yt's with realizations of epsilons
        y(i,1) = y_mean + epsilon(1);
        y(i,j + 1) = mu + Phi * y(i, j) + epsilon(j + 1);
    end
end

%% 2
mu_hat = zeros(M, 1); % vectors to store OLS estimates of mu's
phi_hat = zeros(M, 1);

for i = 1:M
    X = [ones(T-1, 1) y(i, 1:T-1)']; % OLS regressor for i-th simulation
    y_t = y(i, 2:T)'; 
    beta_hat = X\y_t; % Calculate coefficients (b0 and b1)
    mu_hat(i) = beta_hat(1);
    phi_hat(i) = beta_hat(2);
 
end

figure;
subplot(2, 1, 1);
histogram(mu_hat, 'Normalization', 'probability', 'EdgeColor', 'w'); % histogram of OLS estimates for mu
hold on; % retain the current plot and all its properties (like color, line style, etc.) and overlay the next plot on top of it
line([mu mu], [0 0.25], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-'); % plotting the true value
title('\mu Estimates');
xlabel('\mu');
legend('OLS Estimates', 'True Value');

subplot(2, 1, 2);
histogram(phi_hat, 'Normalization', 'probability', 'EdgeColor', 'w');
hold on;
line([Phi Phi], [0 0.25], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-');
title('\Phi Estimates');
xlabel('\Phi');
legend('OLS Estimates', 'True Value');

saveas(gcf, '1.1_2.png');



%% 3


M = 5000;   
phi_values = [0.90, 0.95, 0.97, 0.99];  
T_values = [40, 80, 120, 280];     
rng(42);  

unconditional_mean = 3;
unconditional_std = 3;

results_table = table();

for phi_idx = 1:length(phi_values) % Iterate all phi_values
        Phi = phi_values(phi_idx);
    
    for T_idx = 1:length(T_values) % Iterate all T values
        T = T_values(T_idx);
      
        mu = y_mean * (1 - Phi);
        gammasqr = (std^2) * (1 - Phi^2);
        y = zeros(M, T);
        
        % Create a matrix of simulated y's for each combination of Phi and T
        
        for i = 1:M
                epsilon = 0 + sqrt(gammasqr) * randn(1, T); % simulating epsilons
            for j = 1:T-1
                y(i,1) = y_mean + epsilon(1);
                y(i,j + 1) = mu + Phi * y(i, j) + epsilon(j + 1);
            end
        end 
        
        phi_hat = zeros(M, 1);
        
        % generate OLS estimates of phi and mu for each combination of Phi and T

        for i = 1:M % for each combination of phi and T iterrate through all simulated values of y
            X = [ones(T-1, 1) y(i, 1:T-1)'];
            y_t = y(i, 2:T)';
            beta_hat = X\y_t;
            phi_hat(i) = beta_hat(2);
         
        end % For y calculate the mean estimated phi_hat
            k = mean(phi_hat);
            result_entry = table(Phi, T, k);
            results_table = [results_table; result_entry];
    end
end
disp(results_table);

