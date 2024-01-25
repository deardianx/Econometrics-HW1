% 1.2 Forecasting Under Structural Breaks

%% Parameters
T = 280;
phi = 0.9;
sigma_y = 3;
sigma_e = sqrt((sigma_y^2) * (1 - phi^2));

t_series = datetime(1948, 3, 31):calmonths(3):datetime(2017, 12, 31);

%% Generate mu series
mu_series = zeros(T, 1);
for t = 1:T
    if t_series(t) < datetime(1973, 1, 1)
        mu_series(t, 1) = 5 * (1 - phi);
    elseif t_series(t) < datetime(1996, 1, 1)
        mu_series(t, 1) = 0 * (1 - phi);
    elseif t_series(t) < datetime(2005, 1, 1)
        mu_series(t, 1) = 4.5 * (1 - phi);
    elseif t_series(t) < datetime(2018, 1, 1)
        mu_series(t, 1) = 0.5 * (1 - phi);
    end
end

%% Generate ytrue series
rng(8)
ytrue = zeros(T, 1);
ytrue(1, 1) = mu_series(1, 1);
for t = 1:T
    et = normrnd(0, sigma_e);
    if t == 1
        y0 = mu_series(t, 1) / (1 - phi); % set y0 to the unconditional mean
        yt = mu_series(t, 1) + phi * y0 + et;
    else
        yt = mu_series(t, 1) + phi * ytrue(t-1, 1) + et;
    end
    ytrue(t, 1) = yt;
end

%% Plot ytrue and unconditional mean
hold on
legend
plot(t_series, mu_series / (1 - phi), 'DisplayName','Unconditional Mean');
plot(t_series, ytrue, 'DisplayName','Ytrue');
hold off

%% Researcher 1: constant mean (mu) + expanding window
end_est = sum(t_series < datetime(1990, 1, 1)); % The t index of the end of estimation period 1989Q4
start_pred = end_est + 1; % The t index of the start of prediction period 1990Q1
pred_period = 12;

for t = end_est:(T-1)

    % Estimate with all past data points
    X = ytrue(1:(t-1), 1);
    Y = ytrue(2:t, 1);
    model1 = fitlm(X, Y);

    % Predict
    ypred1 = predAR1(model1, ytrue(t, 1), pred_period);
    if t == end_est
        Ypred1 = ypred1;
    else
        Ypred1 = [Ypred1 ypred1];
    end
end

%% Researcher 2: time-varying mean (mu) + rolling window
for t = end_est:(T-1)

    % Estimate with the past 40 data points
    X = ytrue((t-40):(t-1), 1);
    Y = ytrue((t-39):t, 1);
    model2 = fitlm(X, Y);

    % Predict
    ypred2 = predAR1(model2, ytrue(t, 1), pred_period);
    if t == end_est
        Ypred2 = ypred2;
    else
        Ypred2 = [Ypred2 ypred2];
    end
end

%% Researcher 3: Random Walk
for t = end_est:(T-1)
   
    % Predict using the most recent value
    ypred3 = ones(pred_period, 1) * ytrue(t, 1);
    if t == end_est
        Ypred3 = ypred3;
    else
        Ypred3 = [Ypred3 ypred3];
    end
end

%% Researcher 4: True Model
for t = end_est:(T-1)

    % Predict using the most recent value and the true model(mu and phi)
    ypred4 = zeros(pred_period, 1);
    for pt = 1:pred_period
        % if predict t > T, use NaN since there's no mu value
        if t + pt > T 
            ypred4(pt, 1) = NaN;
            continue
        end
        % Iterate pt make prediction
        if pt == 1
            yprev = ytrue(t, 1);
        else
            yprev = ypred4(pt-1, 1);
        end
        ypred4(pt, 1) = mu_series(t+pt, 1) + phi * yprev; 
    end

    if t == end_est
        Ypred4 = ypred4;
    else
        Ypred4 = [Ypred4 ypred4];
    end
end


%% True y Values 
for t = end_est:(T-1)

    % Predict using the most recent value and the true model(mu and phi)
    ytrue5 = zeros(pred_period, 1);
    for pt = 1:pred_period
        % if predict t > T, use NaN since there's no mu value
        if t + pt > T 
            ytrue5(pt, 1) = NaN;
            continue
        end
        % Iterate tp make prediction
        ytrue5(pt, 1) = ytrue(t+pt, 1); 
    end

    if t == end_est
        Ytrue5 = ytrue5;
    else
        Ytrue5 = [Ytrue5 ytrue5];
    end
end


%% Compare Prediction Accuracy
[mae1, rmse1] = calcAccuracy(Ypred1, Ytrue5);
[mae2, rmse2] = calcAccuracy(Ypred2, Ytrue5);
[mae3, rmse3] = calcAccuracy(Ypred3, Ytrue5);
[mae4, rmse4] = calcAccuracy(Ypred4, Ytrue5);

mae_ratio1 = mae1 ./ mae4;
mae_ratio2 = mae2 ./ mae4;
mae_ratio3 = mae3 ./ mae4;

rmse_ratio1 = rmse1 ./ rmse4;
rmse_ratio2 = rmse2 ./ rmse4;
rmse_ratio3 = rmse3 ./ rmse4;


%% Plot MAE amd RMSE
tiledlayout(2, 1)

% % Ypred
% ax1 = nexttile;
% plot(ax1, 1:pred_period, Ypred1(:,1), 'DisplayName','Researcher 1: OLS + Expanding Window (Constant Mean)')
% hold on
% plot(ax1, 1:pred_period, Ypred2(:,1), 'DisplayName','Researcher 2: OLS + Rolling Window (Time-varying Mean)')
% plot(ax1, 1:pred_period, Ypred3(:,1), 'DisplayName','Researcher 3: Random Walk')
% plot(ax1, 1:pred_period, Ypred4(:,1), 'DisplayName','Researcher 4: True Model')
% plot(ax1, 1:pred_period, Ytrue5(:,1), 'DisplayName','Researcher 5: Ture Y')
% 
% title(ax1, 'Ypred')
% grid(ax1,'on')
% hold off
% legend
% ax1.FontSize = 18;

% MAE
ax1 = nexttile;
plot(ax1, 1:pred_period, mae_ratio1, 'DisplayName','Researcher 1: OLS + Expanding Window (Constant Mean)')
hold on
plot(ax1, 1:pred_period, mae_ratio2, 'DisplayName','Researcher 2: OLS + Rolling Window (Time-varying Mean)')
plot(ax1, 1:pred_period, mae_ratio3, 'DisplayName','Researcher 3: Random Walk')
title(ax1, 'MAE')
grid(ax1,'on')
hold off
legend
ax1.FontSize = 18;

% RMSE
ax2 = nexttile;
plot(ax2, 1:pred_period, rmse_ratio1, 'DisplayName','Researcher 1: OLS + Expanding Window (Constant Mean)')
hold on
plot(ax2, 1:pred_period, rmse_ratio2, 'DisplayName','Researcher 2: OLS + Rolling Window (Time-varying Mean)')
plot(ax2, 1:pred_period, rmse_ratio3, 'DisplayName','Researcher 3: Random Walk')
title(ax2, 'RMSE')
grid(ax2,'on')
hold off
legend
ax2.FontSize = 18;

% Function to calculate MAE and RMSE
% Input: predictions and true values
function [mae, rmse] = calcAccuracy(Ypred, Ytrue)
    Yerror = abs(Ypred - Ytrue);
    mae = nanmean(Yerror, 2);
    rmse = sqrt(nanmean(Yerror.^2, 2));
end

% Function to predict using AR(1)
% Here we input a regression model, and one newx, then we will do rolling
% prediction for pred_periods using the predicted y
function pred = predAR1(model, newX, pred_period)
    ypred = zeros(pred_period, 1);
    for t = 1:pred_period
        if t == 1 % use the provided new x
            newx = newX;
        else % use the predicted y
            newx = ypred(t-1, 1);
        end
        ypred(t, 1) = predict(model, newx);
    end
    pred = ypred;
end

