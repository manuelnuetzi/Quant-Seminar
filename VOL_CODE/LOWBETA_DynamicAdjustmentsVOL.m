% CountryLowBeta: Computes the returns from equally weighted and
% beta-based portfolios on country ETFs
clc
clear
close all
% Parameter selection
% Annualization factor for monthly to annual
annualizationFactor = 12;
% Number of countries held long and short (nShorts is ignored in the
% long-only versions)
% Proportional transaction costs AND Borrowing Cost 
tCost = 0.001;
borrowingRate = 0.02; % annual borrowing cost
financingSpread = 0.0005; % additional financing spread for long-short strat
% Trading lag in 
lag = 1;
% Beta computation lookback period in months. Note that we must use 59 and not 60 months here.
lookbackStart = 59;
nLongs = 5
nShorts = 5

% In-sample and out-of-sample periods
inSampleEnd = datetime(2014,12,31);
outSampleStart = datetime(2015,1,1);


% Load the data and extract the dates, riskless rate, and ETF prices
data_country = readtable('merged_data.xlsx', 'Format', 'auto', 'TreatAsEmpty', {'#N/A N/A'});
data_vix = readtable('VOLINDEX.xlsx', 'Format', 'auto', 'TreatAsEmpty', {'#N/A N/A'});
data_mxwo = readtable('msciallworld.xlsx', 'Format', 'auto', 'TreatAsEmpty', {'#N/A N/A'});
data_RF = readtable('Riskfree_USD3Month.xlsx', 'Format', 'auto', 'TreatAsEmpty', {'#N/A N/A'});

% load factors
% qmj and bab
data_bab_qmj_umd= readtable('BAB_QMJ_UMD.xlsx', 'Format', 'auto', 'TreatAsEmpty', {'#N/A N/A'});
%load ff5 factors
data_ff5 = readtable('F-F-Research_Data_5_Factors_2x3.xlsx');
data_ff5{:, 2:end} = data_ff5{:, 2:end} / 100;


% Define the date formats 
dateFormat = 'dd.MM.yyyy'; % For data_country and data_mxwo
% Convert dates to datetime format
data_country.Date = datetime(data_country.Date, 'InputFormat', dateFormat);
data_RF.Date = datetime(data_RF.Date, 'InputFormat', dateFormat);
data_vix.Date = datetime(data_vix.Date, 'InputFormat', dateFormat);
data_mxwo.Date = datetime(data_mxwo.Date, 'InputFormat', dateFormat);
data_bab_qmj_umd.Date = datetime(data_bab_qmj_umd.Date, 'InputFormat', 'MM/dd/yyyy'); % Adjust format if needed

% Convert numeric YYYYMM to string and then to datetime (initially first day of month)
data_ff5.Date = datetime(num2str(data_ff5.Date), 'InputFormat', 'yyyyMM', 'Format', 'yyyy-MM-dd');
% Adjust to the last day of the respective month
data_ff5.Date = dateshift(data_ff5.Date, 'end', 'month');

% If dates load as datetimes, one can also use readtimetable instead of
% readtable, then the table2timetable commands below can be skipped.
data_country = table2timetable(data_country);
data_vix = table2timetable(data_vix);
data_mxwo = table2timetable(data_mxwo);
data_bab_qmj_umd = table2timetable(data_bab_qmj_umd);
data_ff5 = table2timetable(data_ff5);
data_RF = table2timetable(data_RF);


mergedTable = synchronize(data_country, data_vix, data_mxwo, data_RF, 'first');

%%

% sync factors
mergedFactors_monthly = synchronize(data_bab_qmj_umd, data_ff5, 'first');

% Generate arrays with the different variables matched to the previous sync before
country_prices = table2array(mergedTable(:, 1 : 28));
mxwo_prices = table2array(mergedTable(:, 30));
vix = table2array(mergedTable(:, 29));
Rf = table2array(mergedTable(:, 31))/100;

%fill N/A values of vix values with previous
vix = fillmissing(vix, 'previous');
country_prices = fillmissing(country_prices,'previous');
Rf = fillmissing(Rf,'previous');
mxwo_prices = fillmissing(mxwo_prices,'previous');


%fill N/A values of vix values with previous
vix = fillmissing(vix, 'previous');

% Check for NaNs in each dataset
missing_VIX = any(isnan(vix), 'all');
missing_MXWO = any(isnan(mxwo_prices), 'all');
missing_Rf = any(isnan(Rf), 'all');


% Anzahl der Werte größer als xx (just for idea)
num_values = sum(vix > 18);

% Ergebnis anzeigen
fprintf('Anzahl der Werte in VIX, die größer als xx sind: %d\n', num_values);

% create dates based on merged table
dates = mergedTable.Date;

% Size of the dataset
nDays = length(dates)
nAssets = width(country_prices)

% Generate numeric dates in the format YYYYMMDD so that we can reuse 
% the functions we developed previously. 
datesYyyymmdd = yyyymmdd(dates);

% Compute the return earned on the riskless asset on each trading day, accounting for
% the number of calendar days until the next trading day. Shift the result
% down by one day so that it represents the return accrued on that day
dayCount = days(diff(dates));
% create rfscaled
RfScaled = zeros(nDays, 1);
RfScaled(2 : end, 1) = Rf(1 : end - 1, 1) .* dayCount / 360;

% Debug RfScaled
if ~isnumeric(RfScaled) || any(~isfinite(RfScaled))
    error('RfScaled contains non-numeric or non-finite values');
end

% Compute daily country returns
countryreturns = zeros(nDays, nAssets);
countryreturns(2 : end, :) = country_prices(2 : end, :) ./ country_prices(1 : end - 1, :) - 1;
countryXsReturns = countryreturns - RfScaled;

% Compute daily mxwo returns
mxworeturns = zeros(nDays, 1);
mxworeturns(2 : end, :) = mxwo_prices(2 : end, :) ./ mxwo_prices(1 : end - 1, :) - 1;

% Treasury-Rate als Risk-Free-Rate verwendet.
mxwoXsReturns = mxworeturns - RfScaled;


% Compute cross-sectional average excess returns on the ETFs for beta estimation
% market_country = The xsreturns of the cross sectional average of all countries
market_country = mean(countryXsReturns, 2, "omitnan");
% market_mxwo = The xsreturn of the MSCI All World Index
market_mxwo = mxwoXsReturns;

% beta construction 2: Obtain monthly beta estimates to construct portfolio weights
% First, generate arrays listing the first and last day of each month
[firstDayList, lastDayList] = getFirstAndLastDayInPeriod(datesYyyymmdd, 2);
monthlyDates = dates(lastDayList);
nMonths = length(firstDayList)
monthlyBetas_mxwo = zeros(nMonths, nAssets);
% Second, estimate the betas using five-year trailing windows
firstMonth = lookbackStart + 1;
for m = firstMonth : nMonths
    first = firstDayList(m - lookbackStart);
    last = lastDayList(m) - lag;
    X = [ones(last - first + 1, 1), market_mxwo(first : last, 1)];
    Y = countryXsReturns(first : last, :);
    b = X \ Y;
    monthlyBetas_mxwo(m, :) = b(2, :);    
end

%% Weight calculation based on VIX Leverage Adjustment, only based on current VIX
% VIX thresholds and leverage multipliers
vixThresholds = [15, 25, 40];
leverageMultipliers = [1.5, 1.0, 0.5, 0.0]; % [VIX <15, 15≤VIX≤25, 25<VIX≤40, VIX>40]

% Construct the static portfolios weights for MSCI All World betas
equalWeights_mxwo = zeros(nMonths, nAssets);
lowBetaWeights_mxwo = zeros(nMonths, nAssets);
highBetaWeights_mxwo = zeros(nMonths, nAssets);
longShortWeights_mxwo = zeros(nMonths, nAssets);
marketNeutralWeights_mxwo = zeros(nMonths, nAssets);

% Initialisieren Sie alle benötigten Vektoren
leverage_History = zeros(nMonths, 1);
longShortAdj = zeros(nMonths, 1);

for m = firstMonth:nMonths
    % Portfolio-Gewichte berechnen
    nonMissings_mxwo = isfinite(country_prices(lastDayList(m),:));
    equalWeights_mxwo(m,:) = nonMissings_mxwo / sum(nonMissings_mxwo);
    
    lowBetaWeights_mxwo(m,:) = computeSortWeights(monthlyBetas_mxwo(m,:), nLongs, 0, 0);
    highBetaWeights_mxwo(m,:) = computeSortWeights(monthlyBetas_mxwo(m,:), nLongs, 0, 1);
    
    % Dynamische Leverage-Anpassung
    currentVIX = vix(lastDayList(m));
    
    if currentVIX < vixThresholds(1)
        leverage_History(m) = leverageMultipliers(1);
        longShortAdj(m) = 1;
    elseif currentVIX <= vixThresholds(2)
        leverage_History(m) = leverageMultipliers(2);
        longShortAdj(m) = 1.1;
    elseif currentVIX <= vixThresholds(3)
        leverage_History(m) = leverageMultipliers(3);
        longShortAdj(m) = 1;
    else
        leverage_History(m) = leverageMultipliers(4);
        longShortAdj(m) = 1.4;
    end
    
    % Leverage anwenden
    lowBetaWeights_mxwo(m,:) = lowBetaWeights_mxwo(m,:) * leverage_History(m);
    highBetaWeights_mxwo(m,:) = highBetaWeights_mxwo(m,:) * leverage_History(m);
    
    % Long-Short Portfolio
    longShortWeights_mxwo(m,:) = computeSortWeights(monthlyBetas_mxwo(m,:), nLongs, nShorts, 0);
    longShortWeights_mxwo(m,:) = longShortWeights_mxwo(m,:) * longShortAdj(m);
    
    % Market-Neutral Portfolio
    longBeta_mxwo = max(0.1, sum(lowBetaWeights_mxwo(m,:) .* monthlyBetas_mxwo(m,:), "omitnan"));
    shortBeta_mxwo = max(0.1, sum(highBetaWeights_mxwo(m,:) .* monthlyBetas_mxwo(m,:), "omitnan"));
    
    marketNeutralWeights_mxwo(m,:) = (lowBetaWeights_mxwo(m,:) * longShortAdj(m)) / longBeta_mxwo - ...
                                    (highBetaWeights_mxwo(m,:) / longShortAdj(m)) / shortBeta_mxwo;
end

% Data cleaning continued
% Compute monthly returns on the assets. 
monthlyDayCount = days(diff(monthlyDates));
% Same as: monthlyDayCount = days(monthlyDates(2 : end, 1) - monthlyDates(1 : end - 1, 1));
% Note that we need to use the rates at month-end. Compounding RfScaled would be wrong.
monthEndRf = Rf(lastDayList); 
monthlyRf = zeros(nMonths, 1);
monthlyRf(2 : end, 1) = monthEndRf(1 : end - 1, 1) .* monthlyDayCount / 360;
% Now do the ETFs. To handle delistings that take place during the month, just replace NaNs with zeros.
countryreturns(isnan(countryreturns)) = 0;
monthlyTotalReturns_country = aggregateReturns(countryreturns, datesYyyymmdd, 2);
monthlyXsReturns_country = monthlyTotalReturns_country - monthlyRf;

% doublecheck if factors match the monthly dates so we can drop the first months series in next step
% Suppose monthlyDates is N×1 and mergedFactors_monthly has N rows as well.
% 1) Check if they are the same length:
if numel(monthlyDates) ~= height(mergedFactors_monthly)
    error('They have different lengths, so they cannot match one-to-one.');
end

% 2) Compare year and month only:
sameYearMonth = (year(monthlyDates) == year(mergedFactors_monthly.Date)) & ...
                (month(monthlyDates) == month(mergedFactors_monthly.Date));

% 3) Check if all match:
if all(sameYearMonth)
    disp('All monthlyDates match mergedFactors_monthly by year & month (ignoring day).');
else
    disp('Some entries do NOT match by year & month. See indices below:');
    find(~sameYearMonth)
end

% Drop firstMonth months from the arrays. In order to have weights and
% returns in sync, one drops firstMonth months from the beginning of the 
% return series, and (firstMonth - 1) months from the beginning and one 
% Drop initial months for synchronization
monthlyDates = monthlyDates(firstMonth+1:end);
monthlyTotalReturns_country = monthlyTotalReturns_country(firstMonth+1:end, :);
monthlyRf = monthlyRf(firstMonth+1:end, :);
monthlyXsReturns_country = monthlyXsReturns_country(firstMonth+1:end, :);
monthlyFactors = mergedFactors_monthly(firstMonth+1:end, :);
monthlyFactorsArray = table2array(monthlyFactors(:, 1:8));
equalWeights_mxwo = equalWeights_mxwo(firstMonth:end-1, :);
lowBetaWeights_mxwo = lowBetaWeights_mxwo(firstMonth:end-1, :);
highBetaWeights_mxwo = highBetaWeights_mxwo(firstMonth:end-1, :);
longShortWeights_mxwo = longShortWeights_mxwo(firstMonth:end-1, :);
marketNeutralWeights_mxwo = marketNeutralWeights_mxwo(firstMonth:end-1, :);

monthlyBetas_mxwo = monthlyBetas_mxwo(firstMonth:end-1,:);
nMonths = nMonths - firstMonth;


%% From here we split the sample
%pre split is/oos betas (i just put in the numbers for the split (1:165),
%maybe adjust this later so its based on variable)

% in sample period split
% betas
monthlyBetas_mxwo_IS = monthlyBetas_mxwo(1:165,:);
% rest
insampleIdx = monthlyDates <= inSampleEnd;
monthlyDates_IS             = monthlyDates(insampleIdx);
monthlyTotalReturns_country_IS = monthlyTotalReturns_country(insampleIdx, :);
monthlyRf_IS                 = monthlyRf(insampleIdx, :);
monthlyXsReturns_country_IS = monthlyXsReturns_country(insampleIdx, :);
monthlyFactors_IS            = monthlyFactors(insampleIdx, :);

equalWeights_mxwo_IS         = equalWeights_mxwo(insampleIdx, :);
lowBetaWeights_mxwo_IS       = lowBetaWeights_mxwo(insampleIdx, :);
highBetaWeights_mxwo_IS      = highBetaWeights_mxwo(insampleIdx, :);
longShortWeights_mxwo_IS     = longShortWeights_mxwo(insampleIdx, :);
marketNeutralWeights_mxwo_IS = marketNeutralWeights_mxwo(insampleIdx, :);

nMonths_IS                   = sum(insampleIdx);

% IN SAMPLE RETURN COMPUTATION / PERF STATS / PLOT
% Compute strategy returns without transaction costs. 
% Working with excess returns is easier.
nStrategies = 5
%
stratXsReturnsNoTC_msci_IS = zeros(nMonths_IS, nStrategies);
stratXsReturnsNoTC_msci_IS(:, 1) = sum(monthlyXsReturns_country_IS .* equalWeights_mxwo_IS, 2);
stratXsReturnsNoTC_msci_IS(:, 2) = sum(monthlyXsReturns_country_IS .* lowBetaWeights_mxwo_IS, 2);
stratXsReturnsNoTC_msci_IS(:, 3) = sum(monthlyXsReturns_country_IS .* highBetaWeights_mxwo_IS, 2);
stratXsReturnsNoTC_msci_IS(:, 4) = sum(monthlyXsReturns_country_IS .* longShortWeights_mxwo_IS, 2);
stratXsReturnsNoTC_msci_IS(:, 5) = sum(monthlyXsReturns_country_IS .* marketNeutralWeights_mxwo_IS, 2);


% Performance statistics (benchmark is the mxwo)
strategyNames_mxwo_IS = {'ew_mxwo', 'lowbeta_mxwo','highbeta_mxwo', 'longshort_mxwo', 'bab_mxwo'};
summarizePerformance(stratXsReturnsNoTC_msci_IS, monthlyRf_IS, stratXsReturnsNoTC_msci_IS(:, 1), annualizationFactor, strategyNames_mxwo_IS, 'perf_mxwobeta_IS');

% Compute cumulative returns
cum_returns_msci_IS = cumprod(1 + stratXsReturnsNoTC_msci_IS);

% Define strategy names
strategyNames_msci = {'EW MSCI', 'Low Beta MSCI', 'High Beta MSCI', 'Long-Short MSCI', 'Market Neutral MSCI'};

% Define colors for better visibility
colors = lines(length(strategyNames_msci)); % Use MATLAB's default color map for clarity
% Plot MSCI-based strategies
figure;
hold on;
set(gcf, 'Position', [100, 100, 1200, 600]); % Adjust figure size

for i = 1:size(cum_returns_msci_IS, 2)
    if i == 1
        plot(monthlyDates_IS, cum_returns_msci_IS(:, i), '-', 'Color', colors(i, :), 'LineWidth', 2); % EW strategy solid
    else
        plot(monthlyDates_IS, cum_returns_msci_IS(:, i), '--', 'Color', colors(i, :), 'LineWidth', 1.5); % Others dashed
    end
end

% Improve readability
xlabel('Date', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative Return', 'FontSize', 14, 'FontWeight', 'bold');
title('In-Sample Cumulative Performance of MSCI-Based Strategies', 'FontSize', 16, 'FontWeight', 'bold');


set(gca, 'FontSize', 12);
grid on;
box on;

% Improve tick formatting
datetick('x', 'yyyy', 'keeplimits'); 

% Legend settings
legend(strategyNames_msci, 'Location', 'NorthWest', 'FontSize', 12, 'Box', 'off');

% Finalize plot
hold off;

% Improve tick formatting
datetick('x', 'yyyy', 'keeplimits'); 

% Legend settings
legend(strategyNames_msci, 'Location', 'NorthWest', 'FontSize', 12, 'Box', 'off');

% Finalize plot
hold off;


%% Heatmap of Sharpe Ratios for Varying nLongs and nShorts (Corrected)
% Define grid for number of longs and shorts
nLongs_grid = 1:8;
nShorts_grid = 1:8;

% Preallocate matrices for Sharpe ratios
sharpe_ew = zeros(length(nLongs_grid), length(nShorts_grid));
sharpe_lowbeta = zeros(length(nLongs_grid), length(nShorts_grid));
sharpe_highbeta = zeros(length(nLongs_grid), length(nShorts_grid));
sharpe_longshort = zeros(length(nLongs_grid), length(nShorts_grid));
sharpe_marketneutral = zeros(length(nLongs_grid), length(nShorts_grid));

% Loop over grid values
for i = 1:length(nLongs_grid)
    for j = 1:length(nShorts_grid)
        current_nLongs = nLongs_grid(i);
        current_nShorts = nShorts_grid(j);
        
        % Preallocate strategy returns (shifted by 1 month)
        nMonths_usable = nMonths - 1;
        ret_ew = zeros(nMonths_usable, 1);
        ret_lowbeta = zeros(nMonths_usable, 1);
        ret_highbeta = zeros(nMonths_usable, 1);
        ret_longshort = zeros(nMonths_usable, 1);
        ret_marketneutral = zeros(nMonths_usable, 1);
        
        % Loop over months, ensuring no look-ahead
        for m = 1:nMonths_usable
            % Get current month's beta and non-missing assets
            currentBeta = monthlyBetas_mxwo(m, :);
            nonMissings = ~isnan(currentBeta);
            
            if sum(nonMissings) < current_nLongs
                continue; % Skip if not enough valid assets
            end
            
            % --- Compute base weights for next month's returns ---
            % Equal-weighted
            w_ew = nonMissings / sum(nonMissings);
            
            % Low-beta (ascending sort)
            w_low = computeSortWeights(currentBeta, current_nLongs, 0, 0);
            
            % High-beta (descending sort)
            w_high = computeSortWeights(currentBeta, current_nLongs, 0, 1);
            
            % Long-short
            w_longshort = computeSortWeights(currentBeta, current_nLongs, current_nShorts, 0);
            
            % Market-neutral
            longBeta = sum(w_low .* currentBeta, 'omitnan');
            shortBeta = sum(w_high .* currentBeta, 'omitnan');
            if shortBeta == 0 || longBeta == 0
                continue; % Avoid divide by zero
            end
            w_marketneutral = (w_low / longBeta) - (w_high / shortBeta);
            
            % --- Dynamic Leverage Adjustment ---
            currentVIX = vix(lastDayList(m + firstMonth - 1)); % Adjust index for in-sample
            
            if currentVIX < vixThresholds(1)
                leverage = leverageMultipliers(1);
                longShort_Adj = 1;
            elseif currentVIX <= vixThresholds(2)
                leverage = leverageMultipliers(2);
                longShort_Adj = 1.1;
            elseif currentVIX <= vixThresholds(3)
                leverage = leverageMultipliers(3);
                longShort_Adj = 1;
            else
                leverage = leverageMultipliers(4);
                longShort_Adj = 1.4;
            end
            
            % Apply leverage
            w_ew = w_ew * leverage;
            w_low = w_low * leverage;
            w_high = w_high * leverage;
            w_longshort = (w_low * longShort_Adj) - (w_high / longShort_Adj);
            w_marketneutral = (w_low * longShort_Adj / longBeta) - (w_high / longShort_Adj / shortBeta);
            
            % --- Apply to NEXT month's returns ---
            ret_ew(m) = sum(monthlyXsReturns_country(m+1, :) .* w_ew);
            ret_lowbeta(m) = sum(monthlyXsReturns_country(m+1, :) .* w_low);
            ret_highbeta(m) = sum(monthlyXsReturns_country(m+1, :) .* w_high);
            ret_longshort(m) = sum(monthlyXsReturns_country(m+1, :) .* w_longshort);
            ret_marketneutral(m) = sum(monthlyXsReturns_country(m+1, :) .* w_marketneutral);
        end
        
        % Compute Sharpe Ratios
        annualizedReturn = @(x) mean(x, 'omitnan') * annualizationFactor;
        annualizedVol = @(x) std(x, 'omitnan') * sqrt(annualizationFactor);
        
        % Ensure no NaN Sharpe Ratios
        sharpe_ew(i,j) = annualizedReturn(ret_ew) / max(annualizedVol(ret_ew), 1e-6);
        sharpe_lowbeta(i,j) = annualizedReturn(ret_lowbeta) / max(annualizedVol(ret_lowbeta), 1e-6);
        sharpe_highbeta(i,j) = annualizedReturn(ret_highbeta) / max(annualizedVol(ret_highbeta), 1e-6);
        sharpe_longshort(i,j) = annualizedReturn(ret_longshort) / max(annualizedVol(ret_longshort), 1e-6);
        sharpe_marketneutral(i,j) = annualizedReturn(ret_marketneutral) / max(annualizedVol(ret_marketneutral), 1e-6);
    end
end

% plot

figure;
tiledlayout(2, 3);

% Get global color limits for consistency
all_sharpes = [sharpe_ew(:); sharpe_lowbeta(:); sharpe_highbeta(:); 
               sharpe_longshort(:); sharpe_marketneutral(:)];
clim = [min(all_sharpes), max(all_sharpes)];

% Define strategies and their data
strategies = {
    'Equal Weighted',       sharpe_ew;
    'Low Beta',            sharpe_lowbeta;
    'High Beta',           sharpe_highbeta;
    'Long-Short',          sharpe_longshort;
    'Market Neutral',      sharpe_marketneutral;
};

% Plot each heatmap
for i = 1:size(strategies, 1)
    nexttile;
    h = heatmap(nShorts_grid, nLongs_grid, strategies{i,2}, 'Colormap', parula);
    h.ColorLimits = clim;
    title(strategies{i,1});
    xlabel('nShorts');
    ylabel('nLongs');
    
    % Format cell text to 1 decimal place
    h.CellLabelFormat = '%.2f'; 
end

% Adjust figure
sgtitle('Sharpe Ratios (1 Decimal Place)', 'FontSize', 14, 'FontWeight', 'bold');
set(gcf, 'Position', [100, 100, 1200, 800]);

%% OUT OF SAMPLE RETURN COMPUTATION / PERF STATS / PLOT
%oos betas
monthlyBetas_country_OOS = monthlyBetas_mxwo(166:end,:);
% oosample period split
oosampleIdx = monthlyDates >= outSampleStart;
monthlyDates_OOS              = monthlyDates(oosampleIdx);
monthlyTotalReturns_country_OOS = monthlyTotalReturns_country(oosampleIdx, :);
monthlyRf_OOS                = monthlyRf(oosampleIdx, :);
monthlyXsReturns_country_OOS = monthlyXsReturns_country(oosampleIdx, :);
monthlyFactors_OOS            = monthlyFactors(oosampleIdx, :);

equalWeights_mxwo_OOS         = equalWeights_mxwo(oosampleIdx, :);
lowBetaWeights_mxwo_OOS       = lowBetaWeights_mxwo(oosampleIdx, :);
highBetaWeights_mxwo_OOS     = highBetaWeights_mxwo(oosampleIdx, :);
longShortWeights_mxwo_OOS     = longShortWeights_mxwo(oosampleIdx, :);
marketNeutralWeights_mxwo_OOS = marketNeutralWeights_mxwo(oosampleIdx, :);

nMonths_OOS                   = sum(oosampleIdx);

% Compute strategy returns without transaction costs. 
% Working with excess returns is easier.
nStrategies = 5
%
stratXsReturnsNoTC_msci_OOS = zeros(nMonths_OOS, nStrategies);
stratXsReturnsNoTC_msci_OOS(:, 1) = sum(monthlyXsReturns_country_OOS .* equalWeights_mxwo_OOS, 2);
stratXsReturnsNoTC_msci_OOS(:, 2) = sum(monthlyXsReturns_country_OOS .* lowBetaWeights_mxwo_OOS, 2);
stratXsReturnsNoTC_msci_OOS(:, 3) = sum(monthlyXsReturns_country_OOS .* highBetaWeights_mxwo_OOS, 2);
stratXsReturnsNoTC_msci_OOS(:, 4) = sum(monthlyXsReturns_country_OOS .* longShortWeights_mxwo_OOS, 2);
stratXsReturnsNoTC_msci_OOS(:, 5) = sum(monthlyXsReturns_country_OOS .* marketNeutralWeights_mxwo_OOS, 2);


% Performance statistics (benchmark is the mxwo)
strategyNames_mxwo_OSS = {'ew_mxwo', 'lowbeta_mxwo','highbeta_mxwo', 'longshort_mxwo', 'bab_mxwo'};
summarizePerformance(stratXsReturnsNoTC_msci_OOS, monthlyRf_OOS, stratXsReturnsNoTC_msci_OOS(:, 1), annualizationFactor, strategyNames_mxwo_OSS, 'perf_mxwobeta_OOS');

% plot code
% EW ARE FOR BOTH COUNTRY AND MSCI THE SAME
% Compute cumulative returns

cum_returns_msci_OOS = cumprod(1 + stratXsReturnsNoTC_msci_OOS);

% Define strategy names
strategyNames_msci = {'EW MSCI', 'Low Beta MSCI', 'High Beta MSCI', 'Long-Short MSCI', 'Market Neutral MSCI'};

% Define colors for better visibility
colors = lines(length(strategyNames_msci)); % Use MATLAB's default color map for clarity

% Open figure and hold
figure;
hold on;
set(gcf, 'Position', [100, 100, 1200, 600]); % Adjust figure size

% Plot MSCI-based strategies
for i = 1:size(cum_returns_msci_OOS, 2)
    if i == 1
        plot(monthlyDates_OOS, cum_returns_msci_OOS(:, i), '-', 'Color', colors(i, :), 'LineWidth', 2); % EW strategy solid
    else
        plot(monthlyDates_OOS, cum_returns_msci_OOS(:, i), '--', 'Color', colors(i, :), 'LineWidth', 1.5); % Others dashed
    end
end

% Improve readability
xlabel('Date', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative Return', 'FontSize', 14, 'FontWeight', 'bold');
title('Out-Of-Sample Cumulative Performance of MSCI-Based Strategies', 'FontSize', 16, 'FontWeight', 'bold');

% Remove 'YScale','log' to keep linear scale
set(gca, 'FontSize', 12); % Only sets font size, no log scaling
grid on;
box on;

% Improve tick formatting
datetick('x', 'yyyy', 'keeplimits'); % Show years on x-axis

% Legend settings
legend(strategyNames_msci, 'Location', 'NorthWest', 'FontSize', 12, 'Box', 'off');

% Finalize plot
hold off;


%% Backtest over whole period (IS and OOS) + plot --> to compare to Strategy Performance without Leverage Adjustment

% Compute strategy returns without transaction costs. 
% Working with excess returns is easier.
nStrategies = 5
%
stratXsReturnsNoTC_msci_all = zeros(nMonths, nStrategies);
stratXsReturnsNoTC_msci_all(:, 1) = sum(monthlyXsReturns_country .* equalWeights_mxwo, 2);
stratXsReturnsNoTC_msci_all(:, 2) = sum(monthlyXsReturns_country .* lowBetaWeights_mxwo, 2);
stratXsReturnsNoTC_msci_all(:, 3) = sum(monthlyXsReturns_country .* highBetaWeights_mxwo, 2);
stratXsReturnsNoTC_msci_all(:, 4) = sum(monthlyXsReturns_country .* longShortWeights_mxwo, 2);
stratXsReturnsNoTC_msci_all(:, 5) = sum(monthlyXsReturns_country .* marketNeutralWeights_mxwo, 2);


% Performance statistics (benchmark is the mxwo)
strategyNames_mxwo_all= {'ew_mxwo', 'lowbeta_mxwo','highbeta_mxwo', 'longshort_mxwo', 'bab_mxwo'};
summarizePerformance(stratXsReturnsNoTC_msci_all, monthlyRf, stratXsReturnsNoTC_msci_all(:, 1), annualizationFactor, strategyNames_mxwo_all, 'perf_mxwobeta_all');

% plot code
% EW ARE FOR BOTH COUNTRY AND MSCI THE SAME
% Compute cumulative returns

cum_returns_msci_all = cumprod(1 + stratXsReturnsNoTC_msci_all);

% Define strategy names
strategyNames_msci = {'EW MSCI', 'Low Beta MSCI', 'High Beta MSCI', 'Long-Short MSCI', 'Market Neutral MSCI'};

% Define colors for better visibility
colors = lines(length(strategyNames_msci)); % Use MATLAB's default color map for clarity

% Open figure and hold
figure;
hold on;
set(gcf, 'Position', [100, 100, 1200, 600]); % Adjust figure size

% Plot MSCI-based strategies
for i = 1:size(cum_returns_msci_all, 2)
    if i == 1
        plot(monthlyDates, cum_returns_msci_all(:, i), '-', 'Color', colors(i, :), 'LineWidth', 2); % EW strategy solid
    else
        plot(monthlyDates, cum_returns_msci_all(:, i), '--', 'Color', colors(i, :), 'LineWidth', 1.5); % Others dashed
    end
end

% Improve readability
xlabel('Date', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative Return', 'FontSize', 14, 'FontWeight', 'bold');
title('Cumulative Performance of MSCI-Based Strategies (2000-2025)', 'FontSize', 16, 'FontWeight', 'bold');

% Remove 'YScale','log' to keep linear scale
set(gca, 'FontSize', 12); % Only sets font size, no log scaling
grid on;
box on;

% Improve tick formatting
datetick('x', 'yyyy', 'keeplimits'); % Show years on x-axis

% Legend settings
legend(strategyNames_msci, 'Location', 'NorthWest', 'FontSize', 12, 'Box', 'off');

% Finalize plot
hold off;


%% Implement Transaction Cost on whole Backtest period 

turnover = zeros(nMonths, nStrategies);
for m = 1 : nMonths - 1
    currentRf = monthlyRf(m, 1);
    currentRet = monthlyTotalReturns_country(m, :);
    turnover(m, 1) = computeTurnover(equalWeights_mxwo(m, :), equalWeights_mxwo(m + 1, :), currentRet, currentRf);
    turnover(m, 2) = computeTurnover(lowBetaWeights_mxwo(m, :), lowBetaWeights_mxwo(m + 1, :), currentRet, currentRf); 
    turnover(m, 3) = computeTurnover(highBetaWeights_mxwo(m, :), highBetaWeights_mxwo(m + 1, :), currentRet, currentRf);
    turnover(m, 4) = computeTurnover(longShortWeights_mxwo(m, :), longShortWeights_mxwo(m + 1, :), currentRet, currentRf);
    turnover(m, 5) = computeTurnover(marketNeutralWeights_mxwo(m, :), marketNeutralWeights_mxwo(m + 1, :), currentRet, currentRf);    
end
% This is splitting hair a little bit. We're adding the transactions in the initial month.
turnover(1, 1) = turnover(1, 1) + sum(abs(equalWeights_mxwo(1, :))); 
turnover(1, 2) = turnover(1, 2) + sum(abs(lowBetaWeights_mxwo(1, :))); 
turnover(1, 3) = turnover(1, 3) + sum(abs(highBetaWeights_mxwo(1, :))); 
turnover(1, 4) = turnover(1, 4) + sum(abs(longShortWeights_mxwo(1, :))); 
turnover(1, 5) = turnover(1, 5) + sum(abs(marketNeutralWeights_mxwo(1, :))); 


% 1:5 is based on country beta, 6:end is based on mxwo beta
avgTurnover = mean(turnover)

% Add leverage Cost 
% Convert to monthly rates
% Define rates (annualized)
borrowingRate = 0.01;    % Cost to borrow
lendingRate = 0.005;     % Income when holding cash (typically < borrowingRate)
financingSpread = 0.002; % Additional cost for short positions

% Convert to monthly
monthlyBorrowingCost = (1 + borrowingRate)^(1/12) - 1;
monthlyLendingRate = (1 + lendingRate)^(1/12) - 1;
monthlyFinancingSpread = (1 + financingSpread)^(1/12) - 1;

for m = 1:nMonths
    currentLeverage = leverage_History(m);
    
    % Strategy 1: Equal Weight (typically no leverage costs)
    leverageCosts(m,1) = 0; % EW usually has no leverage
    
    % Strategies 2-3: Low/High Beta
    for s = 2:3
        if currentLeverage > 1
            % Pay borrowing cost on leveraged portion
            leverageCosts(m,s) = (currentLeverage - 1) * monthlyBorrowingCost;
        elseif currentLeverage < 1
            % Earn lending income on cash portion
            leverageCosts(m,s) = (1 - currentLeverage) * monthlyLendingRate;
        else
            % Exactly 1x leverage - no cost
            leverageCosts(m,s) = 0;
        end
    end
    
    % Strategy 4: Long-Short
    if currentLeverage > 1
        leverageCosts(m,4) = (currentLeverage - 1) * monthlyBorrowingCost + monthlyFinancingSpread;
    elseif currentLeverage < 1
        % Even if <1x, still pay financing spread for short positions
        leverageCosts(m,4) = (1 - currentLeverage) * monthlyLendingRate + monthlyFinancingSpread;
    else
        leverageCosts(m,4) = monthlyFinancingSpread; % Still pay spread at 1x
    end
    
    % Strategy 5: Market Neutral
    if currentLeverage > 1
        leverageCosts(m,5) = (currentLeverage - 1) * monthlyBorrowingCost + monthlyFinancingSpread;
    elseif currentLeverage < 1
        % Even if <1x, still pay financing spread
        leverageCosts(m,5) = (1 - currentLeverage) * monthlyLendingRate + monthlyFinancingSpread;
    else
        leverageCosts(m,5) = monthlyFinancingSpread; % Still pay spread at 1x
    end
end

% Calculate total costs (transaction + leverage)
totalCosts = tCost * turnover(:,1:end) + leverageCosts;


%
stratXsReturnsTC_msci_all = stratXsReturnsNoTC_msci_all - totalCosts;
% Performance statistics (benchmark is the mxwo)
strategyNames_mxwo_noTC = {'ew_mxwo', 'lowbeta_mxwo','highbeta_mxwo', 'longshort_mxwo', 'bab_mxwo'};
strategyNames_mxwo_withTC = {'ew_mxwoTC', 'lowbeta_mxwoTC','highbeta_mxwoTC', 'longshort_mxwoTC', 'bab_mxwoTC'};
summarizePerformance(stratXsReturnsTC_msci_all, monthlyRf, stratXsReturnsTC_msci_all(:, 1), annualizationFactor, strategyNames_mxwo_withTC, 'perf_mxwobeta_withTC_all');

cum_returns_msci_withTC = cumprod(1 + stratXsReturnsTC_msci_all);

% Open figure and hold
%% Calculate turnover and costs (keep your existing calculations)
% [Previous code for turnover, leverage costs, and return calculations remains unchanged]

%% Plotting cumulative Performance over whole Period (with and without TC and cost of leverage)
figure;
set(gcf, 'Position', [100, 100, 1200, 600]);
hold on;

% Define colors and line styles
colors = lines(5); % Get 5 distinct colors
noTC_LineStyle = '-';  % Solid line for no-TC
TC_LineStyle = ':';    % Dotted line for TC
lineWidth = 1.5;

% Plot both no-TC and with-TC series
for i = 1:size(cum_returns_msci_all, 2)
    % No-TC version (solid line)
    plot(monthlyDates, cum_returns_msci_all(:, i), ...
         'LineStyle', noTC_LineStyle, ...
         'Color', colors(i,:), ...
         'LineWidth', lineWidth);
    
    % With-TC version (dotted line)
    plot(monthlyDates, cum_returns_msci_withTC(:, i), ...
         'LineStyle', TC_LineStyle, ...
         'Color', colors(i,:), ...
         'LineWidth', lineWidth);
end

% Formatting
xlabel('Date', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cumulative Return', 'FontSize', 14, 'FontWeight', 'bold');
title('MSCI Strategy Performance: with and without TC (including leverage)', 'FontSize', 16);
set(gca, 'FontSize', 12, 'GridAlpha', 0.3);
grid on; box on;
datetick('x', 'yyyy', 'keeplimits');

% Create combined legend
strategyNames = {'EW', 'Low Beta', 'High Beta', 'Long-Short', 'Market Neutral'};
legendEntries = [strcat(strategyNames, ' (no TC)'); strcat(strategyNames, ' (with TC)')];
legend(legendEntries, 'Location', 'NorthWest', 'FontSize', 10, 'Box', 'off');

hold off;


%% %% Factor Regression
% Extract AQR and FF5 factors
aqrFactors = monthlyFactorsArray(:, [1, 2, 3, 4]); % BAB, QMJ, UMD, Mkt_RF
ff5Factors = monthlyFactorsArray(:, 4:8); % Mkt_RF, SMB, HML, RMW, CMA

strategyNames_msci4factors = {'ew_mxwo', 'lowbeta_mxwo', 'highbeta_mxwo', 'longshort_mxwo', 'bab_mxwo'};

% Initialize results cell array
resultsCell = {};

% Loop through each strategy
for i = 1:size(stratXsReturnsNoTC_msci_all, 2)
    strategyReturns = stratXsReturnsNoTC_msci_all(:, i);
    strategyName = strategyNames_msci4factors{i};
    
    % AQR Regression (included Mkt_RF)
    modelAQR = fitlm(aqrFactors, strategyReturns, 'VarNames', {'BAB','QMJ','UMD','Mkt_RF','Returns'});
    coeffAQR = modelAQR.Coefficients;
    rsqAQR = modelAQR.Rsquared.Adjusted;
    
    % FF5 Regression
    modelFF5 = fitlm(ff5Factors, strategyReturns, 'VarNames', {'Mkt_RF','SMB','HML','RMW','CMA','Returns'});
    coeffFF5 = modelFF5.Coefficients;
    rsqFF5 = modelFF5.Rsquared.Adjusted;
    
    % Format coefficients with significance stars and t-stats
    formatCoeff = @(c) sprintf('%.3f%s (%.2f)', c.Estimate, ...
        stars(c.pValue), c.tStat);
    
    % AQR Coefficients
    aqrIntercept = formatCoeff(coeffAQR('(Intercept)',:));
    aqrBAB = formatCoeff(coeffAQR('BAB',:));
    aqrQMJ = formatCoeff(coeffAQR('QMJ',:));
    aqrUMD = formatCoeff(coeffAQR('UMD',:));
    aqrMkt = formatCoeff(coeffAQR('Mkt_RF',:));
    
    % FF5 Coefficients
    ff5Intercept = formatCoeff(coeffFF5('(Intercept)',:));
    ff5Mkt = formatCoeff(coeffFF5('Mkt_RF',:));
    ff5SMB = formatCoeff(coeffFF5('SMB',:));
    ff5HML = formatCoeff(coeffFF5('HML',:));
    ff5RMW = formatCoeff(coeffFF5('RMW',:));
    ff5CMA = formatCoeff(coeffFF5('CMA',:));
    
    % Append to results
    resultsCell(end+1,:) = {strategyName, aqrIntercept, aqrBAB, aqrQMJ, aqrUMD, aqrMkt, ...
        sprintf('%.3f', rsqAQR), ff5Intercept, ff5Mkt, ff5SMB, ff5HML, ff5RMW, ff5CMA, ...
        sprintf('%.3f', rsqFF5)};
end

% Create and display table
varNames = {'Strategy','AQR_Alpha','AQR_BAB','AQR_QMJ','AQR_UMD','AQR_Mkt','AQR_R2',...
    'FF5_Alpha','FF5_Mkt','FF5_SMB','FF5_HML','FF5_RMW','FF5_CMA','FF5_R2'};
resultsTable = cell2table(resultsCell, 'VariableNames', varNames);
disp(resultsTable);

% Write to Excel file
writetable(resultsTable, 'Vol_adjusted_factor_regression_results.xlsx', 'Sheet', 'Results');



%% Other Plots to use for presentation and paper

% Plot VIX over time (maybe useful for our presentation, to explain why we
% took tresholds)


% Plot VIX over time with threshold annotations
figure;
set(gcf, 'Position', [100, 100, 1200, 600]);

% Plot VIX time series
plot(dates, vix, 'b-', 'LineWidth', 1.5);
hold on;

colors = {'g', 'm', 'r'}; % Colors for each threshold

% Plot threshold lines
for i = 1:length(vixThresholds)
    yline(vixThresholds(i), '--', ...
          sprintf('Threshold %d: %.1f', i, vixThresholds(i)), ...
          'Color', colors{i}, ...
          'LineWidth', 1.3, ...
          'LabelHorizontalAlignment', 'left', ...
          'FontSize', 10);
end

% Highlight periods of high volatility
highVolPeriods = vix > 30; % Define what you consider "high volatility"
area(dates, highVolPeriods.*max(vix)*1.1, ...
     'FaceColor', [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

% Formatting
xlabel('Year', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('VIX Level', 'FontSize', 14, 'FontWeight', 'bold');
title('CBOE Volatility Index (VIX) 1996-2025', 'FontSize', 16);
set(gca, 'FontSize', 12, 'GridAlpha', 0.3);
grid on; 
box on;

% Set y-axis to start at 0
ylim([0 max(vix)*1.1]);

% Improve date display
if isdatetime(dates)
    datetick('x', 'yyyy', 'keeplimits');
else
    % Convert if monthlyDates is numeric
    datetick('x', 'yyyy', 'keeplimits');
end

% Add annotation explaining thresholds
annotation('textbox', [0.15, 0.7, 0.2, 0.1], ...
           'String', 'Dashed lines show volatility regimes used for leverage adjustment', ...
           'FitBoxToText', 'on', ...
           'BackgroundColor', [1 1 1], ...
           'FontSize', 10);

hold off;


% heatmap, testing different leverage and thresholds

%% Robustness Check for VIX Thresholds and Leverage Adjustments
% Define parameter ranges to test
vixThresholds_test = {[12 22 35], [15 25 40], [18 28 45]}; % Different threshold sets
adjFactors_test = 0.7:0.1:1.5; % Adjustment factor range
leverageMultipliers_test = {[1.8 1.2 0.6 0.0], [1.5 1.0 0.5 0.0], [1.2 0.8 0.4 0.0]}; % Different leverage profiles

% Initialize results storage
results = struct();
strategies = {'EqualWeight', 'LowBeta', 'HighBeta', 'LongShort', 'MarketNeutral'};
numStrategies = length(strategies);

% Preallocate result matrices
for s = 1:numStrategies
    results.(strategies{s}).sharpe = zeros(length(vixThresholds_test), length(adjFactors_test), length(leverageMultipliers_test));
    results.(strategies{s}).returns = cell(length(vixThresholds_test), length(adjFactors_test), length(leverageMultipliers_test));
end

% Main Robustness Check Loop
for t = 1:length(vixThresholds_test)
    for a = 1:length(adjFactors_test)
        for l = 1:length(leverageMultipliers_test)
            % Current parameter set
            currentThresholds = vixThresholds_test{t};
            currentAdj = adjFactors_test(a);
            currentLeverage = leverageMultipliers_test{l};
            
            % Initialize strategy returns
            strategyReturns = zeros(nMonths-firstMonth+1, numStrategies);
            
            % Portfolio construction loop
            for m = firstMonth:(nMonths-1)  % Critical change here
    % Get current data
    nonMissings = isfinite(country_prices(lastDayList(m),:));
    currentBeta = monthlyBetas_mxwo(m,:);
    
    % VIX-based adjustment
    currentVIX = vix(lastDayList(m));
    
    if currentVIX < currentThresholds(1)
        leverage = currentLeverage(1);
        adjFactor = currentAdj * 0.9;
    elseif currentVIX <= currentThresholds(2)
        leverage = currentLeverage(2);
        adjFactor = currentAdj;
    elseif currentVIX <= currentThresholds(3)
        leverage = currentLeverage(3);
        adjFactor = currentAdj * 1.1;
    else
        leverage = currentLeverage(4);
        adjFactor = currentAdj * 1.3;
    end
    
    % Portfolio construction
     w_ew = nonMissings / sum(nonMissings) * leverage;
                w_low = computeSortWeights(currentBeta, nLongs, 0, 0) * leverage;
                w_high = computeSortWeights(currentBeta, nShorts, 0, 1) * leverage;
                w_ls = computeSortWeights(currentBeta, nLongs, nShorts, 0) * adjFactor;
                
                % Market neutral weights
                longBeta = max(0.1, sum(w_low .* currentBeta, "omitnan"));
                shortBeta = max(0.1, sum(w_high .* currentBeta, "omitnan"));
                w_mn = (w_low * adjFactor)/longBeta - (w_high / adjFactor)/shortBeta;
    
    % Safe to access next month's returns now
    nextReturns = monthlyXsReturns_country(m+1,:);
                strategyReturns(m-firstMonth+1,:) = [...
                    sum(nextReturns .* w_ew, "omitnan"), ...
                    sum(nextReturns .* w_low, "omitnan"), ...
                    sum(nextReturns .* w_high, "omitnan"), ...
                    sum(nextReturns .* w_ls, "omitnan"), ...
                    sum(nextReturns .* w_mn, "omitnan")];
            end
            
            % Calculate performance metrics
            for s = 1:numStrategies
                validReturns = strategyReturns(isfinite(strategyReturns(:,s)), s);
                if length(validReturns) >= 12
                    results.(strategies{s}).sharpe(t,a,l) = mean(validReturns)/std(validReturns)*sqrt(12);
                    results.(strategies{s}).returns{t,a,l} = validReturns;
                end
            end
        end
    end
end


% Create summary table
summaryTable = table();
for s = 1:numStrategies
    [maxSharpe, idx] = max(results.(strategies{s}).sharpe(:));
    [t_opt, a_opt, l_opt] = ind2sub(size(results.(strategies{s}).sharpe), idx);
    
    summaryTable.Strategy(s) = strategies(s);
    summaryTable.MaxSharpe(s) = maxSharpe;
    summaryTable.VIXThresholds(s) = {vixThresholds_test{t_opt}};
    summaryTable.AdjFactor(s) = adjFactors_test(a_opt);
    summaryTable.Leverage(s) = {leverageMultipliers_test{l_opt}};
end

% Display table
disp(summaryTable);
