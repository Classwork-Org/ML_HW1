clear all, close all,

%% ========================= Conditional PDF Paramemter Setup ========================= %%
n = 3; % number of feature dimensions
N = 10000; % number of iid samples
L = 4; % number of labels
label_text = {'0', '1', '2', '3'}';
mu(:,1) = [0; 10;10]; 
mu(:,2) = [0; 10;-10];
mu(:,3) = [0; -10;-10]; 
mu(:,4) = [0; -10;10];

assymmetry = 5;
seperation = 0.06;
proximity = 1/seperation;
lambda = repmat(proximity,1,L)+(0:L-1).*assymmetry;

Sigma(:,:,1) = lambda(1).*eye(n);
Sigma(:,:,2) = lambda(2).*eye(n);
Sigma(:,:,3) = lambda(3).*eye(n);
Sigma(:,:,4) = lambda(4).*eye(n);

priors = [0.2, 0.25, 0.25, 0.3]; % class priors for labels 0 -> 3
p = cumsum(priors); 

%% ========================= Data Generation ========================= %%
[x, labels] = generateData4Gaussians(n, N, p, mu, Sigma);

%% ========================= Data Scatter Plot ========================= %%
figure(1), clf,
subplot(6,5,[1 2 6 7]);
mShapes = 'ox+*.';
mColors = 'rgbmy';
for l = 0:L-1
    scatter3(x(1,labels == l), x(2,labels == l), x(3,labels == l), strcat(mShapes(l+1),mColors(l+1))), hold on;
end
title('Scatter Plot For Data in Terms of Its Components');
xlabel('x_1'), ylabel('x_2'), zlabel('x_3'),
legend('Class 0', 'Class 1', 'Class 2', 'Class 3');

%% ========================= Data 2D Projection with PCA Scatter Plot ========================= %%
for l = 0:L-1
    x_label = x(:,labels==l);
    pca_components = pca(x_label, Sigma(:,:,l+1), mu(:, l+1));
    subplot(6,5,5);
    plot(pca_components(1,:), pca_components(2,:), strcat(mShapes(l+1),mColors(l+1))), hold on;
    subplot(6,5,10);
    plot(pca_components(2,:), pca_components(3,:), strcat(mShapes(l+1),mColors(l+1))), hold on;
    subplot(6,5,[3 4 8 9]);
    scatter3(pca_components(1,:), pca_components(2,:),pca_components(3,:), strcat(mShapes(l+1),mColors(l+1))), hold on;
end
subplot(6,5,5);
title('1st and 2nd PCA components');
xlabel('x^{pca}_1'), ylabel('x^{pca}_2'),
lgnd = legend('Class 0', 'Class 1', 'Class 2', 'Class 3');
lgnd.Location = 'bestoutside';

subplot(6,5,10);
title('2nd and 3rd PCA components');
xlabel('x^{pca}_2'), ylabel('x^{pca}_3'),
lgnd = legend('Class 0', 'Class 1', 'Class 2', 'Class 3');
lgnd.Location = 'bestoutside';

subplot(6,5,[3 4 8 9]);
title('Scatter Plot For all PCA components');
xlabel('x^{pca}_1'), ylabel('x^{pca}_2'), zlabel('x^{pca}_3'),
lgnd = legend('Class 0', 'Class 1', 'Class 2', 'Class 3');
lgnd.Location = 'bestoutside';

% 
%% ========================= ERM Classification ========================= %%
lossMatrix = ones(L,L)-eye(L);
[avg_cost, p_error, decision_label] = ERMClassifyWithLlabels(x, labels, L, N, lossMatrix, priors, mu, Sigma);
display('Q2-A 0-1 loss matrix');
avg_cost
p_error

%% ========================= Confusion Matrix and P(error) ========================= %%
ConfusionMatrix = zeros(L,L);
for d = 0:L-1 % each decision option
    for l = 0:L-1 % each class label
        ind_dl = find(decision_label==d & labels==l);
        ConfusionMatrix(d+1,l+1) = sum(ind_dl)/sum(find(labels==l));
    end
end

%% ========================= Confusion Matrix Plot ========================= %%
% calculate the percentage accuracies
subplot(6,5,[14 15 19 20]),
plotConfusionMatrix(ConfusionMatrix, label_text);

%% ========================= Classification Scatter Plot With PCA ========================= %%
for l = 0:L-1 % each class label
    ind_l = find(labels==l);
    classification_result = (decision_label(ind_l) == l);
    correct_classifications = find(classification_result == 1);
    incorrect_classifications = find(classification_result == 0);
    x_label = x(:,labels==l);
    pca_components = pca(x_label, Sigma(:,:,l+1), mu(:, l+1));
    subplot(6,5,18),
    plot(pca_components(1,correct_classifications), pca_components(2,correct_classifications), strcat(mShapes(l+1),'g')), hold on
    plot(pca_components(1,incorrect_classifications), pca_components(2,incorrect_classifications), strcat(mShapes(l+1),'r')), hold on;
    subplot(6,5,13),
    plot(pca_components(2,correct_classifications), pca_components(3,correct_classifications), strcat(mShapes(l+1),'g')), hold on
    plot(pca_components(2,incorrect_classifications), pca_components(3,incorrect_classifications), strcat(mShapes(l+1),'r')), hold on;
    subplot(6,5,[11 12 16 17]),
    scatter3(x_label(1, correct_classifications), x_label(2, correct_classifications), x_label(3, correct_classifications), strcat(mShapes(l+1),'g')), hold on;
    scatter3(x_label(1, incorrect_classifications), x_label(2, incorrect_classifications), x_label(3, incorrect_classifications), strcat(mShapes(l+1),'r')), hold on;
%     subplot(6,5,,,10),
%     plot(pca_components(1,correct_classifications), pca_components(3,correct_classifications), strcat(mShapes(l+1),'g')), hold on
%     plot(pca_components(1,incorrect_classifications), pca_components(3,incorrect_classifications), strcat(mShapes(l+1),'r')), hold on;
end
subplot(6,5,18),
title('1st and 2nd PCA Classification');
xlabel('x^{pca}_1'), ylabel('x^{pca}_2'),
lgnd = legend('Class 0 Correcti', 'Class 0 Incorrect', ...
    'Class 1 Correcti', 'Class 1 Incorrect', ...
    'Class 2 Correcti', 'Class 2 Incorrect', ...
    'Class 3 Correcti', 'Class 3 Incorrect');
lgnd.Location = 'bestoutside';

subplot(6,5,13),
title('2nd and 3rd PCA Classification');
xlabel('x^{pca}_2'), ylabel('x^{pca}_3'),
lgnd = legend('Class 0 Correcti', 'Class 0 Incorrect', ...
    'Class 1 Correcti', 'Class 1 Incorrect', ...
    'Class 2 Correcti', 'Class 2 Incorrect', ...
    'Class 3 Correcti', 'Class 3 Incorrect');
lgnd.Location = 'bestoutside';


subplot(6,5,[11 12 16 17]),
title('Scatter Plot For all PCA components Classification');
xlabel('x^{pca}_1'), ylabel('x^{pca}_2'), zlabel('x^{pca}_3'),
lgnd = legend('Class 0 Correcti', 'Class 0 Incorrect', ...
    'Class 1 Correcti', 'Class 1 Incorrect', ...
    'Class 2 Correcti', 'Class 2 Incorrect', ...
    'Class 3 Correcti', 'Class 3 Incorrect');
lgnd.Location = 'bestoutside';



%% ========================= Loss Matrix Modification ========================= %%
lossMatrix = [0 1 2 3; 10 0 5 10; 20 10 0 1; 30 20 1 0];
[avg_cost, p_error, decision_label] = ERMClassifyWithLlabels(x, labels, L, N, lossMatrix, priors, mu, Sigma);
display('Q2-B loss matrix');
avg_cost
p_error

%% ========================= P(error) as a fn of seperation ========================= %%
sweep_assymmetry = 0;
sweep_seperation = 0.001:0.001:0.1;
N = 1000;
p_error_sweep = zeros(1,length(sweep_seperation));
p_error_sweep_pca = zeros(1,length(sweep_seperation));
lossMatrix = ones(L,L)-eye(L);

parfor i = 1:length(sweep_seperation)

    % set required seperations
    proximity = 1/sweep_seperation(i);
    lambda = repmat(proximity,1,L)+(0:L-1).*sweep_assymmetry;

    sigma1 = lambda(1).*eye(n);
    sigma2 = lambda(2).*eye(n);
    sigma3 = lambda(3).*eye(n);
    sigma4 = lambda(4).*eye(n);
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);
    
    % generate data at required seperations

    [x, labels] = generateData4Gaussians(n, N, p, mu, Sigmahat);
    
    % estimate mean and covariance using sample mean and sample covariance
    
    x_label0 = x(:,labels==0);
    x_label1 = x(:,labels==1);
    x_label2 = x(:,labels==2);
    x_label3 = x(:,labels==3);
    
    mlabel = sort(labels, 'ascend');
    
    sigma1 = cov(x_label0');
    sigma2 = cov(x_label1');
    sigma3 = cov(x_label2');
    sigma4 = cov(x_label3');
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);
    
    muhat1 = mean(x_label0, 2);
    muhat2 = mean(x_label1, 2);
    muhat3 = mean(x_label2, 2);
    muhat4 = mean(x_label3, 2);
    
    muhat = cat(2, muhat1, muhat2, muhat3, muhat4);
    
    % apply ERM with estimates

    [~, p_error_tmp, ~] = ERMClassifyWithLlabels(x, labels, L, N, lossMatrix, priors, muhat, Sigmahat);
    p_error_sweep(i) = p_error_tmp.*100;
    
    % apply pca to data and treat projections as new data
    
    pca_components_label0 = pca(x_label0);
    pca_components_label1 = pca(x_label1);
    pca_components_label2 = pca(x_label2);
    pca_components_label3 = pca(x_label3);
    
    pca_components = cat(2, pca_components_label0, pca_components_label1, pca_components_label2, pca_components_label3);
    
    sigma1 = cov(pca_components_label0');
    sigma2 = cov(pca_components_label1');
    sigma3 = cov(pca_components_label2');
    sigma4 = cov(pca_components_label3');
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);
    
    muhat1 = mean(pca_components_label0, 2);
    muhat2 = mean(pca_components_label1, 2);
    muhat3 = mean(pca_components_label2, 2);
    muhat4 = mean(pca_components_label3, 2);

    muhat = cat(2, muhat1, muhat2, muhat3, muhat4);
    
    [~, p_error_tmp, ~] = ERMClassifyWithLlabels(pca_components, mlabel, L, N, lossMatrix, priors, muhat, Sigmahat);
    p_error_sweep_pca(i) = p_error_tmp.*100;
end

subplot(6,5, [21 22 23 24 25]), plot(sweep_seperation, p_error_sweep, 'm'), hold on,
plot(sweep_seperation, p_error_sweep_pca, 'r'),
title('P(error) as a function of class conditional proximity'),
xlabel('Seperation'), ylabel('P(error)'), legend('ERM with original Data', 'ERM with pca wightening');


%% ========================= P(error) as a fn of sweep_assymmetry ========================= %%
sweep_assymmetry = 0:1:99;
sweep_seperation = 0.1;
N = 1000;
p_error_sweep = zeros(1,length(sweep_assymmetry));
p_error_sweep_pca = zeros(1,length(sweep_seperation));

parfor i = 1:length(sweep_assymmetry)

    proximity = 1/sweep_seperation;
    lambda = repmat(proximity,1,L)+(0:L-1).*sweep_assymmetry(i);

    sigma1 = lambda(1).*eye(n);
    sigma2 = lambda(2).*eye(n);
    sigma3 = lambda(3).*eye(n);
    sigma4 = lambda(4).*eye(n);
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);

    [x, labels] = generateData4Gaussians(n, N, p, mu, Sigmahat);
 % estimate mean and covariance using sample mean and sample covariance
    
    x_label0 = x(:,labels==0);
    x_label1 = x(:,labels==1);
    x_label2 = x(:,labels==2);
    x_label3 = x(:,labels==3);
    
    mlabel = sort(labels, 'ascend');
    
    sigma1 = cov(x_label0');
    sigma2 = cov(x_label1');
    sigma3 = cov(x_label2');
    sigma4 = cov(x_label3');
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);
    
    muhat1 = mean(x_label0, 2);
    muhat2 = mean(x_label1, 2);
    muhat3 = mean(x_label2, 2);
    muhat4 = mean(x_label3, 2);
    
    muhat = cat(2, muhat1, muhat2, muhat3, muhat4);
    
    % apply ERM with estimates
    
    [~, p_error_tmp, ~] = ERMClassifyWithLlabels(x, labels, L, N, lossMatrix, priors, mu, Sigmahat);
    p_error_sweep(i) = p_error_tmp.*100;
    
    pca_components_label0 = pca(x_label0);
    pca_components_label1 = pca(x_label1);
    pca_components_label2 = pca(x_label2);
    pca_components_label3 = pca(x_label3);
    
    pca_components = cat(2, pca_components_label0, pca_components_label1, pca_components_label2, pca_components_label3);

    sigma1 = cov(pca_components_label0');
    sigma2 = cov(pca_components_label1');
    sigma3 = cov(pca_components_label2');
    sigma4 = cov(pca_components_label3');
    
    Sigmahat = cat(3, sigma1 ,sigma2, sigma3, sigma4);
    
    muhat1 = mean(pca_components_label0, 2);
    muhat2 = mean(pca_components_label1, 2);
    muhat3 = mean(pca_components_label2, 2);
    muhat4 = mean(pca_components_label3, 2);

    muhat = cat(2, muhat1, muhat2, muhat3, muhat4);
    
    [~, p_error_tmp, ~] = ERMClassifyWithLlabels(pca_components, mlabel, L, N, lossMatrix, priors, muhat, Sigmahat);
    p_error_sweep_pca(i) = p_error_tmp.*100;
end

subplot(6,5, [26 27 28 29 30]), plot(sweep_assymmetry, p_error_sweep, 'm'), hold on,
plot(sweep_assymmetry, p_error_sweep_pca, 'r');
title('P(error) as a function of class conditional assymmetry'),
xlabel('Assymmetry'), ylabel('P(error)'), legend('ERM with original Data', 'ERM with pca wightening');

%% ========================= Helper Functions ========================= %%
function [avg_cost, p_error, decision_label] = ERMClassifyWithLlabels(x, labels, L, N, lossMatrix, priors, mu, Sigma)

p_x_given_l = zeros(L,N);
for l = 0:L-1
    p_x_given_l(l+1,:) = evalGaussian(x, mu(:,l+1), Sigma(:,:,l+1));
end
p_x = priors*p_x_given_l; % For minimization I probably don't need this because it's just a scaling factor
p_l_given_x = p_x_given_l.*repmat(priors',1,N)./repmat(p_x,L,1);
expectedRisks = lossMatrix*p_l_given_x;
[~, decision_label] = min(expectedRisks, [], 1);
decision_label = decision_label - 1;

costs = zeros(1,N);
for i = 1:N
    costs(i) = lossMatrix(decision_label(i)+1, labels(i)+1);
end

avg_cost = sum(costs)/length(costs);

p_error = length(find(decision_label ~= labels))/length(labels);

end

function [x, labels] = generateData4Gaussians(n, N, p, mu, Sigma)

labels = zeros(1,N);
x = zeros(n,N); % save up space
for i = 1:N
    random_number = rand(1,1);
    if random_number >= p(3)
        labels(i) = 3;
        x(:,i) = mvnrnd(mu(:,4),Sigma(:,:,4));
    elseif random_number >= p(2)
        labels(i) = 2;
        x(:,i) = mvnrnd(mu(:,3),Sigma(:,:,3));
    elseif random_number >= p(1)
        labels(i) = 1;
        x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2));
    else
        labels(i) = 0;
        x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1));
    end
end
end

function [] = plotConfusionMatrix(ConfusionMatrix, labels)

L = length(labels);

confpercent = ConfusionMatrix.*100;

% plotting the colors
imagesc(confpercent),
title('Confusion Matrix');
ylabel('Decision'); xlabel('Label');

% set the colormap
colormap(flipud(gray));

% Create strings from the matrix values and remove spaces
textStrings = strcat(num2str(confpercent(:), '%.1f\n'),'%');
textStrings = strtrim(cellstr(textStrings));

% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:L);
hStrings = text(x(:),y(:),textStrings(:), ...
    'HorizontalAlignment','center');

% Get the middle value of the color range
midValue = mean(get(gca,'CLim'));

% Choose white or black for the text color of the strings so
% they can be easily seen over the background color
textColors = repmat(confpercent(:) > midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));

% Setting the axis labels
set(gca,'XTick',1:L,...
    'XTickLabel',labels,...
    'YTick',1:L,...
    'YTickLabel',labels,...
    'TickLength',[0 0]);

end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [y] = pca(x, sigma, mu)

% If original distribution parameters not passed then 
% use sample-based estimates of mean and covariance matrix
if ~exist('sigma', 'var')
    Sigmahat = cov(x');
else
    Sigmahat = sigma;
end

if ~exist('mu', 'var')
    muhat = mean(x,2)';
else
    muhat = mu';
end

% make data 0 mean
xzm = x - muhat*ones(size(x));

% Get the eigenvectors (in Q) and eigenvalues (in D) of the
% estimated covariance matrix
[Q,D] = eig(Sigmahat);

% Sort the eigenvalues from large to small, reorder eigenvectors
% accordingly as well.
[d,ind] = sort(diag(D),'descend');
Q = Q(:,ind);
D = diag(d);

% Calculate the principal components (in y)
% Also whiten components so their norms are 1
y = D^(-1/2)*Q'*xzm; 
end

