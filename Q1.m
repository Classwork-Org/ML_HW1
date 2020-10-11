clear all, close all,

%% ========================= Conditional PDF Paramemter Setup ========================= %%

n = 4; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-1;-1;-1;-1]; 
mu(:,2) = [1;1;1;1];
Sigma(:,:,1) = [2 -0.5 0.3 0; -0.5 1 -0.5 0; 0.3 -0.5 1 0; 0 0 0 2];
Sigma(:,:,2) = [1 0.3 -0.2 0; 0.3 2 0.3 0; -0.2 0.3 1 0; 0 0 0 3];
p = [0.7,0.3]; % class priors for labels 0 and 1 respectively

labels = rand(1,N) >= p(1);
Nc = [length(find(labels==0)),length(find(labels==1))]; % number of samples from each class
x = zeros(n,N); % save up space

%% ========================= Dataset Generation ========================= %%

for i = 1:N
    if(labels(i)==0)
        x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1));
    else
        x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2));
    end
end

%% ========================= ERM descriminator score evaluation ========================= %%

p_x_given_l_equals_0 = evalGaussian(x,mu(:,1),Sigma(:,:,1));
p_x_given_l_equals_1 = evalGaussian(x,mu(:,2),Sigma(:,:,2));

descriminant_score_ERM = log(p_x_given_l_equals_1./p_x_given_l_equals_0);

%% ========================= ROC plot ========================= %%

[PfpERM,PfnERM,PtpERM,PtnERM,PerrorERM,thresholdListERM] = ROCcurve(descriminant_score_ERM,labels);
figure(1), clf,
subplot(5,2,3), hold on, plot(PfpERM,PtpERM,'m'),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for ERM Discriminant Scores'),
subplot(5,2,4), hold on, plot(thresholdListERM,PerrorERM,'m'),
xlabel('Thresholds'), ylabel('P(error) for ERM Discriminant Scores'), title('P(error; Threshold)')

%% ========================= Theoretical and Empirical optimal descriminator threshold ========================= %%
lambda = [0 1;1 0]; % loss values
theoretical_gamma = log(((lambda(2,1)-lambda(1,1))*p(1))/((lambda(1,2)-lambda(2,2))*p(2))); %threshold
min_perrorERM_index = find(PerrorERM == min(PerrorERM));
optimal_emperical_gamma = thresholdListERM(min_perrorERM_index);
PtpERM_at_optimal_emperical_gamma = PtpERM(min_perrorERM_index);
PfpERM_at_optimal_emperical_gamma = PfpERM(min_perrorERM_index);

tau = theoretical_gamma;
decisions = (descriminant_score_ERM >= tau);
PerrorERM_theoretical = sum(decisions~=labels)/length(labels);

%% ========================= Threshold plots on ROC curve and P(error;Tau) curve ========================= %%
subplot(5,2,4), hold on
plot(optimal_emperical_gamma(1),min(PerrorERM),'g*'), hold on,
yline(min(PerrorERM))
legend('P(error,Threshold)', ['Tau = ' num2str(optimal_emperical_gamma(1),'%02f')], ['min P(error) = ' num2str(min(PerrorERM),'%02d')]), 

subplot(5,2,3), hold on
plot(PfpERM_at_optimal_emperical_gamma,PtpERM_at_optimal_emperical_gamma,'g*'), hold on,
legend('ROC curve', ['Tau = ' num2str(optimal_emperical_gamma(1),'%02f')]), 

%% ========================= 2D Scatter Plots for X components ========================= %%
label0_indexes = find(labels==0);
label1_indexes = find(labels==1);

subplot(5,2,1),
plot(x(1,label0_indexes),x(2,label0_indexes),'o'), hold on,
plot(x(1,label1_indexes),x(2,label1_indexes),'+'),
legend('Class 0','Class 1'), 
title('Data and their true labels For First And Second Component of x'),
xlabel('x_1'), ylabel('x_2'), 

subplot(5,2,2),
plot(x(3,label0_indexes),x(4,label0_indexes),'o'), hold on,
plot(x(3,label1_indexes),x(4,label1_indexes),'+'),
legend('Class 0','Class 1'), 
title('Data and their true labels For Third And Fourth Component of '),
xlabel('x_3'), ylabel('x_4');

%% ========================= Contour Plot of Decision Boundry ========================= %%
% to draw 2D contors, let's pretend the problem has been reduced to 
% 2 problems with 2 class ERM and x in 2D, mu and sigma are then submatricies of the
% original matricies. The optimal theoretical gamma is independent of
% number of components of x. Therefore if we just look at the covariance
% matrix of only 2 components at a time plus their means, the optimal
% decision boundry will be defined at the same level (0) of the 3D surface
% score = log(g1) - log(g2) - log(gamma). Contour plots this surface can
% then be plotted with a mesh grid from the min and max of the pair of
% components being looked at. This process can be done for all component
% pairs and it should given that that they are not independent from each
% other, however, it has only been done for 2 possible component pairs for
% illustration. 

aGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),100);
bGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),100);
cGrid = linspace(floor(min(x(3,:))),ceil(max(x(3,:))),100);
dGrid = linspace(floor(min(x(4,:))),ceil(max(x(4,:))),100);

[a, b] = meshgrid(aGrid,bGrid);
[c, d] = meshgrid(bGrid,cGrid);

discriminantScoreGridValues = log(evalGaussian([a(:)';b(:)'],mu(1:2,2),Sigma(1:2,1:2,2)))-log(evalGaussian([a(:)';b(:)'],mu(1:2,1),Sigma(1:2,1:2,1))) - log(theoretical_gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,100,100);

subplot(5,2,1),
contour(aGrid,bGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
legend('Class 0','Class 1', 'Contours of discriminant function');

discriminantScoreGridValues = log(evalGaussian([c(:)';d(:)'],mu(3:4,2),Sigma(3:4,3:4,2)))-log(evalGaussian([c(:)';d(:)'],mu(3:4,1),Sigma(3:4,3:4,1))) - log(theoretical_gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,100,100);

subplot(5,2,2),
contour(bGrid,cGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
legend('Class 0','Class 1', 'Contours of discriminant function');


%% ========================= Applying Naive Bayesian Assumptions ========================= %%
Sigmahat(:,:,1) = Sigma(:,:,1) .* eye(4);
Sigmahat(:,:,2) = Sigma(:,:,2) .* eye(4);

naive_p_x_given_l_equals_0 = evalGaussian(x,mu(:,1),Sigmahat(:,:,1));
naive_p_x_given_l_equals_1 = evalGaussian(x,mu(:,2),Sigmahat(:,:,2));

naive_descriminant_score_ERM = log(naive_p_x_given_l_equals_1./naive_p_x_given_l_equals_0);

%% ========================= ROC plot for Naive Bayesian ========================= %%
[naive_PfpERM,naive_PfnERM,naive_PtpERM,naive_PtnERM,naive_PerrorERM,naive_thresholdListERM] = ROCcurve(naive_descriminant_score_ERM,labels);
subplot(5,2,5), hold on, 
plot(naive_PfpERM,naive_PtpERM,'m'),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for Naive ERM Discriminant Scores');

subplot(5,2,6), hold on, plot(naive_thresholdListERM, naive_PerrorERM,'m'),
xlabel('Thresholds'), ylabel('P(error) for Naive ERM Discriminant'), title('P(error; Threshold) Naive ERM Discriminant');

%% ========================= Threshold plots on ROC curve and P(error;Tau) (Naive Bayesian) ========================= %%
naive_min_perrorERM_index = find(naive_PerrorERM == min(naive_PerrorERM));
naive_optimal_emperical_gamma = naive_thresholdListERM(naive_min_perrorERM_index);
naive_PtpERM_at_optimal_emperical_gamma = naive_PtpERM(naive_min_perrorERM_index);
naive_PfpERM_at_optimal_emperical_gamma = naive_PfpERM(naive_min_perrorERM_index);

subplot(5,2,6), hold on
plot(naive_optimal_emperical_gamma(1),min(naive_PerrorERM),'g*'), hold on,
yline(min(naive_PerrorERM))
legend('P(error,Threshold)', ['Tau = ' num2str(naive_optimal_emperical_gamma(1),'%02f')], ['min P(error) = ' num2str(min(naive_PerrorERM),'%02f')]), 

subplot(5,2,5), hold on
plot(naive_PfpERM_at_optimal_emperical_gamma(1),naive_PtpERM_at_optimal_emperical_gamma(1),'g*'), hold on,
legend('ROC curve', ['Tau = ' num2str(naive_optimal_emperical_gamma(1),'%02f')]), 

%% ========================= LDA sample mean and covariance estimate ========================= %%
muhat(:,1) = mean(x(:,label0_indexes),2);
muhat(:,2) = mean(x(:,label1_indexes),2);
Sigmahat(:,:,1) = cov(x(:,label0_indexes)');
Sigmahat(:,:,2) = cov(x(:,label1_indexes)');

%% ========================= LDA between class and within class scatter matrix calculation ========================= %%
Sb = (muhat(:,1)-muhat(:,2))*(muhat(:,1)-muhat(:,2))'; Sw = Sigmahat(:,:,1) + Sigmahat(:,:,2);

%% ========================= LDA weights calculation ========================= %%
[V,D] = eig(Sw\Sb); [~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector at greatest eigen value
y1 = w'*x(:,label0_indexes); y2 = w'*x(:,label1_indexes); 
if mean(y2)<=mean(y1), w = -w; end % push label0 projections to the left

%% ========================= LDA Projection Plot ========================= %%
subplot(5,2,[7 8]), plot(y1(1,:),zeros(1,size(y1,2)),'r*'); hold on;
plot(y2(1,:),zeros(1,size(y2,2)),'bo'); axis equal,

%% ========================= LDA ROC Plot ========================= %%
y=w'*x;
[LDA_Pfp,LDA_Pfn,LDA_Ptp,LDA_Ptn,LDA_Perror,LDA_thresholdList] = ROCcurve(y,labels);
subplot(5,2,9), hold on, 
plot(LDA_Pfp, LDA_Ptp,'m'),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for LDA');

subplot(5,2,10), hold on, plot(LDA_thresholdList, LDA_Perror,'m'),
xlabel('Thresholds'), ylabel('P(error) for LDA'), title('P(error; Threshold) LDA');

%% ========================= Threshold plots on LDA ROC curve and LDA P(error;Tau) curve ========================= %%
subplot(5,2,10), hold on
LDA_min_perror_index = find(LDA_Perror == min(LDA_Perror));
LDA_optimal_emperical_gamma = LDA_thresholdList(LDA_min_perror_index);
plot(LDA_optimal_emperical_gamma(1),min(LDA_Perror),'g*'), hold on,
yline(min(LDA_Perror))
legend('P(error,Threshold)', ['Tau = ' num2str(LDA_optimal_emperical_gamma(1),'%02f')], ['min P(error) = ' num2str(min(LDA_Perror),'%02d')]), 

LDA_Ptp_at_optimal_emperical_gamma = LDA_Ptp(LDA_min_perror_index);
LDA_Pfp_at_optimal_emperical_gamma = LDA_Pfp(LDA_min_perror_index);
subplot(5,2,9), hold on
plot(LDA_Pfp_at_optimal_emperical_gamma(1),LDA_Ptp_at_optimal_emperical_gamma(1),'g*'), hold on,
legend('ROC curve', ['Tau = ' num2str(LDA_optimal_emperical_gamma(1),'%02f')]), 

%% ========================= Helper Functions ========================= %%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [Pfp,Pfn,Ptp,Ptn,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
Ptp = zeros(1,length(thresholdList));
Ptn = zeros(1,length(thresholdList));
Pfp = zeros(1,length(thresholdList));
Pfn = zeros(1,length(thresholdList));
Perror = zeros(1,length(thresholdList));
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1));
    Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0));
    Ptn(i) = length(find(decisions==0 & labels==0))/length(find(labels==0));
    Pfn(i) = length(find(decisions==0 & labels==1))/length(find(labels==1));
    Perror(i) = sum(decisions~=labels)/length(labels);
end
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
