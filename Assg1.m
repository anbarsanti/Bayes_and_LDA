% Prepare our data
X_test = load ('data_test.mat');
X_train = load('data_train.mat');
Y_train = load('label_train.mat');

X_test = X_test.data_test;
X_train = X_train.data_train;
Y_train = Y_train.label_train;

%% Naive Bayes Classifier
% Train the Naive Bayes Classifier
cnb_classifier = fitcnb(X_train, Y_train);

% Make Prediction 
[cnb_predlabels, cnb_score] = predict(cnb_classifier, X_test);
%cnb_predlabels

% Visualize
% gscatter(X_train(:,1), X_train(:,2), Y_train, 'rgb', 'x', [], 'off');
% hold on;
% gscatter(X_test(:,1), X_test(:,2), cnb_predlabels, 'rgb', '.', [], 'off');
% % cnb_classifier.ClassNames
% legend('Class 1 (Training Data)', 'Class 2 (Training Data)', 'Testing Data Predicted as Class 1', 'Testing Data Predicted as Class 2');
% hold off;

%% Linear Discriminant Analysis
% % Train the LDA Classifier
lda_classifier = fitcdiscr(X_train, Y_train, 'DiscrimType', 'linear', 'Prior', 'uniform');

% % Make Prediction
lda_predlabels = predict(lda_classifier, X_test);
lda_predlabels'

% Visualize
gscatter(X_train(:,1), X_train(:,2), Y_train, 'rgb', 'x', [], 'off');
hold on;
gscatter(X_test(:,1), X_test(:,2), lda_predlabels, 'rgb', '.', [], 'off');
% cnb_classifier.ClassNames
legend('Class 1 (Training Data)', 'Class 2 (Training Data)', 'Testing Data Predicted as Class 1', 'Testing Data Predicted as Class 2');
hold off;

%% Bayes Decision Rule
% Estimate Priors
classes = unique(Y_train);
numClasses = length(classes);
priors = zeros(1, numClasses);
for i = 1:numClasses
    priors(i) = sum(Y_train == classes(i))/length(Y_train);
end

% Estimate Likelihoods, X as feature
numFeatures = size(X_train, 2);
likelihoods = zeros(numFeatures, numClasses);
for i = 1:numClasses
    classData = X_train(Y_train == classes(i), :);
    likelihoods(:,i) = mean(classData); %store the mean for gaussian
end

% Estimate Likelihoods no. 2, X as data points

% Define Cost Matrix
cost = ones(numClasses) - eye(numClasses);

% Apply Bayes Decision rule
% Compute posterior probabilities
posteriors = likelihoods* diag(priors);

% % Compute expected costs and make decisions
expectedCost = cost * posteriors';
[~, classIndex] = min(expectedCost);

% Evaluate the classifier
bdr_predlabels = classes(classIndex);
