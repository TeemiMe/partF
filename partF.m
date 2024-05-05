inputData = zeros(1000, 3); %1000 rows and 3 columns (inputs)

inputData(:,1) = rand(1000, 1); %Input 1 between 0-1
inputData(:,2) = rand(1000, 1); %Input 2 between 0-1
inputData(:,3) = rand(1000, 1); %Input 3 between 0-1


targets = evalfis(fis, inputData); %Generate the targets (Number of spares) by feeding input data to the fuzzy inference system

dataset = [inputData, targets];

trainingRatio = 0.8; %set the ratio of training to 80%

cv = cvpartition(1000, 'HoldOut', 1 - trainingRatio); %make a cross-validation partition object

trainIDX = cv.training;
testIDX = ~trainIDX;

trainData = dataset(trainIDX, :);
testData = dataset(testIDX, :);

%extracting input and target variables from the processed training data
trainInput = trainData(:, 1:end-1);
trainTarget = trainData(:, end);
testInput = testData(:, 1:end-1); 
testTarget = testData(:, end); 

%creating and training neuro fuzzy inference system
fis_rev=genfis1([trainInput trainTarget],[3],'trimf','linear');

Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'200', '0', '0.01', '0.9', '1.1'};
dims = [1 60];

PARAMS=inputdlg(Prompt,Title,dims,DefaultValues);
pause(0.01);

MaxEpoch=str2num(PARAMS{1});                
ErrorGoal=str2num(PARAMS{2});              
InitialStepSize=str2num(PARAMS{3}); 

StepSizeDecreaseRate=str2num(PARAMS{4});   
StepSizeIncreaseRate=str2num(PARAMS{5});    
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1; %Hybrid
           
fis_rev=anfis([trainInput trainTarget],fis_rev,TrainOptions,DisplayOptions,[],OptimizationMethod);

%Generate the targets by feeding test data to your neuro fuzzy inference system
TrainOutputs = evalfis(trainInput,fis_rev);
TestOutputs = evalfis(testInput,fis_rev);

%calculating errors, MSE, RMSE, mean and standard deviation for testing dataset.
TrainErrors=trainTarget-TrainOutputs;
TrainMSE=mean(TrainErrors.^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

%Plot between test target and test output.
figure;
plot(testTarget, TestOutputs, 'bo');
hold on;
xlabel('Test Target');
ylabel('Test Output');
title('Test Target vs Test Output');
grid on;

%inputs beyond the universe of discourse
inputBeyondUOD = [3, 6, 11];

%outputs FIS
fisOutputs = evalfis(fis, inputBeyondUOD);

%outputs neuro-fuzzy inference system
nfisOutputs = evalfis(fis_rev, inputBeyondUOD);

%comparison between FIS and neuro-fuzzy outputs
figure;
subplot(2, 1, 1);
plot(fisOutputs, 'ro', 'DisplayName', 'FIS Outputs');
hold on;
plot(nfisOutputs, 'bx', 'DisplayName', 'Neuro-Fuzzy Outputs');
xlabel('Data Point');
ylabel('Output');
title('Comparison of FIS and Neuro-Fuzzy Outputs');
legend;
grid on;

