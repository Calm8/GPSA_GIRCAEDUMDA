 clc;clear all;close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GECAD SSCI 2025 Competition: Evolutionary Computation in the Energy Domain: 2025 Edition of the Risk-based Energy Scheduling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('functions') %Necessary functions to run the algorithms (encrypted)

tTotalTime=tic; % lets track total computational time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Select_algorithm=2;
%1: HyDE algorithm (test algorithm)
%3：GPSA_GIRCAEDUMDA


%%%%%%%%%%%%%%%%%% FEs %%%%%%%%%%%%%%%%%%%%%%%%%
No_evals=5000; %Number of function evaluations % This are the number of function evaluations per trial. The lower the value, more points for the algorithm!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if No_evals > 5000
  error('5,000 evaluations is the maximum number allowed!');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load MH parameters (e.g., get MH parameters from DEparameters.m file)
switch Select_algorithm
       case 1
          addpath('HyDE')
          algorithm='HyDE-Algorithm'; %'The participants should include their algorithm here'
          DEparameters %Function defined by the participant
          No_solutions=deParameters.I_NP; %Notice that some algorithms are limited to one individual
          case 2
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          % GPSA_GIRCAEDUMDA
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          addpath('GPSA_GIRCAEDUMDA')
          algorithm='GPSA_GIRCAEDUMDA'; %'The participants should include their algorithm here'
          RCEDUMDAparameters %Function defined by the participant
          No_solutions=vRCEDUMDAparameters.I_NP; %Notice that some algorithms are limited to one individual      
          No_solutions=1; 
    otherwise
          fprintf(1,'No algorithm selected\n');
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data base 
[caseStudyData, DB_name]=callDatabase();
noRuns=20; %this can be changed but final results should be based on 20 trials

%% Label of the algorithm and the case study
Tag.algorithm=algorithm;
Tag.DB=DB_name;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set other parameters
otherParameters =setOtherParameters(caseStudyData, No_solutions);
otherParameters.one = setOtherParameters(caseStudyData,1);
%% Set lower/upper bounds of variables 
[lowerB,upperB] = setVariablesBounds(caseStudyData,otherParameters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Call the MH for optimizationclear 
ResDB=struc([]);
for iRuns=1:noRuns %Number of trails
    tOpt=tic;
    rand('state',sum(noRuns*100*clock))% ensure stochastic indpt trials
    otherParameters.iRuns=iRuns;      

    switch Select_algorithm
        case 1
            [ResDB(iRuns).Fit_and_p, ...
            ResDB(iRuns).sol, ...
            ResDB(iRuns).fitVector]= ...                   
            HyDE(deParameters,caseStudyData,otherParameters,lowerB,upperB,No_evals);
        case 2
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % GIRCEDUMDA2
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  [ResDB(iRuns).Fit_and_p, ...
                      ResDB(iRuns).sol, ...
                      ResDB(iRuns).fitVector]= ...
                      GPSA_GIRCAEDUMDA(vRCEDUMDAparameters,caseStudyData,otherParameters,lowerB,upperB,No_evals);  
                
    end 
        ResDB(iRuns).noEvals=No_evals;
        ResDB(iRuns).tOpt=toc(tOpt); % time of each trial
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Save the results and stats
        Save_results
end

%% End of MH Optimization
for j=1:noRuns
    lower_violations=ResDB(j).sol<lowerB;
    upper_violations=ResDB(j).sol>upperB;
    many_low(j)=sum(lower_violations);
    many_up(j)=sum(upper_violations);
    fprintf('lower_violations : %d .\n', many_low(j));
    fprintf('upper_violations : %d .\n', many_up(j));
end

tTotalTime=toc(tTotalTime); %Total time

