function [Fit_and_p,FVr_bestmemit, fitMaxVector] = ...
    GPSA_GIRCAEDUMDA(vRCEDUMDAparameters,caseStudyData,otherParameters,low_habitat_limit,up_habitat_limit,No_evals)

I_NP         = vRCEDUMDAparameters.I_NP;
I_D          = numel(up_habitat_limit);

I_lowPop     = vRCEDUMDAparameters.I_lowPop;

FVr_minbound = low_habitat_limit;
FVr_maxbound = up_habitat_limit;

I_itermax    = vRCEDUMDAparameters.I_itermax;

I_indByCell         = vRCEDUMDAparameters.I_indByCell;
I_StructureWidth    = vRCEDUMDAparameters.I_StructureWidth;
I_StructureHeight   = vRCEDUMDAparameters.I_StructureHeight;
I_NeighborhoodType  = vRCEDUMDAparameters.I_NeighborhoodType;
I_NeighborhoodParam = vRCEDUMDAparameters.I_NeighborhoodParam;

I_SelectionMethod   = vRCEDUMDAparameters.I_SelectionMethod;
I_Selected          = vRCEDUMDAparameters.I_Selected; 

I_elitismParam = vRCEDUMDAparameters.I_elitismParam;

I_CodeSize     = vRCEDUMDAparameters.I_CodeSize;

I_EstimationParamsMethod = vRCEDUMDAparameters.I_EstimationParamsMethod; 
I_EstimationParamsParam  = vRCEDUMDAparameters.I_EstimationParamsParam;

I_EstimationStructureMethod = vRCEDUMDAparameters.I_EstimationStructureMethod; 
I_EstimationStructureParam  = vRCEDUMDAparameters.I_EstimationStructureParam ;

I_ScalerMethod = vRCEDUMDAparameters.I_ScalerMethod; 
I_ScalerParam  = vRCEDUMDAparameters.I_ScalerParam;

I_SamplingMethod = vRCEDUMDAparameters.I_SamplingMethod; 
I_SamplingParam  = vRCEDUMDAparameters.I_SamplingParam;

I_maxFits     = vRCEDUMDAparameters.I_maxFits; 
I_repeatEvals = vRCEDUMDAparameters.I_repeatEvals; 

global I_evaluationProxy                 ;
I_evaluationProxy                 = vRCEDUMDAparameters.I_evaluationProxy;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnc          =  otherParameters.fnc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fitMaxVector = nan(1,I_itermax);

[~,solSize] = size(FVr_minbound); 

global FullStoreSol;  
global FullStoreVal;  
global FullStoreCount;

if (I_repeatEvals == 0)
    FullStoreSol   = zeros(I_NP * I_itermax,solSize);
    FullStoreVal   = zeros(1,I_NP * I_itermax);
    FullStoreCount = 0;
end 
if (I_repeatEvals == 2)
    FullStoreSol   = zeros(I_elitismParam,solSize);
    FullStoreVal   = zeros(1,I_elitismParam);
    FullStoreCount = 0;
end 


global EvaluationProxyInitTrainGenCount;  
global EvaluationProxyEvalGenCount;  
global EvaluationProxyTrainGenCount;
global EvaluationProxyState;

EvaluationProxyInitTrainGenCount = 0;  
EvaluationProxyEvalGenCount      = 0;  
EvaluationProxyTrainGenCount     = 0;
EvaluationProxyState             = 0;

global Train_X;
global Train_Y;
global Train_Count;
global ModelTrained;

[~,featuresSize] = size(FVr_minbound);

Train_X = nan(vRCEDUMDAparameters.maxEvals, featuresSize);
Train_Y = nan(vRCEDUMDAparameters.maxEvals, 1); 
Train_Count = 0;

global group_importance;
group_importance=zeros(1,I_D);
delta_OF=zeros(1,5);
global group_sizes;
group_sizes=[0,42,500,25,2,1];

gen = 1; 
evalCount = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gp_popSize = 70;
gp_maxDepth = 8;
gp_crossoverProb = 0.5;
gp_mutationProb = 0.1;
gp_elitismCount = 14;

gp_fitness = zeros(1, gp_popSize);

new_population = cell(1, gp_popSize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----FM_pop is a matrix of size I_NPx(I_D+1). It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------
% FLC modification - vectorization
minPositionsMatrix=repmat(FVr_minbound,I_NP,1);
maxPositionsMatrix=repmat(FVr_maxbound,I_NP,1);

%code inicialized
codes = initializeCodes(FVr_minbound,FVr_maxbound);

% generate initial population.
rand('state',otherParameters.iRuns) %Guarantee same initial population

gp_population = initialize_gp_population(gp_popSize, gp_maxDepth,1);
deCodedPop = genpop(I_NP,I_D,minPositionsMatrix,maxPositionsMatrix);

for iiii = 1:I_lowPop
    deCodedPop(iiii,:) = FVr_minbound;
end    

S_val = my_feval(fnc,deCodedPop,caseStudyData, otherParameters,I_repeatEvals,No_evals);   
Full_val(gen,:)=S_val;
[~,I_best_index] = min(S_val); 
 
FVr_bestmemit = deCodedPop(I_best_index,:); 
fitMaxVector(1,gen)   = S_val(I_best_index);

enCodedPop = enCodePop(deCodedPop,codes,FVr_minbound,FVr_maxbound);
newEnCodedPop = enCodedPop;
CurPop=enCodedPop;

[~,pbidx_all] = sort(S_val);
pbidx = pbidx_all(1:I_elitismParam);    
pbpop = enCodedPop(pbidx,:);

 if (I_repeatEvals == 2)       
     FullStoreSol   = deCodedPop(pbidx,:);
     FullStoreVal   = S_val(pbidx);
     FullStoreCount = I_elitismParam; 
 end

eCount = 0;

while (eCount<I_itermax) && (evalCount < vRCEDUMDAparameters.maxEvals-I_NP) 
    b=1;
    if gen==60
        codes=initializeCodes1(codes,FVr_minbound,FVr_maxbound,FVr_bestmemit);
    end
    for cellIndexH = 1:I_StructureHeight
        for cellIndexW = 1:I_StructureWidth

            NeighborsIndexes          = getNeighborsIndexes(cellIndexH,cellIndexW,I_StructureHeight,I_StructureWidth,I_NeighborhoodType,I_NeighborhoodParam,I_indByCell);

            SelectedIndividualIndexes = getSelectedIndividualIndexes(NeighborsIndexes,S_val,I_SelectionMethod,I_Selected);        
            
            [Px,Pxy] = getEstimadedParams(enCodedPop,S_val,SelectedIndividualIndexes,codes,I_EstimationParamsMethod,I_EstimationParamsParam);
           
            [Dxgy,EdgesList,EdgesCount] = getEstimadedStructure(Px,Pxy,I_EstimationStructureMethod,I_EstimationStructureParam);
            
            [Px,Pxy] = getScaledParams(Px,Pxy,length(SelectedIndividualIndexes),I_ScalerMethod,I_ScalerParam);
             
            cellLinearIndex = I_StructureWidth * (cellIndexH-1) + cellIndexW;
            CellIndividualIndexes = getCellIndividualLinearIndexes(cellLinearIndex, I_indByCell); 
            
            NewCellPop = sampleNewIndividualsFromModel(I_indByCell,Px,Pxy,Dxgy,EdgesList,EdgesCount,I_SamplingMethod,I_SamplingParam);

            for ii = 1: I_indByCell            
                newEnCodedPop(CellIndividualIndexes(ii),:) = NewCellPop(ii,:);
            end
              
        end
    end
    
    enCodedPop = newEnCodedPop;
    
    for ii = 1: I_elitismParam            
        enCodedPop(pbidx(ii),:) = pbpop(ii,:); 
    end
    
    
    deCodedPop = deCodePop(enCodedPop,codes);
   
    %Evaluation of new Pop
    S_val = my_feval(fnc,deCodedPop,caseStudyData, otherParameters,I_repeatEvals,No_evals);
    
      
        % Metropolis 
            delta=Full_val-S_val;
            for k=1:I_NP  
               if mod(k, gp_popSize) ~= 0
                     b = mod(k, gp_popSize);%14
               end
                coeff=evaluate_gp_tree_on_gen(gp_population{b}, gen,delta(k));
              
                if coeff<0.1||coeff>1
                    gp_population{b}=build_gp_tree(gp_maxDepth, 0);
                    while evaluate_gp_tree_on_gen(gp_population{b},gen,delta(k))<0.6||evaluate_gp_tree_on_gen(gp_population{b},gen,delta(k))>1%0.1
                        gp_population{b} = build_gp_tree(gp_maxDepth, 0);
                    end
                end
                coeff=evaluate_gp_tree_on_gen(gp_population{b}, gen,delta(k));
                coeff = max(0.1, min(1.0, coeff)); 
                if delta(k) > 0 || rand <  coeff  %accept
                    CurPop(k,:) = enCodedPop(k,:);
                    Full_val(k)= S_val(k);
                    if Full_val(k) <= fitMaxVector(1,gen)
                        bestSol = CurPop(k,:);
                        deCodedPop(k,:)= deCodePop(CurPop(k,:),codes);
                       fitMaxVector(1,gen+1) = Full_val(k);
                    end
                end
            end
        
    [~,pbidx_all] = sort(Full_val);
    pbidx = pbidx_all(1:I_elitismParam);   
    pbpop = CurPop(pbidx,:);

    if (I_repeatEvals == 2)       
        FullStoreSol   = deCodedPop(pbidx,:);
        FullStoreVal   = Full_val(pbidx);
        FullStoreCount = I_elitismParam; 
    end   

    avgV   = mean(Full_val);
    worstV = max(Full_val);
        
    %% update best results
    [S_bestval,I_best_index] = min(Full_val);
    FVr_bestmemit = deCodedPop(I_best_index,:);

    fprintf('Best Worst Avg Fitness value: %f %f %f %f\n',S_val(I_best_index),S_bestval,avgV,worstV);
    fprintf('Generation: %d\n',gen);
    gen=gen+1; 
    val(gen,:)=Full_val;
    fitMaxVector(1,gen) = Full_val(I_best_index);

       
   if gen>7

        %GP
        delta_Full_val=val(gen-1,:)-val(gen,:);


        for jj=1:gp_popSize
            gp_fitness(jj)=sum(delta_Full_val(jj:gp_popSize:end))/(I_NP/gp_popSize);
        end

       for a=1:gp_popSize
            if rand < gp_crossoverProb
                parent1 = roulette_wheel_selection(gp_population, gp_fitness);
                parent2 = roulette_wheel_selection(gp_population, gp_fitness);
                child = crossover_gp_trees(parent1, parent2, gp_maxDepth);
            elseif rand < gp_mutationProb
                child = mutate_gp_tree(child, gp_maxDepth);
            else
                child = roulette_wheel_selection(gp_population, gp_fitness);
            end
           
            new_population{a} = child;
       end


        % =======================
        [~, sorted_indices] = sort(gp_fitness, 'descend');
        elite_count = min(gp_elitismCount, length(sorted_indices));
        elite_indices = sorted_indices(1:elite_count);
        new_population(end-elite_count+1:end) = gp_population(elite_indices); 
        % ========================

        gp_population = new_population;   
   end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Divide into 5 groups, each running one generation separately to calculate the fitness difference.
    if gen>=2 && gen<=6
        delta_OF(1,gen-1)=fitMaxVector(1,gen-1)-fitMaxVector(1,gen);
        fprintf('delta_OF: %d\n',delta_OF);
    end
    if gen==7
        for i=1:5
            if delta_OF(i)>0
                for j=1:24
                    group_importance(1+sum(group_sizes(1:i))+570*(j-1):570*(j-1)+sum(group_sizes(1:i+1)))=1+0.1*(delta_OF(i)-min(delta_OF))/(max(delta_OF)-min(delta_OF));
                end
            else
                for j=1:24
                    group_importance(1+sum(group_sizes(1:i))+570*(j-1):570*(j-1)+sum(group_sizes(1:i+1)))=1-(group_sizes(i+1)-min(group_sizes))/(max(group_sizes)-min(group_sizes))+0.01;
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eCount = eCount + 1;

end
Fit_and_p=[fitMaxVector(1,gen) 0]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plotResults(fitMaxVector);
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
function codes = initializeCodes(FVr_minbound,FVr_maxbound)
    ws = length(FVr_minbound);
    d=FVr_maxbound-FVr_minbound;
    codeSize= ceil(d)+1;
    codes = nan(max(codeSize),ws);
    for jjj=1:ws
           range = FVr_maxbound(jjj) - FVr_minbound(jjj);
           for iii=1:codeSize(jjj) 
               if codeSize(jjj)==1
                   codes(iii,jjj)=FVr_minbound(jjj);
               else
                    codes(iii,jjj) = FVr_minbound(jjj) + ( range / (codeSize(jjj) - 1.0) * (iii-1.0));
               end
           end
           
           if (codes(iii,jjj)>FVr_maxbound(jjj))
               codes(codeSize(jjj),jjj) = FVr_maxbound(jjj);
           end
    end
end    

function codes = initializeCodes1(codes,FVr_minbound,FVr_maxbound,FVr_bestmemit)
    ws = length(FVr_minbound);
      
    for jjj=1:ws
          hs=sum(~isnan(codes(:,jjj)));
        if hs==2 && FVr_bestmemit(jjj)==FVr_minbound(jjj)
            codes(hs,jjj)=0;
        elseif  hs==2 && FVr_bestmemit(jjj)==FVr_maxbound(jjj)
             codes(1,jjj)=0;
        end
    end
end    

function enCodedPop = enCodePop(pop,codes,FVr_minbound,FVr_maxbound)
    [hs,ws] = size(pop);
    enCodedPop = nan(hs,ws);
    for iii=1:hs
       enCodedPop(iii,:) = enCode(pop(iii,:),codes,FVr_minbound,FVr_maxbound);
    end
end    
    
function enCoded = enCode(indiv,codes,FVr_minbound,FVr_maxbound)
    [~,ws] = size(codes);
    
    enCoded = nan(1,ws);
    for jjj=1:ws
       hs=sum(~isnan(codes(:,jjj)));
       if (FVr_minbound(jjj)==FVr_maxbound(jjj)) 
            enCoded(1,jjj)=1;
       else        
            range = (FVr_maxbound(jjj) - FVr_minbound(jjj))/hs;
            s = FVr_minbound(jjj) + range;
            iii = 1;
            while (indiv(jjj) > s) && (iii<hs) 
                  iii = iii + 1;
                  s = s + range;
            end
            enCoded(1,jjj)=iii;
       end
    end
end    
    
function deCodedPop = deCodePop(enCodedPop,codes)
    [hs,ws] = size(enCodedPop);
    deCodedPop = nan(hs,ws);
    for iii=1:hs
       deCodedPop(iii,:) = deCode(enCodedPop(iii,:),codes);
    end
end    
    
function indiv = deCode(enCoded,codes)
   [~,ws] = size(codes);
    indiv = nan(1,ws);   
    for jjj=1:ws
        pos = enCoded(1,jjj);
        v   = codes(pos,jjj);
        indiv(1,jjj)=v;
    end
end    

function NeighborsIndexes=getNeighborsIndexes(~,cellIndexW,~,I_StructureWidth,I_NeighborhoodType,I_NeighborhoodParam,I_indByCell)
    switch I_NeighborhoodType
        case 1 % left-right 2k-1
            NeighborsCellIndexes = getNeighborsCellIndexes_LeftRight(cellIndexW,I_StructureWidth,I_NeighborhoodParam);   
        case 2 % Moore 8
            NeighborsCellIndexes = getNeighborsCellIndexes_Moore(cellIndexH, cellIndexW, I_StructureHeight, I_StructureWidth);
    end
    NeighborsIndexes = getCellIndividualLinearIndexes(NeighborsCellIndexes(1), I_indByCell);
    [~,ws] = size(NeighborsCellIndexes);
    for index = 2:ws
       M = getCellIndividualLinearIndexes(NeighborsCellIndexes(index), I_indByCell);
       NeighborsIndexes = [NeighborsIndexes M];
    end
end    

function SelectedIndividualIndexes = getSelectedIndividualIndexes(NeighborsIndexes,S_val,I_SelectionMethod,I_Selected)
    
    switch I_SelectionMethod
        case 1 % Trunc Selection
            SelectedIndividualIndexes = getTruncSelectedIndividualIndexes(NeighborsIndexes,S_val,I_Selected);   
        otherwise
            SelectedIndividualIndexes = [];
    end
end    


 
function [Px,Pxy] = getEstimadedParams(enCodedPop,S_val,SelectedIndividualIndexes,codes,I_EstimationParamsMethod,~)
    switch I_EstimationParamsMethod
        case 1    % 1 Bivariated
            %%%% Removed method. It is not relevant for RCED-UMDA
            % [Px,Pxy] = getEstimadedParamsBivariated(enCodedPop,SelectedIndividualIndexes,codes);
        otherwise % 0 Univariated
            [Px,Pxy] = getEstimadedParamsUnivariated(enCodedPop,S_val,SelectedIndividualIndexes,codes);    
    end
end    
        
function [Px,Pxy] = getEstimadedParamsUnivariated(FM_pop,~,SelectedIndividualIndexes,codes)
    parents = FM_pop(SelectedIndividualIndexes, :);
    [chs,cws] = size(codes); 
    [phs,~] = size(parents);
    Pxy = nan;
    
    Px   = codes;
    Px( (Px ~= 0) & ~isnan(Px) ) = 0;
    
    ivx = 1;
    while(ivx<=cws)
      
       id = 1;
       while (id <=phs)
          codeX = parents(id,ivx);
          Px(codeX,ivx) = Px(codeX,ivx)+1;              
          id = id + 1; 
       end
        
       ivx = ivx + 1; 
    end

end 
    
function [Dxgy,EdgesList,EdgesCount] = getEstimadedStructure(~,~,I_EstimationStructureMethod,~)
    switch I_EstimationStructureMethod
        case 1
            %%%% Removed method. It is not relevant for RCED-UMDA
            % [Dxgy,EdgesList,EdgesCount] = getEstimadedStructure_OrderPrecedenceMaxDependency(Px,Pxy,I_EstimationStructureParam); 
        otherwise
            Dxgy = zeros(1,1);
            EdgesList  = nan(1,1);
            EdgesCount = zeros(1,1);           
    end
end       
   
function [Px,Pxy] = getScaledParams(Px,Pxy,n,I_ScalerMethod,I_ScalerParam)
    switch I_ScalerMethod
        case 1 
           %%%% Removed method. It is not relevant for RCED-UMDA
           % [Px,Pxy] = getScaledParamsWithMinValue(Px,Pxy,n,I_ScalerParam);          
        case 2   
           [Px,Pxy] = getScaledParamsAddingValue(Px,Pxy,n,I_ScalerParam);
           
        otherwise    
                    
    end
end      
    
function [Px,Pxy] = getScaledParamsAddingValue(Px,Pxy,n,I_ScalerParam)
    [chs,cws] = size(Px);  
    group_matrix=zeros(1,I_D);
    if gen>=1 && gen<=5
        group_matrix=generate_group_matrix(gen,I_D);
    end

    singleAdd = I_ScalerParam * n / 3;
    doubleAdd = I_ScalerParam * n / (chs*chs);

    if gen<6
        Px  = singleAdd* group_matrix + Px;% group_matrix = 1 for the current group and 0 otherwise
    else
        Px  = singleAdd* group_importance + Px; 
    end

    Pxy = doubleAdd + Pxy;
end    

function group_matrix=generate_group_matrix(gen,I_D)
    group_matrix=zeros(1,I_D);
    offset = 0;

    if gen>=2 && gen<=6
        for i = 1:(gen-1)
            offset = offset + group_sizes(i);
        end
        group_size = group_sizes(gen);
        for j=1:24
            start_index = 1+offset+(j-1)*570;
            end_index =start_index+group_size-1;
            group_matrix(start_index:end_index)=1;
        end
    end
end
    
function NewCellPop = sampleNewIndividualsFromModel(I_indByCell,Px,~,~,~,~,I_SamplingMethod,~)
   switch I_SamplingMethod
        case 1
           %%%% Removed method. It is not relevant for RCED-UMDA
           % NewCellPop = sampleNewIndividualsFrom_OrderPrecedenceMaxDependency(I_indByCell,Px,Pxy,Dxgy,EdgesList,EdgesCount);
       otherwise    
           Px(isnan(Px)) = 0;
           NewCellPop = sampleNewIndividualsFromModelWithoutDependencies(I_indByCell,Px);         
   end
end      
    
function NewCellPop = sampleNewIndividualsFromModelWithoutDependencies(I_indByCell,Px)
    [~,vws] = size(Px);
   
     NewCellPop = rand(I_indByCell,vws);
    
     for ic = 1: I_indByCell            
        for iv = 1: vws
           NewCellPop(ic,iv) = rulete(NewCellPop(ic,iv), Px(:,iv));
        end
     end
end     
             
function value = rulete(rv, Counters) 
    len = length(Counters); 
    
    total = 0;
    for value=1:len
        total = total + Counters(value);
    end
    
    rv = rv * total;
    
    sumsum = 0;
    for value=1:len
        sumsum = sumsum + Counters(value);
        if (rv<=sumsum+eps(sumsum)) 
            return;
        end
    end
    value = len;
end    
   
function NeighborsIndexes = getNeighborsCellIndexes_LeftRight(cellIndexW,I_StructureWidth,I_NeighborhoodParam)
    NeighborsIndexes = nan(1,2*I_NeighborhoodParam+1,'single');
    for index = 1:2*I_NeighborhoodParam+1
        NeighborsIndexes(index) = cellIndexW - I_NeighborhoodParam - 1 + index;
        if (NeighborsIndexes(index) < 1)
           NeighborsIndexes(index) = I_StructureWidth + NeighborsIndexes(index);
        end
        if (NeighborsIndexes(index) > I_StructureWidth)
           NeighborsIndexes(index) = NeighborsIndexes(index) - I_StructureWidth;
        end  
    end
end 


function CellIndividualLinearIndexes = getCellIndividualLinearIndexes(cellIndex,I_indByCell)
    CellIndividualLinearIndexes = nan(1,I_indByCell,'single');
    base = (cellIndex-1) * I_indByCell;
    for index = 1:I_indByCell
       CellIndividualLinearIndexes(index) = base + index;
    end
end    

function SelectedIndividualIndexes = getTruncSelectedIndividualIndexes(NeighborsIndexes,S_val,I_Selected)
    [~,niS] = size(NeighborsIndexes);
    NeighborsVal = nan(1,niS,'single');
    for index = 1:niS
        NeighborsVal(index)=S_val(NeighborsIndexes(index));
    end
    [~,I] = sort(NeighborsVal);

    SelectedIndividualIndexes = nan(1,I_Selected);
    for index = 1:I_Selected
        SelectedIndividualIndexes(index)=NeighborsIndexes(I(index));
    end
end    
 
function plotResults(Results)
    [~,ws] = size(Results);
    X=1:1:ws;
    Y=Results;
    plot(X,Y); 
    grid on
end    

% VECTORIZED THE CODE INSTEAD OF USING FOR
function pop=genpop(a,b,lowMatrix,upMatrix)
    pop=unifrnd(lowMatrix,upMatrix,a,b);
end

function [Val, evalCount] = my_feval(fnc,deCodedPop,caseStudyData, otherParameters, I_repeatEvals,No_evals)    
    switch I_repeatEvals
        case 0
           %%%% Removed method. It is not relevant for RCED-UMDA
           % [Val, evalCount] = my_feval_withStore(fnc,deCodedPop,caseStudyData, otherParameters, evalCount);
        case 2
           Val = my_feval_withStoreForElitism(fnc,deCodedPop,caseStudyData, otherParameters,No_evals); 
        otherwise    
            [s,~] = size(deCodedPop);    
            evalCount = evalCount + s;
            [Val, ~] = feval(fnc,deCodedPop,caseStudyData, otherParameters,No_evals);
            fprintf('evalCount: %d\n',evalCount);       
    end
end  

function Val = my_feval_withStoreForElitism(fnc,deCodedPop,caseStudyData, otherParameters,No_evals)    
     [cs,ws] = size(deCodedPop);
     Val    = nan(1,cs);     
     
     count = 0;
     xPop  = nan(cs,ws);
     xIndex = nan(cs,1);
     
     for iii=1:cs
        index = searchSolution(deCodedPop(iii,:),ws);
        if index > 0
            Val(iii) = FullStoreVal(index);           
        else
           count = count + 1;
           xIndex(count) =  iii;
           xPop(count,:) = deCodedPop(iii,:);                   
        end
     end
     
     xxPop = xPop(1:count,:);
     xxIndex = xIndex(1:count);
     
     auxVal = feval_proxy(fnc,xxPop,caseStudyData, otherParameters,No_evals);
     
     Val(xxIndex) = auxVal(:);
     
     fprintf('evalCount: %d\n',evalCount);
end

function newPop = reduceToDecisionFeatures(pop)
   newPop = pop(:,decisionFeatures); 
end

function Val = feval_proxy(fnc,pop,caseStudyData, otherParameters,No_evals) 
    [s1,s2] = size(pop);
    
    if (I_evaluationProxy == 0)
        %% evaluo f 
        %% fprintf('Eval ...\n');
        Val    = nan(1,s1); 
        for index=1:s1
           %fprintf('index: %d\n',index); 
           [auxVal, ~] = feval(fnc,pop(index,:),caseStudyData, otherParameters,No_evals);
           Val(index) = auxVal;
           evalCount = evalCount + 1; 
           
        end
    else
       %%%% Removed block. It is not relevant for this submission.
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end   
end
     
function index = searchSolution(sol, ws)
     index = 0; 
    
     for iii=1:FullStoreCount
                 
         flag = 1;
         for jjj=1:ws
            if FullStoreSol(iii,jjj)~=sol(jjj)
              flag = 0;
              break;
            end
         end
         
         if flag == 1
            index = iii;
            fprintf('*');
            break; 
         end         
     end
end            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% GP
function population = initialize_gp_population(popSize, maxDepth,delta)
    population = cell(1, popSize);
    for i = 1:popSize
        population{i} = build_gp_tree(maxDepth, 0);
        while evaluate_gp_tree_on_gen(population{i},gen,delta)<0.1||evaluate_gp_tree_on_gen(population{i},gen,delta)>1%0.6
            population{i} = build_gp_tree(maxDepth, 0);
        end
    end
end

function tree = build_gp_tree(maxDepth, currentDepth)
    func_set = {'add', 'sub', 'mul', 'div', 'sin', 'cos','exp'};
    
    if currentDepth >= maxDepth || (currentDepth > 0 && rand < 0.3)
        if rand < 0.8 
            if rand<0.5
                tree = {'delta'};
            else
                tree = {'gen'};
            end
        else
            tree = {num2str(0.1 + 0.9 * rand)};
        end
    else
        op_idx = randi(length(func_set));
        op = func_set{op_idx};
        switch op
            case { 'sin', 'cos','exp'}
                child = build_gp_tree(maxDepth, currentDepth + 1);
                tree = {op, child};
            otherwise
                left = build_gp_tree(maxDepth, currentDepth + 1);
                right = build_gp_tree(maxDepth, currentDepth + 1);
                tree = {op, left, right};
        end
    end
end


function value = evaluate_gp_tree_on_gen(tree, gen,delta)
    if iscell(tree) && numel(tree) == 1
        tree = tree{1};
    end
    
    if iscell(tree)
        op = tree{1};
        switch op
            case 'add'
                left = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                right = evaluate_gp_tree_on_gen(tree{3}, gen,delta);
                value = left + right;
            case 'sub'
                left = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                right = evaluate_gp_tree_on_gen(tree{3}, gen,delta);
                value = left - right;
            case 'mul'
                left = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                right = evaluate_gp_tree_on_gen(tree{3}, gen,delta);
                value = left * right;
            case 'div'
                left = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                right = evaluate_gp_tree_on_gen(tree{3}, gen,delta);
                value = safe_division(left, right);
            case 'sin'
                arg = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                value = sin(arg);
            case 'cos'
                arg = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                value = cos(arg);
            case 'exp'
                arg = evaluate_gp_tree_on_gen(tree{2}, gen,delta);
                value = exp(arg);
            otherwise
                error('Unknown operator: %s', op);
        end
    else
        if ischar(tree) && strcmp(tree, 'gen')
            value = gen;
        else
            if ischar(tree)
                value = str2double(tree);
            else
                value = tree;
            end
            if isnan(value)
                value = 0;  
            end
        end
    end
    
    if ~isscalar(value)
        value = mean(value(:)); 
    end
end
    function selected = roulette_wheel_selection(population, fitness)
     if any(fitness < 0)
        min_negative = min(fitness);
        fitness=fitness+abs(min_negative);
     end

    total_inv = sum(fitness);
    
    prob = fitness / total_inv;
    
    cum_prob = cumsum(prob);
    
    r = rand();
    idx = find(r <= cum_prob, 1, 'first');
    
    selected = population{idx};
end

function child = crossover_gp_trees(parent1, parent2, maxDepth)
    [node1, ~, ~] = select_random_node(parent1);
    [node2, ~, ~] = select_random_node(parent2);
    
    child = replace_node(parent1, node1, node2);
    
    if tree_depth(child) > maxDepth
        child = parent1;
    end
end

function mutated = mutate_gp_tree(tree, maxDepth)
    [node, ~, ~] = select_random_node(tree);
    
    new_subtree = build_gp_tree(4, 0);
    
    mutated = replace_node(tree, node, new_subtree);
    
    if tree_depth(mutated) > maxDepth
        mutated = tree;
    end
end

function [selected_node, parent, path] = select_random_node(tree)
    total_nodes = count_nodes(tree);
    node_index = randi(total_nodes);
    current_count = 0;
    [selected_node, parent, path] = find_node_by_index(tree, {}, [], node_index, current_count);
end

function count = count_nodes(tree)
    if ~iscell(tree) || length(tree) == 1
        count = 1;
    else
        count = 1;
        for i = 2:length(tree)
            count = count + count_nodes(tree{i});
        end
    end
end

function [node, parent, path, current_count] = find_node_by_index(tree, parent_tree, current_path, target_index, current_count)
    current_count = current_count + 1;
    if current_count == target_index
        node = tree;
        parent = parent_tree;
        path = current_path;
        return;
    end
    
    if iscell(tree) && length(tree) > 1
        for i = 2:length(tree)
            new_path = [current_path, i];
            [node, parent, path, current_count] = find_node_by_index(...
                tree{i}, tree, new_path, target_index, current_count);
            if current_count >= target_index
                return;
            end
        end
    end
    
    node = [];
    parent = {};
    path = [];
end

   function new_tree = replace_node(tree, node_to_replace, new_subtree)
    if isequal(tree, node_to_replace)
        new_tree = new_subtree;
        return;
    end
    
    if ~iscell(tree)
        new_tree = tree;
        return;
    end
    
    new_tree = tree; 
    
    for i = 1:length(new_tree)
        current_node = new_tree{i};
        
        if iscell(current_node)
            new_tree{i} = replace_node(current_node, node_to_replace, new_subtree);
        else
            if isequal(current_node, node_to_replace)
                new_tree{i} = new_subtree;
            end
        end
    end
end

function depth = tree_depth(tree)
    if ~iscell(tree) || length(tree) == 1
        depth = 1;
    else
        max_child_depth = 0;
        for i = 2:length(tree)
            child_depth = tree_depth(tree{i});
            if child_depth > max_child_depth
                max_child_depth = child_depth;
            end
        end
        depth = 1 + max_child_depth;
    end
end

function value = safe_division(a, b)
    if abs(b) < 1e-6
        value = a;
    else
        value = a / b;
    end
end

end