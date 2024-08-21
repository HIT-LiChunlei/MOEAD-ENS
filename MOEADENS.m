classdef MOEADENS < ALGORITHM
    % <multi/many> <real/integer>
    % Ensemble of neighborhood search operators for decomposition-based multi-objective evolutionary optimization
    % K --- 5 --- Number of groups

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            K = Algorithm.ParameterSet(5);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);
            % Transformation on W
            W = 1./W./repmat(sum(1./W,2),1,size(W,2));
            % Maximum number of solutions replaced by each offspring
            nr = ceil(Problem.N/100);
            % Cluster the subproblems
            G = kmeans(W,K);
            G = arrayfun(@(S)find(G==S),1:K,'UniformOutput',false); % 生成的G为1x5cell,第一个cell表示的是在第一个聚类簇里面的权重向量的序号

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T); 

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Initial the CMA model
            sk    = cellfun(@(S)S(randi(length(S))),G); % 在G中每个cell中随机选择一个序号
            xk    = Population(sk).decs;
            Sigma = struct('s',num2cell(sk),'x',num2cell(xk,2)','sigma',0.5,'C',eye(Problem.D),'pc',0,'ps',0);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                for s = 1 : Problem.N
                    k = find([Sigma.s]==s);
                    if ~isempty(k)
                        P = B(s,randperm(size(B,2)));
                        % Generate offsprings by CMA
                        Offspring = Problem.Evaluation(mvnrnd(Sigma(k).x,Sigma(k).sigma^2*Sigma(k).C,4+floor(3*log(Problem.D))));
                        % Sort the parent and offsprings
                        Combine   = [Offspring,Population(s)];
                        [~,rank]  = sort(max(abs(Combine.objs-repmat(Z,length(Combine),1)).*repmat(W(s,:),length(Combine),1),[],2));
                        % Update the CMA model
                        Sigma(k)  = UpdateCMA(Combine(rank).decs,Sigma(k),ceil(Problem.FE/Problem.N));
                        if isempty(Sigma(k).s)
                            sk = G{k}(randi(length(G{k})));
                            Sigma(k).s = sk;
                            Sigma(k).x = Population(sk).dec;
                        end
                    else
                        % Generate an offspring by GA
                        if rand < 0.9
                            P = B(s,randperm(size(B,2)));
                        else
                            P = randperm(Problem.N);
                        end
                        Offspring = OperatorGAhalf(Problem,Population(P(1:2)));
                    end
                    for x = 1 : length(Offspring)
                        % Update the ideal point
                        Z = min(Z,Offspring(x).obj);
                        % Update the solutions in P by Tchebycheff approach
                        g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                        g_new = max(repmat(abs(Offspring(x).obj-Z),length(P),1).*W(P,:),[],2);
                        Population(P(find(g_old>=g_new,nr))) = Offspring(x);
                    end
                end
                if Problem.FE/Problem.maxFE >= 0.90 % Apply the NMS local search procedure
                    for n = 1 : Problem.N
                        Y  = NelderMeadSearch (Problem,Population,W,B(n,:),n,Z);
                        Z  = min([Z;Y.objs]);
                        NY = length(Y);
                        for y = 1 : NY
                            g_all       = max(abs(repmat(Y(y).objs-Z,Problem.N,1)).*W,[],2);
                            g_All       = max(abs(Population.objs-repmat(Z,Problem.N,1)).*W,[],2);
                            improve_all = (g_All-g_all)./g_all;
                            [~,index]   = min(improve_all);
                            if g_all(index) < g_all(index)
                                Population(index) = Y(y);
                            end
                        end                             
                    end
                end
            end
        end
    end
end