function [Y] = NelderMeadSearch (Problem,P,W,Bi,i,Z)
% The process of multi-objective Nelder Mead Search
% Problem : the problem solved
% P       : current population
% W       : the weight vectors
% sub     : the subproblem number
% Z       : the ideal point

%% Control paprameters
alpha = 1;
gamma = 2;
beta  = 0.5;

%% Set the vertices indices
num_vert  = Problem.D+1;
Bi(Bi==i) = [];
if length(Bi) < num_vert
    NB = randperm(Problem.N);
    NB(NB==i) = [];
    NB = setdiff(NB,Bi);
    Bi = [Bi,NB(1:num_vert-length(Bi))];
elseif length(Bi) > num_vert
    Bi = Bi(1:num_vert);
end
Pop = P(Bi);

%% Sort the vertices
Fit      = max(abs(Pop.objs-repmat(Z,num_vert,1)).*repmat(W(i,:),num_vert,1),[],2);
[~,rank] = sort(Fit);
Pop      = Pop(rank);
Y        = [];

%% The centroid point
Xc   = mean(Pop(1:end-1).decs,1);
Xc   = Repair(Xc,Problem.lower,Problem.upper);
Pc   = Problem.Evaluation(Xc);
%Fitc = max(abs(Pc.objs-Z).*W(i,:),[],2);
Y    = [Y;Pc];

%% Reflection
Xr   = (1+alpha)*Xc-alpha*Pop(end).decs;
Xr   = Repair(Xr,Problem.lower,Problem.upper);
Pr   = Problem.Evaluation(Xr);
Fitr = max(abs(Pr.objs-Z).*W(i,:),[],2);
Y    = [Y;Pr];

if Fit(1) <= Fitr && Fitr <= Fit(end-1);
else
   %% Expansion
   Xe   = (1+alpha*gamma)*Xc-alpha*gamma*Pop(end).decs;
   Xe   = Repair(Xe,Problem.lower,Problem.upper);
   Pe   = Problem.Evaluation(Xe);
   Fite = max(abs(Pe.objs-Z).*W(i,:),[],2);
   Y    = [Y;Pe];
   if Fite < Fitr && Fitr < Fit(1);
   else
       %% outside contraction
       Xco   = (1+alpha*beta)*Xc-alpha*beta*Pop(end).decs;
       Xco   = Repair(Xco,Problem.lower,Problem.upper);
       Pco   = Problem.Evaluation(Xco);
       Fitco = max(abs(Pco.objs-Z).*W(i,:),[],2);
       Y     = [Y;Pco];
       if Fit(end-1) <= Fitr && Fitr < Fit(end) && Fitco <= Fitr;
       else
           %%  inside contraction
           Xci    = (1-beta)*Xc+beta*Pop(end).decs;
           Xci   = Repair(Xci,Problem.lower,Problem.upper);
           Pci   = Problem.Evaluation(Xci);
           %Fitco = max(abs(Pci.objs-Z).*W(i,:),[],2);
           Y     = [Y;Pci];
       end
   end
end
Y = mutation (Problem,Y);
end

%**************************************************************************
function [Solution_new] = Repair(Solution,lower,upper)
[NP,D] = size(Solution);
Lower  = repmat(lower,NP,1);
Upper  = repmat(upper,NP,1);

New1   = Solution-rand(NP,D).*(Solution-Lower);
New2   = Solution+rand(NP,D).*(Upper-Solution);
Mask1  = Solution < Lower;
Mask2  = Solution > Lower;

Solution_new = Solution;
Solution_new(Mask1) = New1(Mask1);
Solution_new(Mask2) = New2(Mask2);
end
%**************************************************************************
function [Solutions] = mutation (Problem,Solutions)
disM = 20;
proM = 1;
Offspring = Solutions.decs;
[N,D] = size(Offspring);
Lower = repmat(Problem.lower,N,1);
Upper = repmat(Problem.upper,N,1);
Site  = rand(N,D) < proM/D;
mu    = rand(N,D);
temp  = Site & mu<=0.5;
Offspring       = min(max(Offspring,Lower),Upper);
Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
    (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
temp = Site & mu>0.5;
Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
    (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
Solutions = Problem.Evaluation(Offspring);
end




