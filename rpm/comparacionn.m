%% Plot features
% Change default axes fonts.
set(0,'DefaultAxesFontName', 'Arial')
set(0,'DefaultAxesFontSize', 16)
% Change default text fonts.
set(0,'DefaultAxesFontName', 'Arial')
set(0, 'DefaultAxesFontWeight','Bold')
set(0,'DefaultTextFontSize', 16)

%% Prepare
clear all; close all; 
normal=load('rpm_total_buenos.txt'); 
falla=load('rpm_total_falla.txt'); 
input=[falla;normal]; % mezclasmos los datos de falla y los normales para obtener las entradas
targets=[ones(3000,1);zeros(3000,1)]; %se crean los objetivos con la mezcla de la mitad de valores con 1 que significa falla y la otra mitad con 0 que no hay falla
%%
%%mezclamos la datos
save('input','input');
save('targets','targets');
y=randperm(6000); %mezclamos los datos
input=input(y,:);
targets=targets(y,:);
%targets(1:100) para probar la mezcla
save('input','input');
save('targets','targets');
%%
Ent_Test=1000; % datos para usar en el entranimiento (los primeros 1000)
%Targets=rpm_total_buenos; % 0--> benigno ; 1 --> maligno.
%Inputs=rpm_total_falla; % row x column = Característica x Instancia
%% alistar la matriz para el entrenamiento
targets=targets'; % para hacerle como columna y pueda calcular
input=input';
%%
%%Clustering K-means
I=load('rpm_total_falla.txt'); % seleccion de datos
X=(I-min(I(:)))./(max(I(:)-min(I(:)))); % datos entre 0 y 1
figure(1)
plot(X(:,1),X(:,2),'.');
title('DATOS DE RPM');
dist_k='sqeuclidean'; % metodo de distancia
opts=statset('Display','off');
[idx,C,sumd,D]=kmeans(X,4,'Distance',dist_k,...
    'Replicates',5,'Options',opts); % metodo K-means
figure(2)
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
hold on
plot(X(idx==4,1),X(idx==4,2),'y.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
    'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4',...
    'Centroids','Location','NW')
title '2D REPRESENTACION WITH 4 CLUSTER'
figure(3)
[silh4,h]=silhouette(X,idx,dist_k);
xlabel 'Valor Silhouette por classif. cluster'
ylabel 'Cluster'
s=silhouette(X,idx,dist_k);
media=mean(s)
save Kmeanssumd.tt sumd -ascii -double
save kmeansD.txt D -ascii -double
save kmeansC.txt C -ascii -double
save kmeansidx.txt idx -ascii -double

D=sum (sumd);
%% Entrenamiento con Naive Bayes (NB)
%
ML_NB=fitcnb(input(:,1:Ent_Test)',targets(:,1:Ent_Test)');
FD_NB=predict(ML_NB,input(:,Ent_Test:end)');
%
figure(4)
plotconfusion(targets(:,Ent_Test:end),FD_NB','NAIVE BAYES: ');
%
%% Entrenamiento con support vector machines (SVM)
%
ML_SVM=fitcsvm(input(:,1:Ent_Test)',targets(:,1:Ent_Test)',...
    'KernelFunction','rbf');
FD_SVM=predict(ML_SVM,input(:,Ent_Test:end)');
%
figure(5)
plotconfusion(targets(:,Ent_Test:end),FD_SVM','SUPPORT VECTOR MACHINES: ');
%
%% Entrenamiento con decision trees (DT)
%
ML_DT=fitctree(input(:,1:Ent_Test)',targets(:,1:Ent_Test)');
FD_DT=predict(ML_DT,input(:,Ent_Test:end)');
%
figure(6)
plotconfusion(targets(:,Ent_Test:end),FD_DT','ARBOL DE DECISIONES: ');
%
%% Entrenamiento con artificial neural networks (ANN)
%
ML_ANN = fitnet(10);
ML_ANN=train(ML_ANN,input(:,1:Ent_Test),targets(:,1:Ent_Test));
FD_ANN=round(ML_ANN(input(:,Ent_Test:end))');
%
figure(7)
plotconfusion(targets(:,Ent_Test:end),FD_ANN','RED NEURONAL: ');
%
%% Comparación
%
Counter_SVM=0;
Counter_DT=0;
Counter_NB=0;
Counter_ANN=0;
j=0;
%
for i=1:size(FD_NB,1)
    
    Test=size(targets(:,Ent_Test:end),2);
    
    if targets(1,Ent_Test-1+i) == FD_NB(i,1)
        Counter_NB=Counter_NB+1;
    end
    
    if targets(1,Ent_Test-1+i) == FD_SVM(i,1)
        Counter_SVM=Counter_SVM+1;
    end
    
    if targets(1,Ent_Test-1+i) == FD_DT(i,1)
        Counter_DT=Counter_DT+1;
    end
    
    if targets(1,Ent_Test-1+i) == FD_ANN(i,1)
        Counter_ANN=Counter_ANN+1;
    end
end
%
disp(['Performance NB : ' num2str(Counter_NB) '/' num2str(Test)]);
disp(['Performance SVM : ' num2str(Counter_SVM) '/' num2str(Test)]);
disp(['Performance DT : ' num2str(Counter_DT) '/' num2str(Test)]);
disp(['Performance ANN : ' num2str(Counter_ANN) '/' num2str(Test)]);
