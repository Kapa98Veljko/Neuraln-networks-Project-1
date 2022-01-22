clc, clear, close all

%%
data = readtable('Genres.csv');

input  = [data.danceability,data.energy,data.key,data.loudness,data.mode,data.speechiness,data.acousticness,data.instrumentalness,data.liveness,data.valence,data.tempo,];

output = zeros(length(data.genre),1);
output(data.genre=="Rap") = 1;
output(data.genre=="Pop") = 2;
output(data.genre=="RnB") = 3;

input  = input';
output = output';


%%Podela podataka po klasama
K1 = input(:,output==1);
K2 = input(:,output==2);
K3 = input(:,output==3);

%%
figure
histogram(output);

output = zeros(length(data.genre),3);
output(data.genre=="Rap",1) = 1;
output(data.genre=="Pop",2) = 1;
output(data.genre=="RnB",3) = 1;

input  = input';
output = output'; 

N1 = length(K1);
K1trening = K1(:, 1 : cast(0.7*N1,'uint64'));
K1test = K1(:,cast(0.7*N1+1,'uint64') : cast(0.85*N1,'uint64'));
K1val = K1(:, cast(0.85*N1+1,'uint64') : N1);
%%
N2 = length(K2);
K2trening = K2(:, 1 : cast(0.7*N2,'uint64'));
K2test = K2(:,cast(0.7*N2+1,'uint64') : cast(0.85*N2,'uint64'));
K2val = K2(:, cast(0.85*N2+1,'uint64') : N2);

N3 = length(K3);
K3trening = K3(:, 1 : cast(0.7*N3,'uint64'));
K3test = K3(:,cast(0.7*N3+1,'uint64') : cast(0.85*N3,'uint64'));
K3val = K3(:, cast(0.85*N3+1,'uint64') : N3);

input_trening=[K1trening,K2trening,K3trening];
output_trening=output(:,1:length(K1trening));
output_trening=[output_trening,output(:,1+N1:length(K2trening)+N1)];
output_trening=[output_trening,output(:,1+N1+N2:length(K3trening)+N1+N2)];    

input_val=[K1val,K2val,K3val];
output_val=output(:,length(K1trening)+1:length(K1trening)+length(K1val));
output_val=[output_val,output(:,1+N1+length(K2trening):N1+length(K2trening)+length(K2val))];
output_val=[output_val,output(:,1+N1+N2+length(K3trening):length(K3trening)+N1+N2+length(K3val))];    

input_test=[K1test,K2test,K3test];
output_test=[ones(1,cast(0.15*N1,'uint64')),ones(1,cast(0.15*N2,'uint64'))+1,ones(1,cast(0.15*N3,'uint64'))+2];

input_all=[input_trening,input_val];
output_all=[output_trening,output_val];

%% Krosvalidacija
arhitektura = [5,2];
Abest = 0;
F1best = 0;

for reg = [0.1, 0.5, 0.9]
    for w = [2, 5, 10]
        for lr = [0.5, 0.05, 0.005]
            %for arh = length(arhitektura)
                rng(5)
                net = patternnet(arhitektura);

                net.divideFcn = 'divideind';
                net.divideParam.trainInd = 1 : length(input_trening);
                net.divideParam.valInd = length(input_trening)+1 : length(input_all);
                net.divideParam.testInd = [];

                net.performParam.regularization = reg;
                
                net.trainFcn = 'traingd';

                net.trainParam.lr = lr;
                net.trainParam.epochs = 100;
                net.trainParam.goal = 1e-4;
                net.trainParam.max_fail = 20;
                net.trainParam.showWindow =true ;

                weight = ones(1, length(output_all));
                weight(output_all == 2) = w;
                
               [net, info] = train(net, input_all, output_all, [], [], weight);
               
               pred = sim(net,input_val);
               pred = round(pred);
               
               [~, cm] = confusion(output_val, pred);
               A = 100*sum(trace(cm))/sum(sum(cm));
               F1 = 2*cm(2, 2)/(cm(2, 1)+cm(1, 2)+2*cm(2, 2));

               disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ', F1 = ' num2str(F1)])
               disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])

               if F1 > F1best
                   F1best = F1;
                   Abest = A;
                   reg_best = reg;
                   w_best = w;
                   lr_best = lr;
                   %arh_best = arhitektura{arh};
                   ep_best = info.best_epoch;
                end
            %end
        end
    end
end

