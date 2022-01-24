clc, clear, close all

%% Ucitavanje podataka iz .csv dokumneta
data = readtable('Genres.csv');

% Ucitavanje ulaza u vidu matrice iz .csv dokumenta
input = [data.danceability,data.energy,data.key,data.loudness,data.mode,...
         data.speechiness,data.acousticness,data.instrumentalness,...
         data.liveness,data.valence,data.tempo,];

% Ekstrakcija podataka i priprema za crtanje histograma
% gde 1 oznacava "Rap",2 "Pop", a 3 "RnB" muzicki zanr
output = zeros(length(data.genre),1);
output(data.genre=="Rap") = 1;
output(data.genre=="Pop") = 2;
output(data.genre=="RnB") = 3;

% Transponovanje tako da ulazi budu vrste, a u kolonama podaci za
% odgovarajuci ulaz
input  = input';
output = output';


% Podela podataka po klasama
K1 = input(:,output==1);
K2 = input(:,output==2);
K3 = input(:,output==3);

%% Prikaza raspodele odbiraka po klasama
figure
histogram(output);

%% Kodiranje izlaza kao one-hot encoding
output_OH = zeros(length(data.genre),3);
output_OH(data.genre=="Rap",1) = 1;
output_OH(data.genre=="Pop",2) = 1;
output_OH(data.genre=="RnB",3) = 1;

% Transponovanje matrice 
output_OH = output_OH'; 

%% Podela podataka ne-balansiranih klasa na trening,test i validacioni skup
%  70% je trening, 15% je test, 15% je validacija

%  Podela klase 1
N1 = length(K1);
K1_training = K1(:, 1 : cast(0.7*N1,'uint16'));
K1_test = K1(:,cast(0.7*N1+1,'uint16') : cast(0.85*N1,'uint16'));
K1_val = K1(:, cast(0.85*N1+1,'uint16') : N1);

%  Podela klase 2
N2 = length(K2);
K2_training = K2(:, 1 : cast(0.7*N2,'uint16'));
K2_test = K2(:,cast(0.7*N2+1,'uint16') : cast(0.85*N2,'uint16'));
K2_val = K2(:, cast(0.85*N2+1,'uint16') : N2);

%  Podela klase 3
N3 = length(K3);
K3_training = K3(:, 1 : cast(0.7*N3,'uint16'));
K3_test = K3(:,cast(0.7*N3+1,'uint16') : cast(0.85*N3,'uint16'));
K3_val = K3(:, cast(0.85*N3+1,'uint16') : N3);

%% Formiranje zajednockog trening, test i validacionog skupa

N1_training_indx = length(K1_training);
N2_training_indx = N1 + length(K2_training);
N3_training_indx = N1 + N2 + length(K3_training);

N1_val_indx = N1_training_indx + length(K1_val);
N2_val_indx = N2_training_indx + length(K2_val);
N3_val_indx = N3_training_indx + length(K3_val);

%  Ulaz trening
input_training = [K1_training,K2_training,K3_training];

%  Izlaz trening
output_training = output_OH(:,1:N1_training_indx);
output_training = [output_training,output_OH(:,N1+1:N2_training_indx)];
output_training = [output_training,output_OH(:,N1+N2+1:N3_training_indx)];    
                
%  Ulaz validacija
input_val = [K1_val,K2_val,K3_val];

%  Izlaz validacija
output_val = output_OH(:,N1_training_indx+1:N1_val_indx);
output_val = [output_val,output_OH(:,N2_training_indx+1:N2_val_indx)];
output_val = [output_val,output_OH(:,N3_training_indx+1:N3_val_indx)];    

%  Ulaz test
input_test=[K1_test,K2_test,K3_test];

%  Izlaz test
output_test = output_OH(:,N1_val_indx+1:N1);
output_test = [output_test,output_OH(:,N2_val_indx+1:N1+N2)];
output_test = [output_test,output_OH(:,N3_val_indx+1:N1+N2+N3)];
          
input_all=[input_training,input_val];
output_all=[output_training,output_val];

weight = ones(3, length(output_all));
weight(2,1294:1616) = 5;

%% Krosvalidacija
%arhitektura = {[8,5,4],[6,3,2],[7,5,3,3]};
arhitektura = {[100,50,40,10,10]};
arh_best = 0;
A_best = 0;
k1_best=0;
k2_best=0;

for reg = [0.1,0.2]
    for w = [1.5,2,3,5,8]
        for k1 = [1.1 ,1.2, 1.3]
            for k2 = [0.2,0.3,0.5]
                for arh = 1:length(arhitektura)
                rng(5)
                net = patternnet(arhitektura{arh});
                
                %delimo skup podataka po indeksima
                net.divideFcn = 'divideind';
                %indeksi za trening podatke
                net.divideParam.trainInd = 1 : length(input_training);
                %indeksi za validacione podatke
                net.divideParam.valInd = length(input_training)+1 :...
                                                length(input_all);
                %test podatke ne dajemo mrezi odmah                            
                net.divideParam.testInd = [];

                net.performParam.regularization = reg;
                
                net.trainFcn = 'trainrp';
                
                %konstanta obucavanja
                %net.trainParam.lr = lr;
                %broj epoha obucavanja
                net.trainParam.epochs = 3000;
                %masksimalna dozvoljena greska 
                net.trainParam.goal = 1e-2;
                net.trainParam.max_fail = 300;
                %graficki prikaz obucavanja
                net.trainParam.showWindow =false ;

                net.trainParam.delt_inc =k1;
                net.trainParam.delt_dec=k2;  
                net.trainParam.deltamax=40;
                
                
                weight = ones(3, length(output_all));
                weight(2,1294:1616)= w;
                
                [net, info] = train(net, input_all, output_all,[],[],weight);
                %predikcija se vrsi na validacionim podacima jer oni nisu
                %ucestvovali u obucavanju mreze
                pred = sim(net,input_val);
                pred = round(pred);
                
                %konfuziona matrica
                [~, cm] = confusion(output_val, pred);
                %Preciznost, procenat tacno klasifikovanih podatak kao suma
                %elemenata na galvnoj dijagonali kroz suma svih elemenata
                A = 100*sum(trace(cm))/sum(sum(cm));

                disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ',k1=' num2str(k1) ])
                disp(['epoch = ' num2str(info.best_epoch) ',k2=' num2str(k2) ',w=' num2str(w)])
                disp('---------------------');
                if A > A_best
                    A_best = A;
                    reg_best = reg;
                    w_best = w;
                    k1_best=k1;
                    k2_best=k2;
                    arh_best = arhitektura{arh};
                    ep_best = info.best_epoch;
                end
                end
            end
         end
    end

end
disp('Kraj');
%%
net = patternnet(arh_best);

net.divideFcn = '';

net.performParam.regularization = reg_best;

net.trainFcn = 'trainrp';

net.trainParam.epochs = ep_best;
net.trainParam.goal = 1e-2;
net.trainParam.delt_inc =k1_best;
net.trainParam.delt_dec=k2_best; 
net.trainParam.deltamax=40;

weight = ones(3, length(output_all));
weight(2,1294:1616)= w_best;

[net, info] = train(net, input_all, output_all,[],[],weight);

%% Performanse NM
pred = sim(net, input_test);
figure, plotconfusion(output_test, pred);

[~, cm] = confusion(output_test, pred);
A = 100*sum(trace(cm))/sum(sum(cm));
disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ])
disp(['epoch = ' num2str(info.best_epoch)])