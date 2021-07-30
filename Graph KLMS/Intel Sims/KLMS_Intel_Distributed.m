clear all; clc;

M = 5; %Filter order


kernel_bw = 20;
kernel = @(z,zin) exp(-(norm(z-zin)^2)/kernel_bw);

% NETWORK
intel_dataset = load('intel_data_and_structure_nn4.mat');

G = intel_dataset.G; %graph
W = intel_dataset.W; %shift/weight matrix (gaussian from distances)
S = W;

N = G.N;

A = full(G.A);
L = full(G.L);

% Computing combination coefficients
c=eye(N) + A;
count=sum(c');
CC = zeros(N,N);
for i=1:N
    for j=1:N
        if(i~=j)
            if c(i,j)~=0
                Cc(i,j)=1/(max(count(i),count(j)));
            else
                Cc(i,j)=0;
            end
        else
            Cc(i,j)=0;
        end
    end
    Cc(i,i)= 1 - sum(Cc(i,:));
end
C_nl = Cc;

% 
% W = rand(N,N);
% W = (W+W')/2;
% S = W.*A;
[VS,DS] = eig(S);
S = VS*( DS./max(diag(abs(DS))))*VS';
S(S<10^-5) = 0;    

% SIGNAL PARAMS    
% sv=load('sv');
% sigma_x = sv.sigma_x;

% nv=load('nv');
sigma_v = 0.1+rand(N,1)/20;

% Rx = diag(sigma_x);
Rv = diag(sigma_v);


data = load('intel_data_5min_march.mat');
X = data.Xtemp;
Y = data.Xhumid;
time_stamps = data.Time_stamps;

Yrff = zeros(size(Y));


for mu = 4 %Iterates over different step sizes
    tic
    rng(4,'v5uniform');
    % GENERAL PARAMETERS
    T = length(X); %Total time

    number_of_runs = 1; 
    MSE_average_cc = zeros(1,T);
    MSE_average_rf = zeros(1,T);
    MSE_average_lms = zeros(1,T);
    % mu = 0.5;

    % PARAMS CC
    cc_param = 0.1; %0.1 for 32, 0.2 or 0.3 for 64, 0.4 for 128
    dict_target = 32; %Targeted dictionary size

    % PARAMS RF
    D = dict_target; %Making RFF-space dimension equal to the dictionary size

    % CC DICTIONARY TRAINING
    Xtrain = X;
    dict = zeros(1,M);
    while length(dict)~=dict_target
        for t=1:length(Xtrain)
            for m = 0:min(t-1,M-1)  % ICASSP Z
                Ztrain(:,m+1,t) = (S^m)*Xtrain(:,t-m);
            end
        end
        
        for t=5:length(Xtrain)    
            for k = 1:N
                zk = Ztrain(k,:,t);
                kvec = exp(-(vecnorm(dict-zk,2,2).^2)/kernel_bw);
                if max(kvec) <= cc_param
                    dict = [dict;zk];
    %                 fprintf('size %i/n',length(dict));
                end
                if ~all(dict(1,:)), dict(1,:)=[]; end
                if length(dict) == dict_target, break; end
            end
            if length(dict) == dict_target, break; end
        end
    end
    

    % RF MAPPING CONSTRUCTION ( [cos(vp*z + b) ... ] )
    Vp = sqrt(kernel_bw^-1)*randn(M,D); % sampled vectors for RF
    b = 2*pi*rand(D,1); % random phase

    % Kernel LMS MC iterations
    for run=1:number_of_runs % MC iteration
        fprintf('run: %i - ', run);

        % INPUT X AND NOISE V    
        V = mvnrnd(zeros(N,1),Rv,T)';

        % Definition of Z
        % Non-linear filter 
        Z = zeros(N,M,T);

        % Coherence-check variables
        alpha = mat2cell(zeros(dict_target,N),dict_target,ones(1,N));
        beta = alpha;
        kvec = alpha;

        % RF variables
        theta = zeros(D,N); %RF coefficients
        psi = zeros(D,N); %Auxiliary RF coefficients (combine terms)

        % LMS variables
        psi_hat = zeros(M,N);
        h_dist = zeros(M,N);

        % error measures
        e_cc = zeros(N,T);
        mse_cc = zeros(N,T);

        e_rf = zeros(N,T);
        mse_rf = zeros(N,T);

        e_lms = zeros(N,T);
        mse_lms = zeros(N,T);

        % iterations
        for t=1:T

            %INPUT VECTOR AND OUTPUT FOR THAT ITERATION
            for m = 0:min(t-1,M-1)  % ICASSP Z
                Z(:,m+1,t) = (S^m)*X(:,t-m);
            end
%             Y(:,t) = sqrt(Z(:,1,t).^2 + sin(Z(:,4,t)*pi).^2) + (0.8 - 0.5*exp(-Z(:,2,t).^2)).*Z(:,3,t) + V(:,t);


            % LMS
            for k = 1:N %ADAPT
                %ADAPT CC PT1
                kvec{k} = exp(-(vecnorm(dict-Z(k,:,t),2,2).^2)/kernel_bw);
                e_cc(k,t) = Y(k,t) - alpha{k}'*kvec{k};
%                 mse_cc(k,t)= e_cc(k,t)^2/(Y(:,t)'*Y(:,t));;
                mse_cc(k,t)= e_cc(k,t)^2/(Y(k,t)^2);;

                %ADAPT RF
                zk = Z(k,:,t)'; %input vector at node k;
                rf_k = (D/2)^(-1/2)*cos(Vp'*zk + b); 

                e_rf(k,t) = Y(k,t) - theta(:,k)'*rf_k;  % current error   
                Yrff(k,t) = theta(:,k)'*rf_k;
                
%                 mse_rf(k,t)= e_rf(k,t)^2/(Y(:,t)'*Y(:,t));
                mse_rf(k,t)= e_rf(k,t)^2/(Y(k,t)^2);
                psi(:,k) = theta(:,k) + 2*e_rf(k,t)*rf_k;  %updating theta

                %ADAPT LMS
                psi_hat(:,k) = h_dist(:,k) + mu*zk*(Y(k,t) - zk'*h_dist(:,k));
                e_lms(k,t) = Y(k,t) - zk'*h_dist(:,k);
                mse_lms(k,t)= e_lms(k,t)^2;
            end

            for k = 1:N %ADAPT CC PT2
                aux_beta = alpha{k};
                N_k = find(C_nl(k,:));
                for l = N_k
                    e_kl = Y(l,t) - alpha{k}'*kvec{l};
                    aux_beta = aux_beta + mu*C_nl(k,l)*(e_kl)*kvec{l};
                end
                beta{k} = aux_beta;
            end

            % COMBINE %
            %COMBINE CC
            for k = 1:N
                alpha{k} = cell2mat( beta )*Cc(k,:)';
            end
            %COMBINE RF
            theta  = psi*Cc;

            %COMBINE LMS
            h_dist = psi_hat*Cc;
            
            % NMSE COMPUTATION
            for k=1:N
                N_k = find(C_nl(k,:));
                nmse_cc_k(k,t) = sum(e_cc(N_k,t).^2)/sum(Y(N_k,t).^2);
                nmse_rf_k(k,t) = sum(e_rf(N_k,t).^2)/sum(Y(N_k,t).^2);
            end
            

        end
        MSE_cc = (1/N)*sum(mse_cc);
        MSE_average_cc = MSE_average_cc + MSE_cc;

        MSE_rf = (1/N)*sum(mse_rf);
        MSE_average_rf = MSE_average_rf + MSE_rf;

        MSE_lms = (1/N)*sum(mse_lms);
        MSE_average_lms = MSE_average_lms + MSE_lms;

        toc    
    end

    MSE_final_average_rf = MSE_average_rf/number_of_runs;
    MSE_final_average_cc = MSE_average_cc/number_of_runs;
    MSE_final_average_lms = MSE_average_lms/number_of_runs;
    plot(10*log10(MSE_final_average_cc),'r');
    hold on;
    plot(10*log10(MSE_final_average_rf),'g');
%     plot(10*log10(MSE_final_average_lms),'r');
%     xlabel('Iteration','interpreter','latex','fontsize',14);
%     ylabel('MSE dB','interpreter','latex','fontsize',14);
%     xlabel('Iteration','FontName','Times New Roman','FontSize',14);
%     ylabel('MSE [dB]','FontName','Times New Roman','FontSize',14);
%     set(gca,'xtick',[1:10:length(time_stamps)],'xticklabel',time_stamps(1:10:length(time_stamps),:))
%     xtickangle(60);
    
    % fig_legend = legend();
    % fig_legend.FontSize = 14;
    % fig_legend.interpreter = 'latex';

%     save(['MSE_ALL_C0' num2str(cc_param*10) 'D' num2str(D) 'mu0' num2str(mu*10)],'MSE_final_average_cc','MSE_final_average_rf','MSE_final_average_lms','mu','cc_param','dict_target','D','number_of_runs','T')
end

% % %% PLOT
% 
% % NETWORK
% close all
% % pos = G.coords;
% % figure; gsp_plot_graph(G); hold on; for i=1:length(pos), text(pos(i,1)+0.2,pos(i,2)+0.5,int2str(i));end
% 
% k1 = 40;
% k2 = 50;
% % k3 = 20;
% % k4 = 50;
% xmax = 400;
% % ./Y(k1,1:1400).^2
% %SIGNALS ON SENSORS
% subplot(4,1,1);
% plot(  10*log10(e_cc(k1,1:1400).^2./Y(k1,1:1400).^2), 'r' );  title('e^2/y^2 (db)');
% hold on;
% plot(  10*log10(e_rf(k1,1:1400).^2./Y(k1,1:1400).^2), 'g'  );
% xlim([1 xmax]);
% subplot(4,1,2);
% plot(  (e_cc(k1,1:1400).^2./Y(k1,1:1400).^2), 'r' );  title('e^2/y^2 (linear)');
% hold on;
% plot(  (e_rf(k1,1:1400).^2./Y(k1,1:1400).^2), 'g'  );
% xlim([1 xmax]);
% subplot(4,1,3);
% plot(  10*log10(e_cc(k1,1:1400).^2), 'r' );  title('e^2 (db)');
% hold on;
% plot(  10*log10(e_rf(k1,1:1400).^2), 'g'  );
% xlim([1 xmax]);
% subplot(4,1,4);
% plot(  (e_cc(k1,1:1400).^2), 'r' );  title('e^2 (linear)');
% hold on;
% plot(  (e_rf(k1,1:1400).^2), 'g'  );
% xlim([1 xmax]);
% % plot([40,40],[1,-200],'--');
% % legend('CC-based GDKLMS','RFF-based GDKLMS','Location','southeast');
% % xlim([1 1400]); ylim([-150 2]);
% % set(gca,'xticklabel',[])
% % ylim([-1 2])
% xlabel('Iteration index (\it{n})')
% % ylabel('Normalized squared error (dB)')
% 
% % subplot(2,1,2);
% % plot(  10*log10(e_cc(k2,1:1400).^2./Y(k2,1:1400).^2), 'r' ); title('Sensor 50');
% % hold on;
% % plot(  10*log10(e_rf(k2,1:1400).^2./Y(k2,1:1400).^2), 'g'  );
% % % plot([40,40],[1,-200],'--');
% % % legend('CC-based GDKLMS','RFF-based GDKLMS','Location','southeast');
% % % xlim([1 1400]); ylim([-150 2]);
% % % set(gca,'xticklabel',[])
% % xlabel('Iteration index (\it{n})')
% % ylabel('Normalized error (dB)')
% 
% % subplot(2,2,3);
% % plot(  10*log10(e_cc(k3,1:1400).^2./Y(k3,1:1400).^2), 'r' ); title('Station 20');
% % hold on;
% % plot(  10*log10(e_rf(k3,1:1400).^2./Y(k3,1:1400).^2), 'g');
% % % plot([40,40],[1,-200],'--');
% % % legend('CC-based GDKLMS','RFF-based GDKLMS','Location','southeast'); 
% % % xlim([1 1400]); ylim([-150 2]); 
% % % set(gca,'xticklabel',[])
% % xlabel('Iteration index (\it{n})')
% % % ylabel('Normalized error (dB)')
% % 
% % subplot(2,2,4);
% % plot(  10*log10(e_cc(k4,1:1400).^2./Y(k4,1:1400).^2), 'r' ); title('Station 50');
% % hold on;
% % plot(  10*log10(e_rf(k4,1:1400).^2./Y(k4,1:1400).^2), 'g');
% % % plot([40,40],[1,-200],'--');
% % % legend('CC-based GDKLMS','RFF-based GDKLMS','Location','southeast');
% % % xlim([1 1400]); ylim([-150 2]);
% % % xlabel('Iteration index (\it{n})')
% % % ylabel('Normalized error (dB)')