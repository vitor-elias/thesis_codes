%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_v1 : Script utilizado para gerar resultados do SBrT2017.
% Faz compress�o para L linhas buscando melhor Q.
% main_v2 : Usar modelo Nearest-neighbor para v�rias imagens na LINHA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
block_size = [32 32];
Nlinhas = 9; Ncolunas = 9;
ref = [4 7];

% Pasta com variaveis gigantes salvas
vardir = 'C:\Users\Avell\Documents\MATLAB\';

% Pasta com imagens
% imdir = 'C:\Users\Avell\Desktop\LF_DBs\DATABASE_sideboard\'; %Notebook lab
imdir = 'C:\Users\Administrador\Desktop\LF_DBs\DATABASE_sideboard\';
% addpath(imdir);
d = dir([imdir '*.png']); %Seleciona imagens da pasta
n_imagens = length(d); %Conta quantas imagens foram selecionadas
I_teste = im2double(rgb2ycbcr(imread([imdir d(1).name])));
Y_teste = I_teste(:,:,1);
tamanho = floor(size(Y_teste)/32)*32;
Nblocos = tamanho(1)*tamanho(2)/block_size(1)/block_size(2);

fun_redim = @(x) x(1:tamanho(1),1:tamanho(2));
fun_quant = @(X,K) removeKlast(X,K);

% % % show lightfield % % %
% warning('off','all');
% for i = 1:n_imagens, imshow([imdir d(i).name]); end
% warning('on','all');
% % % 


fprintf('Block_size %i x %i\n', block_size(1), block_size(2));
r = 0;

for linha = 1:Nlinhas
    tic
    L = linha-1;

    fprintf('Linha: %i\n', linha);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CELL DE TODOS RESIDUOS
    k = 0;
    for ind_atual = (ref(1) + Ncolunas*L):(ref(2) + Ncolunas*L)
        k = k+1;
        %Obtendo luminancia da imagem atual
        I_atual = fun_redim(im2double(rgb2ycbcr(imread([imdir d(ind_atual).name]))));
        Y_atual = I_atual(:,:,1);

        %Obtendo luminancia da imagem anterior
        I_ant= fun_redim(im2double(rgb2ycbcr(imread([imdir d(ind_atual-1).name]))));
        Y_ant = I_ant(:,:,1);

        %Agrupando K residuos em array 3D
        r_atual_K(:,:,k) = Y_atual - Y_ant;
    end
    
    ref_cell_K = mat2cell(r_atual_K,block_size(1)*ones(1,size(r_atual_K,1)/block_size(1)),block_size(2)*ones(1,size(r_atual_K,2)/block_size(2)),size(r_atual_K,3));
    
 
%     %Transformando referencia em celulas com DIM = block_size
%     %Cada celula � um bloco
%     ref_cell = mat2cell(r_ref,block_size(1)*ones(1,size(r_ref,1)/block_size(1)),block_size(2)*ones(1,size(r_ref,2)/block_size(2)));
%     return

    fprintf('Calculando matrizes de Adjac�ncia A e transformada V.\n');
    %Calculando matriz de Adjacencias A
    %Calculada apenas para o residuo de referencia
    tic
    A_cell = cellfun(@(x) createA_mult(x),ref_cell_K,'UniformOutput',0);
    tocA = toc;
    fprintf('  Tempo gasto com o c�lculo das matrizes A: %f\n', tocA);
    
    
    tic
    V_cell = cellfun(@(x) getV(x),A_cell,'UniformOutput',0);
    tocV = toc;
    fprintf('  Tempo gasto com o c�lculo das matrizes V em cellfun: %f\n', tocV);    
    
    % load('AV_r89_bloco32x32')

    fprintf('Matrizes calculadas.\n');
%     return

    for ind_atual = (2+Ncolunas*L):(Ncolunas+Ncolunas*L)

        r = r+1;
        fprintf('res�duo atual: %i\n', r);

        %Obtendo luminancia da imagem atual
        I_atual = fun_redim(im2double(rgb2ycbcr(imread([imdir d(ind_atual).name]))));
        Y_atual = I_atual(:,:,1);

        %Obtendo luminancia da imagem anterior
        I_ant= fun_redim(im2double(rgb2ycbcr(imread([imdir d(ind_atual-1).name]))));
        Y_ant = I_ant(:,:,1);

        %Calculo do residuo
        r_atual = Y_atual - Y_ant;

        %Transformando em celulas com DIM = block_size
        %Cada celula � um bloco
        r_cell = mat2cell(r_atual,block_size(1)*ones(1,size(r_atual,1)/block_size(1)),block_size(2)*ones(1,size(r_atual,2)/block_size(2)));

        %Aplicando dct2 bloco a bloco
        dct_cell = cellfun(@(x) dct2(x),r_cell,'UniformOutput',0);

    %     break

        %Aplicando GFT
        %Matriz V de autovetores generalizados calculada junto com a GFT

    %     fprintf('Calculando GFT dos blocos.\n');
        gft_cell = cellfun(  @(X,A) gftV(X(:),A)  ,r_cell,  V_cell  ,'UniformOutput',0);


        %"Quantizacao" dos resultados da DCT e da GFT
        %Zerando os K menores coeficientes (em valor absoluto) de cada bloco
        K_dct = 924;
        K_dct_cell = cell(size(dct_cell));
        K_dct_cell(:) = {K_dct};

        dct_quant = cellfun(fun_quant, dct_cell, K_dct_cell, 'UniformOutput',0);
        r_dct_quant = cellfun(@(x) idct2(x), dct_quant, 'UniformOutput',0);
        mse_dct(r) = immse(r_atual,cell2mat(r_dct_quant));


        K_gft = K_dct;
        if r>1
            K_gft = Kgft(r-1);
        end


        K_gft_cell = cell(size(gft_cell));
        K_gft_cell(:) = {K_gft};

        gft_quant = cellfun(fun_quant, gft_cell, K_gft_cell, 'UniformOutput',0);


        r_gft_quant = cellfun(@(x,V) igft(x,V),  gft_quant, V_cell, 'UniformOutput', 0);
        r_gft_quant = cellfun(@(x) reshape(x,block_size),r_gft_quant, 'UniformOutput', 0);

        mse_gft(r) = immse(r_atual,cell2mat(r_gft_quant));


        if mse_gft(r)>mse_dct(r)

            while mse_gft(r)>mse_dct(r)

                K_gft = K_gft-1;

                K_gft_cell = cell(size(gft_cell));
                K_gft_cell(:) = {K_gft};

                gft_quant = cellfun(fun_quant, gft_cell, K_gft_cell, 'UniformOutput',0);

                r_gft_quant = cellfun(@(x,V) igft(x,V),  gft_quant, V_cell, 'UniformOutput', 0);
                r_gft_quant = cellfun(@(x) reshape(x,block_size),r_gft_quant, 'UniformOutput', 0);

                mse_gft(r) = immse(r_atual,cell2mat(r_gft_quant));
            end

        else

            while mse_gft(r)<mse_dct(r)

                K_gft = K_gft+1;

                K_gft_cell = cell(size(gft_cell));
                K_gft_cell(:) = {K_gft};

                gft_quant = cellfun(fun_quant, gft_cell, K_gft_cell, 'UniformOutput',0);

                r_gft_quant = cellfun(@(x,V) igft(x,V),  gft_quant, V_cell, 'UniformOutput', 0);
                r_gft_quant = cellfun(@(x) reshape(x,block_size),r_gft_quant, 'UniformOutput', 0);

                mse_gft(r) = immse(r_atual,cell2mat(r_gft_quant));

            end
            K_gft = K_gft-1;
            K_gft_cell = cell(size(gft_cell));
            K_gft_cell(:) = {K_gft};

            gft_quant = cellfun(fun_quant, gft_cell, K_gft_cell, 'UniformOutput',0);

            r_gft_quant = cellfun(@(x,V) igft(x,V),  gft_quant, V_cell, 'UniformOutput', 0);
            r_gft_quant = cellfun(@(x) reshape(x,block_size),r_gft_quant, 'UniformOutput', 0);

            mse_gft(r) = immse(r_atual,cell2mat(r_gft_quant));
        end

        Kdct(r) = K_dct;
        Kgft(r) = K_gft;



    %     fprintf('\n MSE DCT: %g \n MSE GFT: %g\n\n',mse_dct(residuo),mse_gft(residuo));

    %     
    %     figure; imshow(r_atual);
    %     figure; imshow(kron(mse_compare,ones(block_size))); title(['Tipo Vencedor: GFT Preto: ' num2str(vence_gft),' --- DCT Branco: ' num2str(vence_dct)]);
    %     figure; imshow(kron(Acond_cell,ones(block_size))); title('Matriz A mal condicionada = branco');

    %     figure; hist(r_atual(:),unique(r_atual));  title(['Imagem central: ' num2str(ind_centro) ' ---  Imagem atual: ' num2str(ind_atual)]);


    end
    toc
end

figure;
plot(mse_dct,'.-')
hold on;
plot(mse_gft,'.-')
axis tight
xlabel('�ndice do residuo');
ylabel('MSE')
legend('MSE DCT','MSE GFT');

figure;
plot(Kgft-Kdct,'k.-');
xlabel('�ndice do residuo');
ylabel('Qgft-Qdct')
title('Diferen�a entre quantidade de coeficientes (Qgft-Qdct)');

fprintf('Quantidade m�dia de coeficientes na GFT: %g\n', mean(Kgft));
% dct_resultados = reshape(mse_dct,[3 3])'
% gft_resultados = reshape(mse_gft,[3 3])'



figure;
plot(Kgft(1:15)-Kdct(1:15),'k.-');
xlabel('Residual image index');
ylabel('Q^{GFT}-Q^{DCT}');
xlim([1,15])

% coluna_cell
% cond_cell

%Save kdct kgft mse_dct mse_gft Nblocos ref