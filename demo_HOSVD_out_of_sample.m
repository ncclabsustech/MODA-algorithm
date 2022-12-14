
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'new_dense_ETH80'};

%=============================================
cross_num = 3; % k-CV
for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', name);
    fprintf('Loading   ...');
    load( name );% loading dataset
    fprintf('Done\n');
    nClasses = length(unique(gnd));
    X = fea;
    Y = gnd;
    % =====================================================================
    type_method = 'HOSVD'; % as type = 'NTD_LE' & options.alpha = 0, that is NTD algorithm
    tensor_shape    = [32 32];% the dimension for dataset dependent reshaped tensors
    lowerdims = [8 8];
    % =====================================================================
    runtimes = 10; % runing 10 times
    % =====================================================================
    for rn = 1 : runtimes
        indices = kFoldCV( Y, cross_num,  nClasses);
        % generates cross-validation indices
        t1 = clock;% count time
        for cros = 1 : cross_num
            fprintf('Starting cross validation: %dth\n', cros );
            %=============== test dataset and train dataset =====================%
            test_indx = (indices == cros);
            train_indx = ~test_indx;
            
            train = X( train_indx, :);                         %===================%
            train_label = Y( train_indx );                     %      ѡ??ѵ????    %
            %                                                  %===================%
            
            test = X( test_indx, :);                       %===================%
            test_label = Y(test_indx);                         %     ѡ?????Լ?     %
            test_nsamples = size(test, 1);
            train_nsamples = size(train, 1);
            % =============================================================
            train = train';
            test = test';
            R_nClasses = length(unique(train_label));
            T_nClasses = length(unique(test_label));
            % =================================================================
            dim_c = [lowerdims R_nClasses];
            Xtrain = reshape(train, [tensor_shape train_nsamples]);
            [Us, V, ela]    =     hosvd(Xtrain, dim_c, 'standard');
            %[Us, ela] = EHODA(train, train_label, lowerdims, tensor_shape);
            %trainCore = CoreTensor(train, Us, tensor_shape);
            %V = reshape(trainCore, [prod(lowerdims) train_nsamples]);
            
            % ============================== train  =======================
            R_label = litekmeans(V, R_nClasses,'Replicates', 20);
            R_label = bestMap(train_label, R_label);
            results.train_AC(cros, rn) = length(find(train_label == R_label))/length(train_label);
            results.train_NMI(cros, rn) = MutualInfo(train_label,R_label);
            iinf = cros + (rn - 1)*cross_num;
            results.rse{iinf} = ela;
            clear V R_label;
            % ============================== test  ========================
            testCore = CoreTensor(test, Us, tensor_shape);
            V = reshape(testCore, [prod(lowerdims) test_nsamples]);
            % =============================================================
            cros_num = 5;
            indicess = kFoldCV( test_label, cros_num, T_nClasses );
            % generates cross-validation indices
            XX = V.data';
            for cro = 1 : cros_num
                fprintf('Starting cross validation: %dth\n', cro );
                %=============== test dataset and train dataset =====================%
                testt_indx = (indicess == cro);
                trainn_indx = ~testt_indx;
            
                trainn = XX( trainn_indx, :);                         %===================%
                trainn_label = test_label( trainn_indx );                     %      ѡ??ѵ????    %
                %                                                  %===================%
                testt = XX( testt_indx, :);                       %===================%
                testt_label =test_label(testt_indx);                         %     ѡ?????Լ?     %
                %results.accur(cros, rn) = knnclassifier(train, test, 5);
                results.test(cro, rn) = knnClassification(T_nClasses,trainn,trainn_label,testt,testt_label);
                iinf = cro + (rn - 1)*cros_num;
                results.rse{iinf} = ela;
            end
        end
          
    end
    results.method = type_method;
    save(['Results\', 'res_', type_method,'', name ],  'results');
    clear results;
end

