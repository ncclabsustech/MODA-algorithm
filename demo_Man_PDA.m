%==========================================================================
% Notation:
% Input X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
%       gnd ...(nSmp x 1) data label
%load COIL20.mat;
%train_data = fea(1:800,:)';
%train_classes = gnd(1:800,:) ;
% ======================= test data =======================================
%test_data = fea(800:1440,:)';
%test_classes = gnd(800:1440,:);
% =========================================================================
%load ORL.mat;
%train_data = X'/255;
%train_classes = Y;
% =========================================================================
%load MNIST.mat
%imgs = reshape(imgs, [400 9700]);
%train_data = imgs(:,1000:4000);
%train_classes = labels(1000:4000,1);
%train_classes = train_classes + 1;
% ======================= test data =======================================
%test_data = imgs(:,6000:9000);
%test_classes = labels(6000:9000,1);
%test_classes = test_classes + 1;
% =========================================================================
load new_UMIST.mat;
imgs = fea;
train_data = imgs(:,1:250);
labels = gnd';
train_classes = labels(1:250,1);
% ======================= test data =======================================
test_data = imgs(:,250:end);
test_classes = labels(250:end,1);
% =========================================================================
%load new_PIEP3I3_32x32.mat;
%imgs = fea';
%labels = gnd;
%train_data = imgs(:,2000:3000);
%train_classes = labels(2000:3000,1);
% ======================= test data =======================================
%test_data = imgs(:,3500:4000);
%test_classes = labels(3500:4000,1);
% =========================================================================


train_nsamples = size(train_data, 2);
train_nClasses = length(unique(train_classes));
test_nsamples = size(test_data, 2);
test_nClasses = length(unique(test_classes));
tensor_shape    = [32 32];% the dimension for dataset dependent reshaped tensors
lowerdims = [10 10];
% =================================== train project matrix ================
% algorithm EHODA
%[Us, ~, ela] = EMan_conj(train_data, train_classes, lowerdims, tensor_shape);
[Us, outputs, ela] = Man_PDA(train_data, train_classes, lowerdims, tensor_shape);
fprintf('Man_HODA completed in %f seconds.\n',ela);
% =================================== train Core ==========================
trainCore_ManHODA = CoreTensor(train_data, Us, tensor_shape);
% ==================================== test Core ==========================
testCore_ManHODA = CoreTensor(test_data, Us, tensor_shape);
% =================================== train MIhat =========================
V = reshape(trainCore_ManHODA, [prod(lowerdims) train_nsamples]);
label = litekmeans(V.data', train_nClasses,'Replicates', 20);
trainManHODA_MIhat = MutualInfo(train_classes,label);
clear V label;
% ================================== test MIhat ===========================
V = reshape(testCore_ManHODA, [prod(lowerdims) test_nsamples]);
label = litekmeans(V.data', test_nClasses,'Replicates', 20);
testManHODA_MIhat = MutualInfo(test_classes,label);
clear V label;
% =========================== Print result ================================
disp(['Clustering in the tensor subspace for type algorithm. trainManHODA_MIhat: ',num2str(trainManHODA_MIhat)]);
disp(['Clustering in the tensor subspace for type algorithm. testManHODA_MIhat: ',num2str(testManHODA_MIhat)]);




