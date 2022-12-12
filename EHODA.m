function [Us, ela] = EHODA(Xs, classes, lowerdims, tensor_shape)
% =========================================================================
X_N = size(Xs, 2);
X = reshape(Xs, [tensor_shape X_N]);
Xs = mat_to_cell(X);
Xss = tensor(X);


Xsample1 = Xs{1};
sizeX = size(Xsample1);
nmodes = length(sizeX);
nsamples = length(Xs);
nclasses = length(unique(classes));

Us = cell(1, nmodes);
for kmode = 1:nmodes
    Us{kmode} = orth(randn(sizeX(kmode), lowerdims(kmode)));
end

% calculate Xc - X for each class, where Xc is the class mean and X is the
% overall mean (stored in classmeandiffs) and Xcj - Xc where Xcj is the
% j'th observation from class c (stored in observationdiffs) and the number
% of observations from each class (stored in nis).
[~, ~, cmean_m_xmeans, xi_m_cmeans, nis] = classbased_differences(Xss, classes);
classmeandiffstensor = reshape(cmean_m_xmeans, [tensor_shape nclasses]);
observationdiffstensor = reshape(xi_m_cmeans, [tensor_shape nsamples]);
Rw = observationdiffstensor;
% ======================================== Rb =============================
nis = sqrt(nis);
tensor_nis = tenzeros([tensor_shape nclasses]);
cell_nis = cell(1,nclasses);
for n = 1:nclasses
    cell_nis{n} = nis(n)*ones(tensor_shape);
          switch length(tensor_shape)
            case 2
            tensor_nis(:,:,n) = cell_nis{n};
            case 3
            tensor_nis(:,:,:,n) = cell_nis{n};
            case 4
            tensor_nis(:,:,:,:,n) = cell_nis{n};
            case 5
            tensor_nis(:,:,:,:,:,n) = cell_nis{n};
            otherwise
            disp('tensor_shape is not between 2 and 5.');
         end
end
Rb = classmeandiffstensor.*tensor_nis;
maxits = 1000;


its = 0;
ela = cputime;
while true && its < maxits
    its = its + 1;
    oldUs = Us;
    difference = 0;
    for kmode = 1:nmodes
        QtRb_mm=TensorChainProductNT(Rb,Us,kmode);
        QtRb = matricizing(QtRb_mm.data, kmode);
        B = QtRb*QtRb';
        
        QtRw_mm=TensorChainProductNT(Rw,Us,kmode);
        QtRw = matricizing(QtRw_mm.data, kmode);
        W = QtRw*QtRw';
        
        phi = trace(Us{kmode}' * B * Us{kmode})/trace(Us{kmode}' * W * Us{kmode});
        
        [U, ~] = eigs(B-phi*W, lowerdims(kmode));
        UUt = U*U';
        tempt = TensorChainProductNT(Xss,Us,kmode);
        Xs_minus_n = matricizing(tempt.data, kmode);
        [Us{kmode}, ~] = eigs(UUt * (Xs_minus_n*Xs_minus_n') * UUt, lowerdims(kmode));
        
        difference = norm(Us{kmode}-oldUs{kmode}, 'fro')/(lowerdims(kmode)*sum(sizeX));
    end
    
    if difference<1e-6
        break
    end
    ela=cputime - ela;
end



