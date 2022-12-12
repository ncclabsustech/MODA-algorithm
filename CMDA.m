function [Us, ela] = CMDA(Xs, classes, lowerdims, tensor_shape)
ela = cputime;
%% read input and set parameters
X_N = size(Xs, 2);
X = reshape(Xs, [tensor_shape X_N]);
Xs = mat_to_cell(X);
Xss = tensor(X);
% =========================================================================
Xsample1 = Xs{1};
sizeX = size(Xsample1);
nmodes = length(sizeX);
nsamples = length(Xs);
nclasses = length(unique(classes));
Us = cell(1, nmodes);
for kmode = 1:nmodes
    Us{kmode} = orth(randn(sizeX(kmode), lowerdims(kmode)));
end
tol=1e-6;
Tmax = 100;
usestoppingcrit = true;
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

stop = false;
iit = 0;
innerits = 0;
while ~stop && iit < Tmax
    iit = iit+1;
    oldUs = Us;
    for kmode = 1:nmodes
        innerits = innerits +1;  
        
        QtRb_mm=TensorChainProductNT(Rb,Us,kmode);
        QtRb = matricizing(QtRb_mm.data, kmode);
        B = QtRb*QtRb';
        
        QtRw_mm=TensorChainProductNT(Rw,Us,kmode);
        QtRw = matricizing(QtRw_mm.data, kmode);
        W = QtRw*QtRw';
        
        [U, ~] = svd(W\B, 0);
        Us{kmode} = U(:, 1:lowerdims(kmode));
        
    end
    
    % this is the stopping criterion proposed with CMDA in li14
    if usestoppingcrit && iit > 2
        errs(iit) = 0;
        for kmode = 1:nmodes
            errs(iit) = errs(iit) + norm(Us{kmode}*oldUs{kmode}' - eye(size(Us{kmode},1)), 'fro');
        end
        if errs(iit) <=tol
            stop = true;
        end
    end
end
ela=cputime - ela;
end


