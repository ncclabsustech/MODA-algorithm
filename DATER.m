function [Us, ela] = DATER(Xs, classes, lowerdims, tensor_shape)
% [Us, iit, errs, objfuncvals, Ys] = DATER(Xs, classes, varargin)
% Mandatory input:
% Xs:           Cell array containing the observed tensors.
% classes:      Vector containing class labels. Classes must be sequential
%               numbers starting from one.
%
% Optional input:
% varargin{1}:  Scalar giving the maximal number of iterations.
%               Default: 100.
% varargin{2}:  Vector containing the number of components to estimate for
%               each mode. Default: size of observations.
% varargin{3}:  Boolean indicating whether or not convergence criteria
%               should be allowed to terminate iterations.
%               Default: true.
% varargin{4}:  Cell array of initial projection matrices or the string
%               'randinit' indicating initialisation with random
%               orthogonal matrices. Default: identity matrices as
%               proposed in yan05.
%
% Output:
% Us:           Cell array containing the projection matrices found.
% iit:          The number of outer iterations performed (an outer 
%               iteration consists of one optimisation of each mode).
% errs:         The values from the stopping criterion proposed in yan05, 
%               one for each outer iteration.
% objfuncvals:  Values of the objective function that DATER tries to
%               optimise. Values are given for all inner iterations. An
%               inner iteration consists of the optimisation of one mode.
% objfuncvals_traceratio:   Values of the objective function
%                           tr((U'WU)^(-1)U'BU), which is optimised by the
%                           generalised eigenvalue problem, which is the
%                           heuristic DATER uses during optimisation. W is
%                           the within-class scatter matrix while B is the
%                           between-class scatter matrix.
% Ys:           Projections of the original data in Xs projected onto the
%               final projection matrices in Us.
%
% yan05:
%   S. Yan, D. Xu, Q. Yang, L. Zhang, X. Tang, and H. Zhang
%   'Discriminant Analysis with Tensor Representation'
%   IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2005


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
% multiply all entries in classmeandiffstensor by the square root of the
% size of their class. When Rb is multiplied by its own transpose, the
% class sizes are automatically accounted for in the resulting sum.

innerits = 0;
for iit = 1:Tmax
    
    oldUs = Us;
    for kmode = 1:nmodes
        innerits = innerits +1;
        
        QtRb_mm=TensorChainProductNT(Rb,Us,kmode);
        QtRb = matricizing(QtRb_mm.data, kmode);
        B = QtRb*QtRb';
        
        QtRw_mm=TensorChainProductNT(Rw,Us,kmode);
        QtRw = matricizing(QtRw_mm.data, kmode);
        W = QtRw*QtRw';
        
        
        [U, eigvals] = eig(B, W);
        
        eigvals = diag(eigvals);
        [~, sortedinds] = sort(eigvals, 'descend');
        
        Us{kmode} = U(:, sortedinds(1:lowerdims(kmode)));
    end
    
    % stopping criterion proposed with DATER in yan05
    if usestoppingcrit && (iit > 2)
        stopnow = true;
        for kmode = 1:nmodes
            errs(iit) = norm(Us{kmode} - oldUs{kmode});
            stopnow = stopnow && (norm(Us{kmode} -...
                oldUs{kmode})<sizeX(kmode)*lowerdims(kmode)*tol);
        end
        if stopnow
            break
        end
    end
end
ela=cputime - ela;
end


