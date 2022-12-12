function [Us, outputs, ela] = Man_HODA(Xs, classes, lowerdims, tensor_shape)
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
U.U1 = Us{1};
U.U2 = Us{2};
[~, K1] = size(Us{1});
[~, K2] = size(Us{2});
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
options.intialtau = -1;
options.mxitr = maxits;
options.record = 1;
options.maxiter = maxits;
ela = cputime;
% Stiefel manifold
tuple.U1 = stiefelfactory(size(Us{1}, 1), size(Us{1}, 2));
tuple.U2 = stiefelfactory(size(Us{2}, 1), size(Us{2}, 2));
manifold = productmanifold(tuple);
problem.M = manifold;
Rw = Rw.data;
Rb = Rb.data;
% Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(U, store) mycost(U, store,...
            cmean_m_xmeans, xi_m_cmeans, nis,...
            K1, K2, Rw, Rb);
        
        problem.egrad = @(U, store) mygrad(U, store,...
            cmean_m_xmeans, xi_m_cmeans, nis,...
            K1, K2, Rw, Rb);
        
        % Solve.
        [U, ~, outs] = conjugategradient(problem, U, options);
        fvals = cell2mat({outs.cost});
        outputs.fvals = fvals;
        outputs.outs = outs;
        Us{1} = U.U1;%U(1:N, 1:K1);
        Us{2} = U.U2;%U((N+1):end, (K1+1):end);

% =========================================================================
    ela=cputime-ela;
end

%==========================================================================
function [F, store] = mycost(x, store, classmeandiffs, observationdiffs,...
    nis, K1, K2, Rw, Rb)
[F, ~, ~, ~, store]...
    = tensorsldaobj_matrixdata(x,...
    classmeandiffs, observationdiffs, nis, K1, K2, ...
    Rw, Rb, store);
end


function [G, store] = mygrad(x, store, classmeandiffs, observationdiffs,...
    nis, K1, K2, Rw, Rb)
[~, G, ~, ~, store]...
    = tensorsldaobj_matrixdata(x,...
    classmeandiffs, observationdiffs, nis, K1, K2, ...
    Rw, Rb, store);
end


