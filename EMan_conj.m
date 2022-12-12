function [Us, outputs, ela] = EMan_conj(Xs, classes, lowerdims, tensor_shape)
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
for imode = 1:nmodes
    Us{imode} = orth(randn(sizeX(imode), lowerdims(imode)));
end

% calculate Xc - X for each class, where Xc is the class mean and X is the
% overall mean (stored in classmeandiffs) and Xcj - Xc where Xcj is the
% j'th observation from class c (stored in observationdiffs) and the number
% of observations from each class (stored in nis).
U.U1 = Us{1};
U.U2 = Us{2};
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
ela = cputime;
% Stiefel manifold ========================================================
% Stiefel manifold
tuple.U1 = stiefelfactory(size(Us{1}, 1), size(Us{1}, 2));
tuple.U2 = stiefelfactory(size(Us{2}, 1), size(Us{2}, 2));
manifold = productmanifold(tuple);
problem.M = manifold;

function store = prepare(U, Rb, Rw, nmodes, store)
    Gs = cell(1,nmodes);
    Us = cell(1,nmodes);
    Us{1} = U.U1;
    Us{2} = U.U2;
    for kmode = 1:nmodes
        QtRb_mm=TensorChainProductNT(Rb,Us,kmode);
        QtRb = matricizing(QtRb_mm.data, kmode);
        B = diag(diag(QtRb*QtRb'));
        UtBU = Us{kmode}' * B * Us{kmode};
        store.UtBU = UtBU;
        QtRw_mm=TensorChainProductNT(Rw,Us,kmode);
        QtRw = matricizing(QtRw_mm.data, kmode);
        W = diag(diag(QtRw*QtRw'));
        UtWU = Us{kmode}' * W * Us{kmode};
        store.UtWU = UtWU;
        UtWUinvUtBU = (UtBU)/(UtWU);
        % ================================================================= 
        Gs{kmode} = (2*B*Us{kmode}-2*W*Us{kmode}*UtWUinvUtBU)/UtWU;
        store.Gs = Gs;
    end
end
% Define the problem cost function and its Euclidean gradient.
        problem.cost  = @cost;
        function [f, store] = cost(U, store)
            
            store = prepare(U, Rb, Rw, nmodes, store);
            f = -trace(store.UtBU/store.UtWU);
        end
     
        problem.grad = @grad;
        function [g, store] = grad(U, store)
            store = prepare(U, Rb, Rw, nmodes, store);
            g.U1 = -store.Gs{1};
            g.U2 = -store.Gs{2};
        end
        
        
    
        % options
        % options
        maxits = 1000;
        options.intialtau = -1;
        options.mxitr = maxits;
        options.record = 1;
        options.maxiter = maxits;
        % Minimize the cost function using Riemannian trust-regions
        [U, ~, outs] = conjugategradient(problem, U, options);
        fvals = cell2mat({outs.cost});
        outputs.fvals = fvals;
        outputs.outs = outs;
        
        Us{1} = U.U1;%U(1:N, 1:K1);
        Us{2} = U.U2;%U((N+1):end, (K1+1):end);

% =========================================================================
    ela=cputime-ela;
end





