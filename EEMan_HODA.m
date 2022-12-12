function [Us, outputs, ela] = EEMan_HODA(Xs, classes, lowerdims, tensor_shape)
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
tuple.U1 = stiefelfactory(size(Us{1}, 1), size(Us{1}, 2));
tuple.U2 = stiefelfactory(size(Us{2}, 1), size(Us{2}, 2));
manifold = productmanifold(tuple);
problem.M = manifold;
Rb = Rb.data;
Rw = Rw.data;
function store = prepare(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses)
    U1 = U.U1;
    U2 = U.U2;
    N = tensor_shape(1);
    M = tensor_shape(2);
    K1 = lowerdims(1);
    K2 = lowerdims(2);
        QtRb_mm = tmult(tmult(Rb,U1',1),U2',2);
        QtRb = reshape(permute(QtRb_mm,[2 1 3]),[prod(lowerdims),nclasses]);
        QtRw_mm = tmult(tmult(Rw,U1',1),U2',2);
        QtRw = reshape(permute(QtRw_mm,[2 1 3]),[prod(lowerdims),nsamples]);
        QtBQ = diag(diag(QtRb*QtRb'));
        store.QtBQ = QtBQ;
        QtWQ = diag(diag(QtRw*QtRw'));
        store.QtWQ = QtWQ;
        G = (-2*reshape(permute(Rw,[2 1 3]), M*N, nsamples)*QtRw'*(QtBQ/QtWQ) + 2*reshape(permute(Rb,[2 1 3]), M*N, nclasses)*QtRb')/QtWQ;
        TTT=reshape(permute(reshape(G, M, N, K2, K1), [2 4 1 3]),[N*K1 M*K2]);
        G1 =reshape(TTT*U2(:),[N K1]);
        G2=reshape(TTT'*U1(:),[M K2]);
        store.G1 = G1;
        store.G2 = G2;
end
% Define the problem cost function and its Euclidean gradient.
        problem.cost  = @(U, store) mycost(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses);
        function [f, store] = mycost(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses)
            
        store = prepare(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses);
        QtWQinvQtBQ = store.QtBQ/store.QtWQ;
        f = -trace(QtWQinvQtBQ);
        end
     
    % =====================================================================
        problem.grad = @(U, store) mygrad(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses);
        function [g, store] = mygrad(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses)
            store = prepare(U, Rb, Rw, store, tensor_shape, lowerdims, nsamples, nclasses);
            g.U1 = -store.G1;
            g.U2 = -store.G2;
        end
        
        % Solve.
        maxits = 1000;
        options.intialtau = -1;
        options.mxitr = maxits;
        options.record = 1;
        options.maxiter = maxits;
        [U, ~, outs] = conjugategradient(problem, U, options);
        fvals = cell2mat({outs.cost});
        outputs.fvals = fvals;
        outputs.outs = outs;
        Us{1} = U.U1;%U(1:N, 1:K1);
        Us{2} = U.U2;%U((N+1):end, (K1+1):end);

% =========================================================================
    ela=cputime-ela;
end
% =========================================================================







