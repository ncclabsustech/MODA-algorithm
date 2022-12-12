function [Us, ela] = EMan_trust(Xs, classes, lowerdims, tensor_shape)
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
            store.g = g;
            g = problem.M.egrad2rgrad(U, g);
           
        end
        
        problem.hess = @hess;
        function [h, store] = hess(U, eta, store)
            eeta = cell(1,nmodes);
            eeta{1} = eta.U1;
            eeta{2} = eta.U2;
            hess = cell(1,nmodes);
            for kmode = 1:nmodes
                QtRb_mm = TensorChainProductNT(Rb,Us,kmode);
                QtRb = matricizing(QtRb_mm.data, kmode);
                B = diag(diag(QtRb*QtRb'));
                QtRw_mm = TensorChainProductNT(Rw,Us,kmode);
                QtRw = matricizing(QtRw_mm.data, kmode);
                W = diag(diag(QtRw*QtRw'));
                UtWU = Us{kmode}' * W * Us{kmode};
        % ================================================================= 
                h1 = 2*B*eeta{kmode}*(UtWU)^-1;
                h2 = -4*B*Us{kmode}*(UtWU)^-1*multisym(eeta{kmode}'*W*Us{kmode})*(UtWU)^-1;
                h3 = (-4*W*multisym(eeta{kmode}*Us{kmode}')*B*Us{kmode}-2*W*Us{kmode}*Us{kmode}'*B*eeta{kmode})*(UtWU)^-2;
                h4 = 8*W*Us{kmode}*Us{kmode}'*B*Us{kmode}*(UtWU)^-2*multisym(eeta{kmode}'*W*Us{kmode})*(UtWU)^-2;
                hess{imode} = h1 + h2 + h3 + h4;          
           end
            h.U1 = -hess{1};
            h.U2 = -hess{2};
            h = problem.M.ehess2rhess(U, store.g, h, eta) ;
            %h.U2 = tuple.U2.ehess2rhess(U.U2, store.g.U2, h.U2, eta.U2) ;
        end
    
        % options
        options.maxiter = 200;
        options.maxinner = 30;
        options.maxtime = inf;
        options.tolgradnorm = 1e-5;
        options.Delta_bar = problem.M.typicaldist();
        % Minimize the cost function using Riemannian trust-regions
        U = trustregions(problem, [], options);
        
        Us{1} = U.U1;%U(1:N, 1:K1);
        Us{2} = U.U2;%U((N+1):end, (K1+1):end);

% =========================================================================
    ela=cputime-ela;
end





