% =========================================================================
function [Us, WW, ela] = GLTD(X, nsamples, Graph, options)
ela = cputime;
J = [8 8 nsamples];
K = Graph.K;
epsilon = Graph.epsilon;
maxIter = options.maxIter;
lambda = options.lambda;


% =========================================================================
sizeX = size(X);
nmodes = length(sizeX);
Us = cell(1, nmodes);
% computing the L matrix
% =========================================================================
    XN = tenmat(X, length(sizeX));
    XN = XN.data';
    if lambda > 0
        W = constuct_W( XN, K, epsilon );
        DCol = full(sum(W,2));
        D = spdiags(DCol, 0, sizeX(end), sizeX(end));
        L = D - W;
    else
        D = [];
        W = [];
        L = [];
     end
% =========================================================================
options.tol = 1e-9; options.issym=1; options.disp = 0; 
% =========================================================================
% initialization to Un 
for kmode = 1:nmodes
    Us{kmode} = orth(rand(sizeX(kmode), J(kmode)));
end
% ==================== for updating the projection matrices % =============
X = tensor(X);
% =========================================================================
% for updating the projection matrices
tryNo = 0;
while tryNo < maxIter   
    tryNo = tryNo+1;
    nIter = 0;
    for n = 1:nmodes-1
        Xn = TensorChainProductXTT(X,Us,n);
        Xn = tenmat(Xn, n);
        Xn = Xn.data;
        [Us{n}, ~] = eigs(Xn*Xn', J(n));   
    end
% =========================================================================
% for updating the last projection matrix
        XN = TensorChainProductXTT(X,Us,nmodes);
        XN = tenmat(XN, nmodes);
        XN = XN.data;
        XNL = XN*XN';
        %XNL = XN*XN' - lambda*L;
        [Us{nmodes}, ~] = eigs(XNL, J(nmodes));
        WW = Us{nmodes};

% =========================================================================
    nIter = nIter + 1;
    disp(['GLTD algorithm in the iteration error: ',num2str(nIter)]);
end
ela=cputime - ela;
end



