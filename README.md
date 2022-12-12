# MODA-algorithm

Discriminative subspace learning is an important problem in machine learning, which aims to find the
maximum separable decision subspace. Traditional Euclidean-based methods usually use Fisher discriminant
criterion for finding an optimal linear mapping from a high-dimensional data space to a lower-dimensional
subspace, which hardly guarantee a quadratic rate of global convergence and suffers from the singularity
problem. Here, we propose the manifold optimization-based discriminant analysis (MODA) which is
constructed by using the latent subspace alignment and the geometry of objective function with orthogonality
constraint. MODA is solved by using Riemannian version of trust-region algorithm. Experimental results
on various image datasets and electroencephalogram (EEG) datasets show that MODA achieves the best
separability and is significantly superior to the competing algorithms. Especially for the time series of EEG
signals, the accuracy of MODA is 20% ∼ 30% higher than existing algorithms.
