# Computing manifold SNR and capacity at AlexNet FC2

## Bottom line

For one layer, represent each category by a **concept manifold**: the 50 activation vectors elicited by 50 images of that category. With 50 categories, raw AlexNet FC2 therefore gives 50 matrices

\[
X_{\mathrm{raw}}^a\in\mathbb{R}^{4096\times 50},\qquad a=1,\ldots,50.
\]

The defining DNN extraction code then uses one fixed Gaussian random projection for layers wider than 2,048 units, so the matrices supplied to the metric code are \(X^a\in\mathbb R^{2048\times50}\). The CCN paper computes both metrics from the same 2,500 images, but with two different published algorithms:

- **SNR:** a PCA/covariance description of each manifold, followed by a directed calculation for every ordered category pair. It estimates how reliably a prototype learned from \(m\) examples separates a test example from another concept.
- **Capacity:** replica mean-field theory (MFT), using support/anchor points of each point-cloud manifold. It estimates the critical number of randomly labeled concept manifolds that a linear readout can separate per representation dimension.

The paper says **50 categories with 50 images per category**, not 50 images total. Categories may come from ImageNet-1K, ImageNet-21K, or Places365. For the requested ImageNet-1K analysis, sample 50 of the 1,000 labels and 50 images independently within every sampled label.

## 1. Constructing AlexNet FC2 manifolds

1. Load a standard ImageNet-1K-pretrained AlexNet in evaluation mode. Match the defining extraction code: resize the shorter side to 256 pixels, take a 224×224 center crop, convert to a tensor, and normalize RGB channels with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Do not add stochastic crops or augmentation.
2. Fix and record the 50 category IDs, the 50 image IDs per category, and the random seed. Use exactly this image panel for every model/layer being compared.
3. Extract the 4,096-dimensional **post-ReLU output of the second fully connected layer** (`classifier[5]` in torchvision; the linear FC2 output is `classifier[4]`). This post-ReLU vector is AlexNet's final hidden/feature representation, immediately before the 1,000-way classifier. If “FC2” is intended to mean the pre-ReLU linear output instead, say so explicitly; it is a different representation and will give different geometry.
4. Put the 50 column vectors for category \(a\) into \(X_{\mathrm{raw}}^a\in\mathbb R^{4096\times50}\). Do not softmax, normalize each image vector, or average images before constructing these matrices.
5. To match the defining DNN pipeline, draw one matrix \(A\in\mathbb R^{2048\times4096}\), with independent \(A_{ij}\sim\mathcal N(0,1/2048)\), fix it for all categories and images, and compute

   \[
   X^a=A X_{\mathrm{raw}}^a\in\mathbb R^{2048\times50}.
   \]

   Save the projection seed or the matrix itself. This projection is triggered because AlexNet FC2 has more than 2,048 units. Both metrics should receive these same projected matrices if the goal is to compare them on exactly the same representation.

The fact that AlexNet was trained on all 1,000 ImageNet classes is not part of either calculation. It only determines the learned representation. The analysis uses the selected 50 classes as 50 point clouds in FC2 space.

## 2. Manifold signal-to-noise ratio (SNR)

### Geometry estimated for each category

For category \(a\), let its centroid and centered data be

\[
x_0^a=\frac1P\sum_{\mu=1}^{P}x_\mu^a,
\qquad \widetilde X^a=X^a-x_0^a\mathbf1^\top,
\qquad P=50.
\]

Compute the thin SVD

\[
\widetilde X^a=Q^a\,\operatorname{diag}(r_1^a,\ldots,r_P^a)(V^a)^\top.
\]

With the \(N\times P\) orientation used here, the columns of \(Q^a\) are the feature-space principal axes; denote them by \(u_i^a\). (The reference notebook stores each manifold transposed, so its right-singular-vector variable contains these same axes.) Define

\[
R_a^2=\frac1P\sum_i(r_i^a)^2,
\qquad
D_a=\frac{\left[\sum_i(r_i^a)^2\right]^2}{\sum_i(r_i^a)^4}.
\]

\(R_a\) is the RMS within-category radius and \(D_a\) is its participation-ratio dimension. Because centering removes one degree of freedom, at most 49 singular values should be materially nonzero here.

### Directed pairwise SNR

For each ordered pair \(a\ne b\), define

\[
d_{ab}=\lVert x_0^a-x_0^b\rVert,
\qquad
s_{ab}=\frac{d_{ab}}{R_a},
\qquad
\hat\delta_{ab}=\frac{x_0^a-x_0^b}{d_{ab}}.
\]

The source implementation computes the following overlaps:

\[
c_{a|ab}=\frac{\sum_i(r_i^a)^2(u_i^a\!\cdot\hat\delta_{ab})^2}
{\sum_i(r_i^a)^2},
\]

\[
c_{b|ab}=\frac{\sum_j(r_j^b)^2(u_j^b\!\cdot\hat\delta_{ab})^2}
{\sum_i(r_i^a)^2},
\]

\[
q_{ab}=\frac{\sum_{ij}(r_i^a)^2(r_j^b)^2(u_i^a\!\cdot u_j^b)^2}
{\left[\sum_i(r_i^a)^2\right]^2},
\qquad
B_{ab}=\frac{\sum_j(r_j^b)^2}{\sum_i(r_i^a)^2}-1.
\]

Then the exact dominant-term expression implemented by the cited repository is

\[
\boxed{
\operatorname{SNR}_{a\leftarrow b}(m)=
\frac{1}{2}
\frac{s_{ab}^{\,2}+B_{ab}/m}
{\sqrt{D_a^{-1}/m+s_{ab}^{\,2}\left(c_{a|ab}+c_{b|ab}/m\right)+q_{ab}/m}}
}
\]

where \(m\) is the number of training examples per concept in the hypothetical \(m\)-shot prototype classifier. This is **directed**: generally \(\operatorname{SNR}_{a\leftarrow b}\ne\operatorname{SNR}_{b\leftarrow a}\), because the test concept \(a\) supplies the normalization, dimension, and persistent test-example noise. The corresponding predicted error on examples of \(a\) is

\[
\epsilon_{a\leftarrow b}=H(\operatorname{SNR}_{a\leftarrow b}),
\quad
H(z)=\frac12\operatorname{erfc}\!\left(\frac{z}{\sqrt2}\right).
\]

**One layer-level scalar:** compute the \(50\times49=2{,}450\) off-diagonal directed values and take their arithmetic mean (NumPy nanmean of the matrix, whose diagonal is NaN). This is the aggregation used in the defining notebooks. Save the full matrix as well. Larger SNR means better separation relative to within-concept variation and therefore lower predicted few-shot error.

### Intuition

The numerator is useful centroid separation, adjusted for unequal manifold sizes at finite \(m\). The denominator combines four noise sources: finite-sample centroid error, variation of the test manifold along the category-separation direction, variation of the competing manifold along that direction, and alignment between the two manifolds' variation axes. Thus SNR can be high even when manifolds remain extended, provided their useful separation is large and their variability lies mostly away from the decision direction.

## 3. Manifold classification capacity

Capacity is not ordinary 50-way AlexNet accuracy. Give every whole manifold a random binary label \(y_a\in\{-1,+1\}\) and ask whether one hyperplane can classify **every point** on every manifold with its assigned label. For \(P_c\) manifolds in an \(N\)-dimensional representation, the load is \(\alpha=P_c/N\). The capacity \(\alpha_M\) is the critical load below which typical random manifold dichotomies are linearly separable.

The exact correlated-manifold implementation cited by the paper performs the following preprocessing:

1. Concatenate all 50 manifolds and subtract their global sample mean.
2. Estimate low-rank correlation among the 50 manifold centers by factor analysis (`fun_FA`; 10 repetitions in the repository default). Select the fitted center-subspace rank \(K\), project all manifolds into its null space, and thereby remove shared center correlations.
3. For each projected manifold, compute its residual center \(c_a\), then center and normalize every point:

   \[
   S^a=\frac{X^a_{\perp}-c_a\mathbf1^\top}{\lVert c_a\rVert_2}.
   \]

4. Since \(N>P\), use an economy QR decomposition to express \(S^a\) in its at-most-50-dimensional span without changing its geometry. Append a constant center coordinate of 1 to every point, producing columns \(s^a_\mu=[S^a_\mu;1]\).

For each manifold independently, draw \(n_t\) Gaussian probe vectors \(t\sim\mathcal N(0,I)\) in this augmented space. For each probe, solve

\[
v^*(t)=\arg\min_v\frac12\lVert v-t\rVert_2^2
\quad\text{subject to}\quad
v\cdot s^a_\mu\ge\kappa\ \ \forall\mu,
\]

where \(\kappa=0\) is zero-margin capacity. From the solution, obtain the manifold anchor point \(\tilde s(t)\) (the support point, or convex combination of support points, touched by the separating hyperplane). The code evaluates

\[
\lambda(t)=
\frac{[t\cdot\tilde s(t)+\kappa]_+}{\lVert\tilde s(t)\rVert_2^2},
\qquad
\alpha_a^{-1}=\mathbb E_t\!\left[\lambda(t)^2\lVert\tilde s(t)\rVert_2^2\right],
\]

with the expectation approximated by the \(n_t\) probes. The reported layer-level capacity is the harmonic mean over manifolds,

\[
\boxed{
\alpha_M=\left(\frac1{50}\sum_{a=1}^{50}\alpha_a^{-1}\right)^{-1}
}
\]

not the arithmetic mean of the 50 returned capacities. Larger \(\alpha_M\) means more randomly labeled object manifolds can be linearly separated per feature dimension. In the limiting case where every manifold collapses to a point and \(\kappa=0\), capacity approaches the classical random-point value \(2\).

For the cited Python implementation, the concrete call is conceptually:

```python
capacity_each, radius_each, dimension_each, center_corr, K = \
    manifold_analysis_corr(manifolds, kappa=0, n_t=200)
capacity = 1 / np.mean(1 / capacity_each)
```

Here `manifolds` is a list of 50 arrays, each shaped `(2048, 50)` after the common random projection. `n_t=200` is the repository's recommended default; increasing it reduces Monte Carlo noise.

## 4. Decisions the CCN paper does not specify

The short paper identifies the data panel and repositories, but does **not** state two parameters needed to reproduce a single number:

1. **SNR shot count \(m\).** The Sorscher metric is a family indexed by \(m\), not a parameter-free scalar. The defining paper and its DNN notebooks use \(m=5\) as their standard analysis, so use \(m=5\) for the primary metric unless the CCN authors' configuration establishes otherwise. The CCN short paper itself does not print \(m\). Report it and preferably show sensitivity over \(m\in\{1,2,5,10,\infty\}\).
2. **Random projection.** The defining DNN pipeline projects layers wider than 2,048 dimensions to 2,048; AlexNet FC2 is therefore projected. The CCN short paper does not explicitly restate this extraction step, but its claim of using the defining source code makes the projected result the most source-faithful primary specification. Use the same fixed projection for both SNR and capacity. An unprojected result is a useful robustness check, not the primary reproduction.

Other items that must be fixed are the category/image sample, FC2 pre- versus post-ReLU convention, preprocessing, capacity random seeds, and whether SNR is summarized over directed pairs by the arithmetic mean. Without these details, “AlexNet FC2 SNR/capacity” is not a uniquely reproducible number.

## Sources

- Local CCN paper: [Model_manifold_analysis_CCN.pdf](./Model_manifold_analysis_CCN.pdf)
- Sorscher, Ganguli & Sompolinsky, *Neural representational geometry underlies few-shot concept learning* ([paper](https://ganguli-gang.stanford.edu/pdf/22.GeometryConcepts.pdf); [reference code](https://github.com/bsorsch/geometry-fewshot-learning))
- Chung, Lee & Sompolinsky, *Classification and Geometry of General Perceptual Manifolds* ([paper](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031003))
- Stephenson et al. implementation used by the CCN authors ([reference code and API](https://github.com/schung039/neural_manifolds_replicaMFT))
