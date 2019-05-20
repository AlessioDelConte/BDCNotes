Clustering

## Definition

Given a **set of points** belonging to some **space**, with a notion of **distance** between points, **clustering** aims at groping the point into a number of subsets (**clusters**) such that:

- Points in the same cluster are “**close**“ to one another.
- Points in different clusters are “**distant**” from one another.

The **distance** captures a notion of **similarity**: close points are similar, distant point are dissimilar.

A **clustering problem** is usually defined by requiring that clusters optimize a given **objective function**, and/or satisfy certain **properties**. Numerous clustering problems have been defined, studied and employed in applications over the years.



## Metric Space

Typically, the input of a clustering problem consist of a set of points from a **metric space**.

$\bold{\text{Definition}}$

A  **Metric space** is an ordered pair $(M, d)$ where $M$ is a set ad $d(\cdot)$ is a **metric** on $M$, i.e, a function:
$$
d : M \times M \rarr \R
$$

Such that for every $x,y,z \in M$ the following holds:

- $d(x,y) \ge 0$
- $d(x,y) = 0$ If and only if $x = y$
- $d(x,y) = d(y,x)$ Symmetry
- $d(x,z) \le d(x,y) + d(y,z)$ Triangle inequality

**Remark:** Often, different distance functions can be used to analyze a dataset through clustering . The choice of the function may have a significant impact on the effectiveness of the analysis.



## Distance functions

### Euclidean distance

Let $X,Y \in \R^n$, with $X = (x_1, x_2, \dots, x_n)$ and $Y=(y_1,y_2,\dots, y_n)$. For $r \ge 1$ the $L_r-$ distance between $X$ and $Y$ (a.k.a. $L_r-$norm)
$$
d_{Lr}(X,Y) = \bigg(\sum^n_{i=1}|x_i-y_i|^r\bigg)^{1/r}
$$

- $r = 2$: Standard distance in $\R^n$ (also denoted by $|| \cdot||$)

- $r = 1$: Referred to as *Manhattan distance*, it is the sum of the absolute differences of coordinates in each dimension. Used in grid-like environments.

- $r= \infty$: (Limit for $r$ that tends to $\infty$) it is the maximum absolute differences of coordinates, over all dimensions.  

  

  <img src="assets/distances.png" class="img_border" height=300> 



### Jaccard distance

Used when points are *sets* (e.g documents seen as bag of words). Let $S$ and $T$ be two sets over the same ground set of elements. The **Jaccard distance** between $S$ and $T$ is defined as:
$$
d_{Jaccard}(S,T) = 1 - \frac{|S \cap T|}{|S \cup T|}
$$
Note that the distance ranges in $[0,1]$ and it is $0$ iff $S=T$ and 1 iff $S$ and $T$ are disjoint. The value $\frac{|S \cap T|}{|S \cup T|}$ is referred to as the *Jaccard similarity* of the two sets.



<img src="assets/jaccard.png" height=180 width=450>



### Cosine (or angular) distance

It is used when points are *vectors* in $\R^n$ (or, in general in an inner-product space). Given two vectors $X = (x_1, x_2, \dots, x_n)$ and $Y=(y_1,y_2,\dots, y_n)$ in $R^n$, their **cosine distance** is the angle between them, that is:
$$
d_{cosine} = 
arcos\bigg(\frac{X\cdot Y}{||X||\cdot ||Y||}\bigg)=
arcos\bigg(\frac{\sum_{i=1}^nx_iy_i}
{\sqrt{\sum_{i=1}^n(x_i)^2}\sqrt{\sum_{i=1}^n(y_i)^2}}\bigg)
$$
**Example:** For $X=(1,2,-1)$ and $Y=(2,1,1)$ the cosine distance is:
$$
arcos\bigg(\frac{3}{\sqrt{6}\sqrt{6}}\bigg)=
arcos\bigg(\frac{1}{2}\bigg)=
\frac{\pi}{3}
$$
**Remark:** The cosine distance is often used in **information retrieval** to assess similarity between documents. Consider an alphabet of $n$ words. A document $X$ over this alphabet can be represented as an $n-Vector$, where $x_i$ is the number of occurrences in $X$ of the $i$-th word of the alphabet.

#### Observations on the cosine distance

For non-negative coordinates it takes values in $[0, \pi/2]$ while for arbitrary coordinates the range is $[0, \pi]$. Dividing by $\pi/2$ (or $\pi$) the range becomes $[0,1]$.

In order to satisfy the second property of distance functions for metric spaces scalar multiples of a vector must be regarded as the same vector.

<img src="assets/cosine_dist.png" height=200 width=450>



### Edit distance

Used for strings. Given two strings $X$ and $Y$, their **edit distance** is the minimum number of delation and insertions that must be applied to transform $X$ into $Y$.

**Example:** for $X=ABCDE$ and $Y=ACFDEG$, we have $d_{edit}(X,Y) = 3$, since we can transform $X$ into $Y$ as follows:
$$
ABCDE\ \mathop{\longrightarrow}^{delete\ B}\ 
ACDE\ \mathop{\longrightarrow}^{insert\ F}\ 
ACFDE\ \mathop{\longrightarrow}^{insert\ G}\ 
ACFDEG
$$
and less than 3 deletions/insertions cannot be used.



#### Equivalent definition

Given strings $X$ and $Y$, let their **Longest Common Subsequence** ($LCS(X,Y)$) be the longest string $Z$ whose characters occur in both $X$ and $Y$ in the same order but not necessarily contiguously. For example:
$$
LCS(ABCDE,ACFDEG)=ACDE
$$
It is easy to see that:
$$
d_{edit}(X,Y)=|X|+|Y|-2|LCS(X,Y)|
$$
$LCS(X,Y) $ can be computed in $O(|X|\cdot |Y|)$ time through dynamic-programming.



### Hamming distance

Used when points are *vectors* (as for cosine distance) over some $n$-dimensional space. Most commonly, it is used for binary vectors. The **Hamming distance** between two vectors is the number of coordinates in which they differ.

**Example:** for $X=(0,1,1,0,1)$ and $Y=(1,1,1,0,0)$  (i.e, $n=5$)
$$
d_{Hamming}(X,Y)=2
$$
Since they differ in the first and last coordinates.



## Curse of dimensionality

Random points in a high-dimensional metric space tend to be 

- Sparse.
- Almost equally distant from one another.
- Almost orthogonal (as vectors) to one another.

As consequence, in high dimensions distance functions may loose effectiveness in assessing similarity/dissimilarity. However, this holds in particular when points are random, and it might be less of a problem in some real-world datasets.



## Types of clusterings

Given a set of points in a metric space, hence a distance between points, a clustering problem often specifies an **objective function** to optimize. The objective function also allows to compare different solutions.

Objective functions can be categorized based on whether or not:

- A target number $k$ of clusters is given in input.

- For each cluster a **center** must be identified.

  When centers are required, the value of the objective function depends on the selected centers. Cluster centers must belong to the underlying metric space but, sometimes, they are constrained to belong to the input set.

- **Disjoint clusters** are sought.



## Background

### Combinatorial optimization problem

A (combinatorial) optimization problem is a computational problem $\Pi(I, S, \Phi,m)$ defined by a set of *instances* $I$, a set of *solutions* $S$, an *objective function* $\Phi$ that assigns a real value to each solution $s \in S$ , and $m \in \{\min, \max\}$. For each instance $i\in I$ the problem require to find an *optimal solution*, namely a feasible solution $s\in S$, which **minimizes $\Phi(s)$** if $m = \min$ (minimization problem), or  **maximizes $\Phi(s)$**, if $m=\max$ (maximization problem)

**Observation:** For several important optimization problems no algorithms are known which find an optimal solution in time polynomial in the input size, and it is unlikely that such algorithms exist (***NP-hard problems***). Therefore, especially when thinking of big data, one has to aim at approximations of optimal solutions.

####Approximation algorithm

For $c \ge 1$, a **$\bold{c}$-approximation algorithm $A$** for a combinatorial optimization problem  $\Pi(I, S, \Phi,m)$  is an algorithm that for each instance $i \in I$ returns a feasible solution $A(i) \in S_i$ such that $\Phi(A(i)) \le c\min_{s\in S_i}\Phi(s)$, if $m = \min$, or $\Phi(A(i)) \ge (1/c)\max_{s\in S_i}\Phi(s)$, if $m = \max$. The value $c$ is called **approximation ratio**, and the solution $A(i)$ is called **$\bold{c}$-approximate solution** or **$\bold{c}$-approximation**.

**Observation:** the approximation introduces a flexibility in selecting the solution to a given instance, which algorithms can exploit to run more efficiently.



## Center-based clustering 

Let $P$ be a set of $N$ points in metric space $(M,d)$, and let $k$ be the target number of clusters, $1 \le k \le N$. We define a $k\text{-clustering}$ of $P$ as a tuple $C=(C_1,C_2,\dots,C_k;c_1,c_2,\dots,c_k)$ where:

- $(C_1,C_2,\dots,C_k)$ Defines a partition of $P$ i.e, $P=C_1\cup C_2 \cup\dots\cup C_k$ 
- $c_1,c_2,\dots,c_k$ Are suitably selected **centers** for the clusters, where $c_i \in C_i$ for every $1 \le i \le k$

Observe that the above definition requires the centers to belong to the clusters, hence to the pointset. Whenever appropriate we will discuss the case when the centers can be chosen more freely from the metric space.

$\bold{\text{Definition}}$

Consider a metric space $(M,d)$ and an integer $k>0$. A $k$-clustering problem for $(M,d)$ is an optimization problem whose instances are the finite point sets $P \sube M$. For each instance $P$, the problem requires to compute a $k$-clustering $C$ of $P$ (feasible solution) which **minimizes** a suitable objective function $\Phi(C)$.

The following are three popular objective functions.

- $\Phi_{kcenter}(C) = \max^k_{i=1}\max_{a \in C_i} d(a, c_i)$ 		(**k-center clustering**)
- $\Phi_{kmeans}(C) = \sum^k_{i=1}\sum_{a \in C_i} (d(a, c_i))^2$ 	 	(**k-means clustering**)
- $\Phi_{kcenter}(C) = \sum^k_{i=1}\sum_{a \in C_i} d(a, c_i)$ 			(**k-median clustering**)

In other words, under the above functions, the problem requires to find the $k$-clustering that minimizes, respectively, the maximum distance (k-center), or the average square distance (k-means), or the average distance (k-median) of a point to its cluster center.

### Observations

- All aforementioned problems (k-center, k-means, k-median) are ***NP-hard***. Hence, in general it is impractical to search for optimal solutions.
- There are several efficient approximation algorithms that in practice return good-quality solutions. However, dealing efficiently with large inputs is still a challenge! 
- K-center and k-median belong to the family of **facility-location problems**. In these problems, a set $F$ of candidate *facilities* and a set $C$ of clients are given and the objective is to find a subset of at most $k$ candidate facilities to open and an assignment of clients to them, so to minimize some the maximum or average distance between a client and its assigned facilities. In our formulation, each input point represents both a facility and a client. Numerous variants of these problems have been studied in the literature. 
- k-means objective is also referred to as Sum of **Squared Errors (SSE).**

### Notations

Let $P$ be a point-set from a metric space $(M,d)$. For every integer $k \in [1, |P|]$ we define:

- $\Phi^{opt}_{kcenter}(P,k)$ = minimum value of $\Phi_{kcenter}(C)$ over all possible $k$-clustering $C$ of $P$.

- $\Phi^{opt}_{kmeans}(P,k)$ = minimum value of $\Phi_{kmeans}(C)$ over all possible $k$-clustering $C$ of $P$.
- $\Phi^{opt}_{kmedian}(P,k)$ = minimum value of $\Phi_{kmedian}(C)$ over all possible $k$-clustering $C$ of $P$.

Also, for any point $p \in P$ and any subset $S \sube P$ we define;
$$
d(p,S) = \min\{q \in S : d(p,q)\}
$$
Which generalizes the distance function to distances between points and sets.



## Partitioning primitive

Let $P$ be a point-set and $S \sube P$ a set of $k$ selected centers. For *all* previously defined clustering problems, the best $k$-clustering around these centers is the one where each $c_i$ belongs to a distinct cluster and each point is assigned to the cluster of the closest $c_i$ (ties broken arbitrarily).

We will use primitive $\bold{\text{Partition(P,S)}}$ to denote this task:
$$
\begin{multline*}
\begin{aligned}
& \bold{\text{Partition(P,S)}}\\
&\text{Let}\ S=\{c_1,c_2,\dots, c_k\} \sube P \\
&\bold{\text{for}}\ i\larr 1\ \bold{\text{to}}\ k\ \bold{\text{do}}\ C_i\larr\{c_i\}\\
&\quad l \larr \text{argmin}_{i=1,k}\{d(p,c_i)\}\quad //\ ties\ broken\ arbitrarily\\
&\quad C_l \larr C_l \cup\{p\}\\
&C \larr (C_1,C_2,\dots, C_k;c_1,c_2,\dots,c_k)\\
&\bold{\text{return}}\ C
\end{aligned}
\end{multline*}
$$


## K-Center Clustering

### Farthest-First Traversal

#### Algorithm

It is a popular 2-approximation sequential algorithm developed by T.F. Gonzales. Simple and, somewhat fast, implementation when the input fits in main memory. Powerful primitive for extracting samples for further analyses.

$\bold{\text{Input}}$ Set $P$ of $N$ points from metric space $(M,d)$, integer $k > 1$

$\bold{\text{Output}}$ $k$-clustering $C=(C_1,C_2,\dots,C_k;c_1,c_2,\dots,c_k)$ of $P$ which is a good approximation to the $k$-center problem for P.
$$
\begin{multline*}
\begin{aligned}
& \bold{\text{Farthest-First Traversal(P)}}\\
&S \larr \{c_1\}\quad //c_1 \in P\ arbitrary\ point \\
&\bold{\text{for}}\ i\larr 2\ \bold{\text{to}}\ k\ \bold{\text{do}}\\
&\quad \text{Find the point}\ c_i\in P - S\ \text{that maximizes}\ d(c_i,S)\\
&\quad S \larr S \cup \{c_i\}\\
&\bold{\text{return}}\ \text{Partition(P,S)}
\end{aligned}
\end{multline*}
$$
**Observation:** The invocation of $\bold{\text{Partition(P,S)}}$ at the end can be avoided by maintaining the best assignment of the points to the current set of centers in every iteration of the for-loop.

**Example:**

<table>
  <tr>
    <td><img src="assets/c1.png"></td>
    <td><img src="assets/c2.png"></td>
  </tr>
  <tr>
    <td><img src="assets/c3.png"></td>
    <td><img src="assets/c4.png"></td>
  </tr>
    <tr>
    <td><img src="assets/c5.png"></td>
    <td><img src="assets/c6.png"></td>
  </tr>
</table>



#### Analysis

$\bold{\text{Theorem}}$

*Let $C_{alg}$ be the $k$-clustering of $P$ returned by the Farthest-First Traversal algorithm. Then:*
$$
\Phi_{kcenter}(C_{alg})\le 2 \cdot \Phi_{kcenter}^{opt}(P,k)
$$
*That is, Farthest-First Traversal is a 2-approximation algorithm.*

$\bold{\text{Proof}}$

Let $S=\{c_1,c_2,\dots,c_k\}$ be the set of centers returned by Farthest-Fist Traversal with input $P$, and let $S_i=\{c_1,c_2,\dots,c_i\}$ be the partial set of centers determined up to iteration $i$ of the for-loop, with $2 \le i \le k$.

It is easy to see that:
$$
\begin{align}
d(c_i, S_{i-1})\quad &\le \quad d(d_j,S_{j-1})\ \forall2 \le j < i \le k \\
d(p, S_{i-1})\quad &\le \quad d(d_j,S_{j-1})\ \forall p \in P - S_{i-1}\ \text{and}\ \le j < i \le k
\end{align}
$$
Let $q$ be the point with maximum $d(q,S)$ (i.e, distance between $q$ and the closest center). Therefore, $\Phi_{kcenter}(C_{alg})=d(q,S)=d(q,S_k)$. Let $c_i$ be the closest center to $q$, that is, $d(q,S)=d(q, c_i)$.

We now show that the $k+1$ points $c_1,c_2,\dots,c_k,q$ are such that any two go them are at distance at least $\Phi_{Kcenter}(C_{alg})$ from one another.

Indeed, since we assumed $c_i$ to be the closest center to $q$, we have:
$$
\Phi_{kcenter}(C_{alg})=d(q,c_i)\le d(q,c_j)\quad \forall j\ne i
$$
Also, by Inequalities (12) and (13), we have that for any pair of indices $j_1,j_2$, with $1 \le j_1 < j_2 \le k$,
$$
\Phi_{kcenter}(C_{alg})=d(q,S_k)\le d(c_k,S_{k-1}) \le d(c_{j_2}, S_{j_2-1}) \le d(c_{j_2}, c_{j_1})
$$
Now we know that all pairwise distances between $c_1,c_2,\dots,c_k, q$ are at least $\Phi_{kcenter}(C_{alg})$. Clearly, by the *pigeonhole principle*, two of these $k+1$ points must fall in the same cluster of the optimal clustering.

Suppose that $x,y$ are two points in $\{c_1,c_2,\dots,c_k,q\}$ which belong to the same cluster of the optimal clustering with center $c$. By the triangle inequality, we have that:
$$
\Phi_{kcenter}(C_{alg})\le d(x,y)\le d(x,c) + d(c,y)
\le 2 \Phi^{opt}_{kcenter}(P,k),
$$
and the theorem follows. 

<div style="font-size:30px" align="right">□</div> 

<img src="assets/fft.png">



### Observations on k-center clustering

- K-center clustering provides a strong guarantee on how close **each** point is to the center of its cluster.

- However, for **noisy point-sets** (e.g,  **point-sets whit outliers**) the clustering which optimizes the $k$-center objective may obfuscate some “natural” clustering inherent in the data.

- For any fixed $\epsilon > 0$ it is NP-hard to compute a $k$-clustering $C_{alg}$ of a points $P$ with
  $$
  \Phi_{kcenter}(C_{alg})\le (2-\epsilon)\Phi^{opt}_{kcenter}(P,k)
  $$
  Hence the Farthest-First Traversal is likely to provide almost the best approximation guarantee obtainable in polynomial time.

  

  