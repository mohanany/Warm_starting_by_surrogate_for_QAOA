

**Theorem 1.**  
Let  $N\ge 2$ . Let  $P\in\mathbb{R}^N$  and define the symmetric zero-diagonal coupling matrix  $J'(P)\in\mathbb{R}^{N\times N}$  by

$$
J'_{ij}(P)= \begin{cases} P_i+P_j, & i\neq j,\\ 0, & i=j. \end{cases}
$$

Consider the Ising energy (equivalently, Hamiltonian) over spin configurations  $s\in\{\pm1\}^N$ ,

$$
E_{J'}(s)\triangleq -\frac{1}{2}\,s^\top J'(P)\,s.
$$

Then:

1.  For every  $s\in\{\pm1\}^N$ ,
    
$$
E_{J'}(s)= - \Big(\sum_{i=1}^N P_i s_i\Big)\Big(\sum_{i=1}^N s_i\Big) + \sum_{i=1}^N P_i . \tag{1}
$$
2.  Let  $\pi$  be a permutation that sorts  $P$  in nonincreasing order, i.e.,  $P_{\pi(1)}\ge P_{\pi(2)}\ge \dots \ge P_{\pi(N)}$ .  
    There exists an optimal ground-state configuration  $s^\star$  whose **sorted** form  $s^\star_{\pi(1)},\dots,s^\star_{\pi(N)}$  is a **two-block (threshold) vector**:
    
$$
(s^\star_{\pi(1)},\dots,s^\star_{\pi(N)})= (\underbrace{+1,\dots,+1}_{M^\star}, \underbrace{-1,\dots,-1}_{N-M^\star}) \quad\text{or its global flip,} \tag{2}
$$

for some  $M^\star\in\{0,1,\dots,N\}$ .

3.  Consequently, the ground state of  $J'(P)$  can be found by sorting  $P$  and selecting the best threshold  $M$  (as exploited by the analytic solver in \[6\]).

**Remark.** This theorem formalizes the analytic solvability of the structured Ising family used in \[6\]. The explicit sorting step resolves index-label dependence: the optimal pattern is a threshold **in the ordered basis**, and can be mapped back to the original indexing by  $\pi^{-1}$ .

* * *

Proof
-----

### Part (1): Closed-form energy identity

Because  $J'(P)$  is symmetric with  $J'_{ii}=0$ , we have

$$
s^\top J'(P)\,s = \sum_{i\neq j} (P_i+P_j)s_i s_j.
$$

Split the sum:

$$
\sum_{i\neq j}(P_i+P_j)s_is_j = \sum_{i\neq j}P_i s_i s_j + \sum_{i\neq j}P_j s_i s_j.
$$

By swapping dummy indices  $(i,j)$  in the second term, the two terms are equal, hence

$$
\sum_{i\neq j}(P_i+P_j)s_is_j = 2\sum_{i\neq j}P_i s_i s_j.
$$

Now rewrite

$$
\sum_{i\neq j}P_i s_i s_j = \sum_{i=1}^N P_i s_i \sum_{j\neq i} s_j = \sum_{i=1}^N P_i s_i\,(S - s_i),
$$

where  $S\triangleq \sum_{j=1}^N s_j$ . Since  $s_i^2=1$ ,

$$
\sum_{i=1}^N P_i s_i\,(S-s_i) = S\sum_{i=1}^N P_i s_i - \sum_{i=1}^N P_i.
$$

Therefore,

$$
s^\top J'(P)\,s = 2\Big( S\sum_{i=1}^N P_i s_i - \sum_{i=1}^N P_i\Big).
$$

Plugging into  $E_{J'}(s)=-\tfrac12 s^\top J'(P)s$  yields

$$
E_{J'}(s)= -S\sum_{i=1}^N P_i s_i + \sum_{i=1}^N P_i,
$$

which is exactly (1). ∎

* * *

### Part (2): Threshold optimality in the sorted basis

Fix any integer  $M\in\{0,1,\dots,N\}$ , and consider the subset

$$
A \triangleq \{i:\ s_i=+1\},\qquad |A|=M.
$$

Then the magnetization is fixed by  $M$ :

$$
S=\sum_{i=1}^N s_i = M-(N-M)=2M-N. \tag{3}
$$

Define the weighted sum

$$
W \triangleq \sum_{i=1}^N P_i s_i = \sum_{i\in A} P_i - \sum_{i\notin A} P_i = 2\sum_{i\in A} P_i - \sum_{i=1}^N P_i. \tag{4}
$$

Using (1), minimizing  $E_{J'}(s)$  is equivalent (up to the constant  $\sum_i P_i$ ) to **maximizing** the product  $S\cdot W$ :

$$
E_{J'}(s)= -S\,W + \sum_{i=1}^N P_i \quad\Longrightarrow\quad \arg\min_s E_{J'}(s)=\arg\max_s \big(SW\big). \tag{5}
$$

For fixed  $M$ , the value  $S$  is fixed by (3). Hence, for fixed  $M$ , maximizing  $SW$  is equivalent to:

*   maximize  $W$  if  $S>0$  (i.e.,  $M>\tfrac{N}{2}$ ),
*   minimize  $W$  if  $S<0$  (i.e.,  $M<\tfrac{N}{2}$ ),
*   any  $W$  if  $S=0$  (i.e.,  $M=\tfrac{N}{2}$ ), since  $SW=0$ .

By (4),  $W$  is an affine function of  $\sum_{i\in A}P_i$ . Therefore, for fixed  $M$ ,

*   maximizing  $W$  is equivalent to choosing  $A$  as the set of indices of the ** $M$  largest** entries of  $P$ ,
*   minimizing  $W$  is equivalent to choosing  $A$  as the set of indices of the ** $M$  smallest** entries of  $P$ .

Let  $\pi$  sort  $P$  in nonincreasing order. Then the set of the  $M$  largest entries is precisely  $\{\pi(1),\dots,\pi(M)\}$ . Hence, for each fixed  $M$ , there exists an optimal configuration whose spins are  $+1$  on  $\pi(1),\dots,\pi(M)$  and  $-1$  on  $\pi(M+1),\dots,\pi(N)$ , i.e., a threshold vector in the sorted order. Optimizing over all  $M\in\{0,\dots,N\}$  yields an optimal ground state of the form (2). ∎

* * *

### Part (3): Permutation (indexing) invariance and “sorting then mapping back”

Let  $P_\pi$  denote the permuted vector  $(P_{\pi(1)},\dots,P_{\pi(N)})$ . Define the permutation matrix  $Q$  such that  $P_\pi = Q^\top P$  and  $s_\pi = Q^\top s$ . Then

$$
J'(P_\pi)=Q^\top J'(P)\,Q
$$

and thus

$$
E_{J'(P)}(s)= -\tfrac12 s^\top J'(P)s = -\tfrac12 s_\pi^\top J'(P_\pi)s_\pi =E_{J'(P_\pi)}(s_\pi).
$$

Therefore, solving in the sorted basis (where the optimizer is a threshold vector) and then mapping back via  $\pi^{-1}$  yields a ground state for the original indexing. This is exactly the robust implementation strategy consistent with the analytic solution exploited in \[6\]. ∎

* * *

Practical corollary (what our code is doing vs. what \[6\] assumes)
--------------------------------------------------------------------

**Corollary 1.**  
For any labeling (any permutation of indices), if the couplings satisfy the structural condition  $J'_{ij}=P_i+P_j$  (with zero diagonal), then the ground state is always obtainable by:

1.  sorting  $P$ ,
2.  selecting the best threshold  $M$ ,
3.  mapping the threshold spin vector back to the original indexing.

This does **not** enlarge the solvable model class beyond \[6\]; it makes the required ordering explicit and ensures the solver is correct independent of index labeling, fully consistent with \[6\].

* * *

[6] A. Rezaei, M. Hasani, A. Rezaei, and S. M. H. Halataei, “Continuous Approximation of the Ising Hamiltonian: Exact Ground States and Applications to Fidelity Assessment in Ising Machines,” arXiv:2411.19604v3 [physics.comp-ph], Nov. 2025. [Online]. Available: https://arxiv.org/abs/2411.19604v3