# Theory Boundary

The experiment asks whether a reward model should be selected for preference fit or for
the local policy update it induces.

Let `g(r) = E[s r]`, with prompt centering used to remove reward gauges, and let `F` be
the Fisher matrix of the reference policy in the trainable LoRA-B tangent. The local
KL-regularized policy optimum induced by reward `r` has beta-free direction

```text
d_r = F^-1 g(r).
```

The corresponding local regret to the oracle direction is

```text
R_beta(r) = (1 / (2 beta))
            * (g(r) - g(r*))^T F^-1 (g(r) - g(r*)).
```

Finite data uses `F + lambda I`. This ridge-stabilized empirical quantity is a local
surrogate, not an exact identity for a finite neural-network update.

MLE-RM projects oracle pair probabilities onto the linear reward class in Bernoulli-KL
geometry. Pro-RM projects the oracle reward moment in policy geometry. Under reward-class
misspecification, these projections need not agree. The desired experimental case is one
where MLE-RM fits preferences at least as well but Pro-RM produces lower local regret and
better regularized policy utility.

The Skywork model is an operational oracle, not latent human utility. Its parameter count
is not the mathematical dimension of the true reward function. The controlled dimension
comparison is instead between an unrestricted oracle reward over responses, the 7168
coordinate policy tangent, and the 1536-dimensional learned linear reward head.
