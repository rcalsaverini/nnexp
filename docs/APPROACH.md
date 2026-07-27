# Architectural Approach: LLMs, Constrained Generation, and PPO

To fulfill the objectives outlined in the Research Program, this repository implements a generalized architectural strategy. We leverage pre-trained Large Language Models (LLMs) and strictly constrain their generative space, bridging the discrete optimization gap via Reinforcement Learning and highly optimized parameter-efficient tuning.

## End-to-End Architectural Pipeline

Our system maps arbitrary domain contexts directly to discrete symbolic expressions without relying on custom, recursive tree-emitting neural networks. The architecture is organized into four distinct, modular phases:

### Phase I: Context Ingestion
To generate an optimal expression, the model must be conditioned on the environment or task parameters (e.g., target coordinates, positive/negative string sets, or unit test specifications). Depending on the problem domain, this context is either tokenized as a standard textual prompt or ingested via a domain-specific continuous projection layer (bypassing the tokenizer) to form the base embedding sequence.

### Phase II: Constrained Processing
The context embeddings are processed by a pre-trained causal LLM. To meet local hardware constraints, the base model is frozen and injected with trainable Quantized Low-Rank Adaptation (QLoRA) matrices. Before initialization, the model undergoes **Vocabulary Surgery**, slicing away tens of thousands of parameters to leave only the required domain symbols. As the model autoregressively outputs tokens, a strict Finite State Machine applies a **Grammar Mask**, forcing the probability of any syntactically invalid operation to exactly zero, guaranteeing that the model yields a valid Abstract Syntax Tree (AST) for the target domain.

### Phase III: Black-Box Evaluation and RL Optimization
Evaluating an AST against an environment to yield an error or utility metric is inherently discrete and non-differentiable. We utilize **Proximal Policy Optimization (PPO)** to bypass this bottleneck. The generated AST string is executed by an isolated, domain-specific Python interpreter (e.g., a regex matcher, a math evaluator, or a game engine). The interpreter returns a scalar reward metric. PPO uses this scalar to directly update the QLoRA adapter weights, teaching the model to increase the probability of generating high-reward expressions.

### Phase IV: Inference Compression
Once the policy converges, the model is prepared for high-speed local inference using **Wanda Pruning**. This carves out dead attention blocks without requiring any gradient updates, leaving a highly sparse, ultra-fast symbolic optimization engine.

---

## Grammar-Guided Generation (Constrained Decoding)

Traditional language models sample from a categorical distribution over their entire vocabulary. Constrained decoding methods enforce syntactic correctness at inference time by masking invalid tokens. 

Let the vocabulary be denoted as $\mathcal{V}$ and the raw logits output by the model at step $t$ be $z_t \in \mathbb{R}^{|\mathcal{V}|}$. We define a Finite State Machine (FSM) that tracks the current AST state $s_t$. The FSM dictates the set of valid next tokens, $\mathcal{V}_{valid}(s_t) \subseteq \mathcal{V}$. 

A constraint mask $M(a, s_t)$ is applied to the raw logits prior to sampling:

$$
M(a, s_t) = \begin{cases} 
0 & \text{if } a \in \mathcal{V}_{valid}(s_t) \\ 
-\infty & \text{if } a \notin \mathcal{V}_{valid}(s_t) 
\end{cases}
$$

The final generation probability for token $a_t$ becomes:

$$
\pi_\theta(a_t|s_t) = \frac{\exp(z_t(a_t) + M(a_t, s_t))}{\sum_{a' \in \mathcal{V}} \exp(z_t(a') + M(a', s_t))}
$$

This guarantees that the probability mass for any syntactically illegal token evaluates to exactly zero.

## Pre-Training Optimization: Vocabulary Surgery

Standard LLMs carry a massive vocabulary matrix (often 30,000+ tokens). Since symbolic grammars typically only require a tiny fraction of these (e.g., $K \approx 30$ tokens), computing the full logit vector is computationally wasteful. 

Before training, the model's token embedding matrix $\mathbf{W}_{embed} \in \mathbb{R}^{|\mathcal{V}| \times d}$ and language modeling head $\mathbf{W}_{head} \in \mathbb{R}^{d \times |\mathcal{V}|}$ are surgically sliced. Let $\mathcal{I}_{target}$ be the ordered list of indices corresponding to our valid grammar tokens. We construct the pruned matrices by gathering these specific rows/columns:

$$
\mathbf{W}_{embed}^{pruned} = \mathbf{W}_{embed}[\mathcal{I}_{target}, :] \quad \in \mathbb{R}^{K \times d}
$$

$$
\mathbf{W}_{head}^{pruned} = \mathbf{W}_{head}[:, \mathcal{I}_{target}] \quad \in \mathbb{R}^{d \times K}
$$

This reduces the final matrix multiplication from $\mathcal{O}(d \times |\mathcal{V}|)$ down to $\mathcal{O}(d \times K)$, exponentially accelerating the forward pass.

## Quantized Low-Rank Adaptation (QLoRA)

To manage memory constraints, the network undergoes parameter-efficient fine-tuning using QLoRA. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters.

During fine-tuning, the goal is to learn an additional matrix $\Delta W$, such that $y = (W + \Delta W)x = Wx + \Delta Wx$. We freeze the matrix $W$ and solve the optimization task to find matrices $A$ and $B$ that contain far fewer parameters than $\Delta W$. For an intermediate hidden state $x$, the updated projection becomes:

$$
y = \mathbf{W}x + \mathbf{B}\mathbf{A}x
$$

Where $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$ is fixed, $\mathbf{A} \in \mathbb{R}^{r \times d_{in}}$, and $\mathbf{B} \in \mathbb{R}^{d_{out} \times r}$. The rank $r$ is chosen such that $r \ll \min(d_{in}, d_{out})$. 

## Proximal Policy Optimization (PPO)

Because evaluating an AST to yield a utility metric is a discrete, non-differentiable process, we rely on Reinforcement Learning. 

The model generates a discrete AST, which is evaluated against the domain-specific environment to calculate a raw scalar reward $R$. Let $r_t(\theta)$ denote the probability ratio between the updated policy $\pi_\theta$ and the old policy $\pi_{\theta_{old}}$:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

PPO maximizes the clipped surrogate objective to prevent destructively large weight updates:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
$$

Where $\hat{A}_t$ is the estimated advantage of the selected token, and $\epsilon$ is the trust-region clipping hyperparameter.

## Post-Training Sparsification (Wanda)

To compress the final model for high-speed local inference, we apply Wanda (Pruning by Weights and activations). Compared to magnitude pruning, Wanda removes weights on a per-output basis by the product of weight magnitudes and input activation norms.

Let $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$ be a weight matrix, and $\mathbf{X} \in \mathbb{R}^{N \times C_{in}}$ be a set of input activations. The Wanda metric evaluates each individual weight $W_{i,j}$ as:

$$
S_{i,j} = |W_{i,j}| \cdot ||\mathbf{X}_{:, j}||_2
$$

For each output neuron $i$, we sort the metric scores $S_{i,j}$ across its input connections $j$. We zero out the connections with the lowest scores until the target sparsity ratio $s$ is reached. Notably, Wanda requires no retraining or weight updates.