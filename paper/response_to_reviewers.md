# Formal Point-by-Point Author Response to Reviewers

**Manuscript ID:** 1637  
**Title:** *Spatiotemporal Deepfake Detection via Texture-Enhanced Multi-Attentional ResNeXt50-BiLSTM Architecture*  
**Submission Category:** Springer LNCS Full Research Paper  
**Recommendation:** Major Revision  
**Authors:** Aarav Rajput, Trapti Sharma, Rajit Nair, Hasan Alkahtani, Adel Musaad Alrasheedi, Sami Morsi, Ahmed A.F. Osman, Theyazn H.H. Aldhyani  

---

## General Remarks

We sincerely thank the Editor and the Reviewers for their constructive, insightful, and comprehensive review of our manuscript (**Manuscript ID: 1637**). We deeply appreciate the Reviewer's recognition of the significance and relevance of our research.

In accordance with the Reviewer's recommendations, we have conducted an extensive, rigorous revision of the manuscript:
1. **Mathematical Rigor:** Reformulated the Regional Independence Loss ($\mathcal{L}_{RIL}$) as a dual spatial-feature regularizer with explicit unit-normalized cosine similarity, formalized the Bilinear Attention Pooling (BAP) matrix algebraic equations, and detailed the linear projection bottleneck and recurrent dimensional transitions.
2. **Experimental Protocols:** Clarified the unseen cross-category evaluation methodology on DF40, justified static image temporal replication via dynamical fixed-point convergence, and rigorously eliminated data leakage via subject-disjoint 5-fold partitioning.
3. **Statistical Analysis:** Augmented empirical findings with paired two-tailed $t$-tests across folds ($df=4$), Bonferroni corrections ($\alpha_{adj} = 0.00625$), Benjamini-Hochberg False Discovery Rate ($q < 0.05$) adjustments, Shapiro-Wilk normality tests, and Wilcoxon signed-rank tests.

Below is our detailed, point-by-point response to each major concern raised in the review report.

---

## Detailed Point-by-Point Responses

### Major Concern 1: Regional Independence Loss ($\mathcal{L}_{RIL}$) Formulation
> **Reviewer Comment:**  
> *"The formulation of the Regional Independence Loss ($\mathcal{L}_{RIL}$) is questionable. The manuscript claims that $\mathcal{L}_{RIL}$ penalizes overlapping attention maps, but the proposed loss is actually computed from the cosine similarity between regional feature vectors ($V_i$) and ($V_j$). This does not directly measure spatial overlap between attention maps. Therefore, the claim that the loss 'forces each attention head to cover a distinct facial region' is not sufficiently supported by the presented formulation."*

**Author Response:**  
We thank the Reviewer for identifying this crucial theoretical discrepancy. The Reviewer is completely correct: cosine similarity between regional feature vectors operates in the latent descriptor space rather than measuring 2D spatial overlap across the feature grid.

To resolve this issue with complete mathematical rigor, we have reformulated $\mathcal{L}_{RIL}$ in **Section 3.4.1 (Equations 8–11)** as a **dual-regularized loss function** that explicitly decouples 2D spatial co-activation from regional feature diversity:
$$\mathcal{L}_{RIL} = \mathcal{L}_{spatial} + \gamma \cdot \mathcal{L}_{feat}, \quad (\gamma = 1.0)$$

1. **Pairwise Spatial Co-Activation Penalty ($\mathcal{L}_{spatial}$):**  
   Each 2D spatial attention map $A_k \in \mathbb{R}^{H \times W}$ is normalized over the spatial grid via spatial softmax:
   $$\sum_{x=1}^{H} \sum_{y=1}^{W} A_k(x,y) = 1, \quad \forall k \in \{1, \dots, M\}$$
   The spatial penalty directly computes the inner product between pairs of spatial probability distributions, averaged across all $\binom{M}{2}$ unique head pairs:
   $$\mathcal{L}_{spatial} = \frac{2}{M(M-1)} \sum_{i=1}^{M-1} \sum_{j=i+1}^{M} \sum_{x=1}^{H} \sum_{y=1}^{W} A_i(x,y) \cdot A_j(x,y)$$
   Because $A_i$ and $A_j$ are non-negative normalized distributions, $\sum_{x,y} A_i(x,y) A_j(x,y)$ is the dot product between the two flattened 2D spatial probability distributions. Minimizing $\mathcal{L}_{spatial}$ penalizes simultaneous activation at identical spatial coordinates, **encouraging the $M=4$ attention heads to place their probability mass on spatially distinct facial regions**.

2. **Margin-Based Feature Diversity Regularizer ($\mathcal{L}_{feat}$):**  
   To reduce representation redundancy in descriptor space, we define the unit-normalized regional feature vector:
   $$\hat{V}_k = \frac{V_k}{\max(\|V_k\|_2, \epsilon)}, \quad k \in \{1, \dots, M\}$$
   where $\epsilon = 10^{-8}$ ensures numerical stability. For descriptors with $\|V_k\|_2 \ge \epsilon$, $\hat{V}_k$ is the unit-normalized descriptor and the inner product $\hat{V}_i^T \hat{V}_j$ equals their exact cosine similarity $\cos(V_i, V_j) \in [-1, 1]$. The feature diversity regularizer applies a margin-based hinge penalty:
   $$\mathcal{L}_{feat} = \frac{2}{M(M-1)} \sum_{i=1}^{M-1} \sum_{j=i+1}^{M} \max\big(0,\, \hat{V}_i^T \hat{V}_j - m\big)$$
   where $m = 0.2$ is the margin hyperparameter.

3. **Clarifications on Mechanism and Specialization:**
   * **Feature Diversity vs. Strict Orthogonality:** $\mathcal{L}_{feat}$ enforces *feature diversity* rather than rigid algebraic orthogonality $(\hat{V}_i^T \hat{V}_j)^2$. The hinge formulation permits natural semantic correlations below margin $m=0.2$ while penalizing collinear feature collapse.
   * **Emergent Specialization:** The spatial loss encourages spatial separation, while the specific regional specialization of each head is an emergent property learned from task gradients, architectural inductive biases, and input data distribution.
   * **Dual Complementarity:** $\mathcal{L}_{spatial}$ reduces **spatial co-activation** on the 2D grid (*where heads attend*), while $\mathcal{L}_{feat}$ reduces **excessive feature-level similarity** in the 2304-dimensional regional feature space (*what heads represent*). Together, the proposed RIL encourages regional independence through complementary regularization of spatial co-activation and feature-level similarity among attention heads.

* **Manuscript Changes:** Section 3.4.1 (Equations 8–11) and the ablation analysis in Section 4.3 (Table 6) have been thoroughly updated to reflect this dual-regularization framework.

---

### Major Concern 2: Definition of Bilinear Attention Pooling (BAP)
> **Reviewer Comment:**  
> *"The definition of Bilinear Attention Pooling (BAP) is ambiguous. The manuscript uses the term 'outer product', but does not provide a formal matrix formulation. Furthermore, it is unclear whether feature concatenation ($F_{tex}$ and $F_{sem}$) occurs before or after attention weighting."*

**Author Response:**  
We appreciate the Reviewer's request for formal clarity. We have revised **Section 3.3.3 (Equations 3–6)** to provide the explicit matrix algebraic definition of Bilinear Attention Pooling and clarify the precise order of tensor operations.

1. **Channel-Wise Concatenation (Prior to Attention):**  
   Let $F_{sem} \in \mathbb{R}^{2048 \times H \times W}$ denote the deep semantic feature map from ResNeXt50 Stage 4 and $F_{tex} \in \mathbb{R}^{256 \times H \times W}$ denote the shallow texture feature map from the TEB ($H=8, W=8$). Channel-wise concatenation occurs **first** to form the unified spatial representation tensor:
   $$F = [F_{sem} \,\|\, F_{tex}] \in \mathbb{R}^{C \times H \times W}, \quad \text{where } C = 2048 + 256 = 2304$$

2. **Formal Matrix Bilinear Attention Pooling:**  
   Let $\mathbf{F} \in \mathbb{R}^{C \times K}$ denote the unrolled spatial feature matrix ($K = H \times W = 64$) and let $\mathbf{A} = [A_1; A_2; \dots; A_M] \in \mathbb{R}^{M \times K}$ denote the stacked attention weight matrix ($M=4$). Bilinear Attention Pooling is formally computed via matrix multiplication:
   $$\mathbf{V} = \mathbf{F} \, \mathbf{A}^T = \big[V_1, V_2, \dots, V_M\big] \in \mathbb{R}^{C \times M} = \mathbb{R}^{2304 \times 4}$$
   where each column descriptor $V_k \in \mathbb{R}^{2304}$ corresponds to:
   $$V_k = \sum_{x=1}^{H} \sum_{y=1}^{W} A_k(x,y) \cdot F(x,y)$$
   Each regional vector is unit-normalized: $\hat{V}_k = \frac{V_k}{\max(\|V_k\|_2, \epsilon)}$, yielding a parts-based descriptor where each head isolates forensic evidence from distinct spatial regions.

* **Manuscript Changes:** Section 3.3.3 (Equations 3–6) and Figure 1 have been revised with the complete matrix formulation and dimensionality flow.

---

### Major Concern 3: CNN-LSTM Feature Pipeline & Dimensionalities
> **Reviewer Comment:**  
> *"The transition between spatial multi-head features and the LSTM temporal model lacks mathematical precision. How are the $M$ regional feature vectors aggregated into the frame-level feature vector $x_t$? What are the exact dimensions at each stage?"*

**Author Response:**  
We have expanded **Section 3.3.4 (Equations 7–9)** and added **Table 1 (Tensor Dimension Trace Table)** to define every dimensional transition across the entire network:

1. **Aggregation and Linear Bottleneck Projection ($W_{proj}$):**  
   For frame $t \in \{1, \dots, T\}$, the $M=4$ normalized regional vectors $\{\hat{V}_1^{(t)}, \dots, \hat{V}_M^{(t)}\} \subset \mathbb{R}^{2304}$ are concatenated into a single frame descriptor $V_{concat}^{(t)} \in \mathbb{R}^{9216}$ ($4 \times 2304 = 9216$). To prevent parameter explosion in the recurrent cell, $V_{concat}^{(t)}$ is projected to $d_{in}=512$ via a learned linear projection $W_{proj} \in \mathbb{R}^{512 \times 9216}$ with Layer Normalization and GELU activation:
   $$x_t = \text{GELU}\Big(\text{LayerNorm}(W_{proj} V_{concat}^{(t)} + b_{proj})\Big) \in \mathbb{R}^{512}$$

2. **Recurrent Sequence Modeling & MLP Classifier:**  
   The sequence $\{x_1, \dots, x_T\} \in \mathbb{R}^{T \times 512}$ ($T=20$) is processed by a 2-layer Bidirectional LSTM with hidden dimension $d_h = 256$ per direction ($h_t \in \mathbb{R}^{512}$). Temporal average pooling yields $h_{seq} = \frac{1}{T}\sum_{t=1}^T h_t \in \mathbb{R}^{512}$, which is classified by an MLP ($512 \to 128 \to 2$) into binary logits $\hat{y} \in \mathbb{R}^2$.

3. **Full Tensor Dimension Trace Table (Table 1):**

| Pipeline Stage | Input Shape | Output Shape | Details / Layer Specification |
| :--- | :--- | :--- | :--- |
| **Input Video Segment** | $(B, T, 3, 256, 256)$ | $(B \cdot T, 3, 256, 256)$ | $B=16, T=20$ frames |
| **ResNeXt50 Stage 4** | $(B \cdot T, 3, 256, 256)$ | $(B \cdot T, 2048, 8, 8)$ | Semantic map $F_{sem}$ |
| **TEB Texture Block** | $(B \cdot T, 64, 64, 64)$ | $(B \cdot T, 256, 8, 8)$ | Texture map $F_{tex}$ |
| **Feature Fusion** | $F_{sem}, F_{tex}$ | $(B \cdot T, 2304, 8, 8)$ | Fused map $F$ ($C=2304$) |
| **Multi-Head Attention** | $(B \cdot T, 2304, 8, 8)$ | $(B \cdot T, 4, 8, 8)$ | $M=4$ attention maps $A_k$ |
| **BAP Pooling** | $F$ and $A$ | $(B \cdot T, 4, 2304)$ | $M=4$ regional vectors $\hat{V}_k$ |
| **Bottleneck Projection** | $(B \cdot T, 9216)$ | $(B \cdot T, 512)$ | Linear $W_{proj} \in \mathbb{R}^{512 \times 9216}$ |
| **Temporal Unfold** | $(B \cdot T, 512)$ | $(B, 20, 512)$ | Temporal sequence $x_t$ |
| **2-Layer BiLSTM** | $(B, 20, 512)$ | $(B, 20, 512)$ | Hidden $d_h = 256 \times 2 = 512$ |
| **Temporal Mean Pool** | $(B, 20, 512)$ | $(B, 512)$ | Global descriptor $h_{seq}$ |
| **MLP Classifier** | $(B, 512)$ | $(B, 2)$ | Binary logits (Real / Fake) |

* **Manuscript Changes:** Section 3.3.4, Equations 7–9, and Table 1 provide the full mathematical and tensor derivation.

---

### Major Concern 4: DF40 Evaluation Protocol & Cross-Category Generalization
> **Reviewer Comment:**  
> *"The evaluation on the DF40 benchmark lacks clear experimental distinction between cross-category testing (evaluating on unseen manipulation types) and standard within-domain cross-validation. How is 5-fold CV applied to unseen generator benchmarks?"*

**Author Response:**  
We thank the Reviewer for pointing out the need for an unambiguous description of the cross-category protocol. We have added **Section 3.2.3 and Section 4.4** to formally document the protocol:

* **Training Stage:** 5-fold cross-validation is performed *within* a single manipulation family (e.g., Train on Face Swapping [FS], encompassing 10 generator methods). The source identities are partitioned into 5 subject-disjoint folds, yielding 5 distinct trained model checkpoints.
* **Unseen Generalization Testing:** Each of the 5 trained model checkpoints is independently evaluated on the remaining manipulation categories (Test on Face Reenactment [FR], Entire Face Synthesis [EFS], and Face Editing [FE]).
* **Strict Unseen Definition:** An "unseen category" represents a manipulation paradigm from which **zero training images, zero video clips, and zero checkpoint weights** were accessible during training.
* **Reported Metrics:** In Table 4 (Section 4.4.1), each cell reports the $\text{Mean} \pm \text{Standard Deviation}$ across all 5 evaluation checkpoints.

* **Manuscript Changes:** Section 3.2.3 and Section 4.4.1 (Table 4) now explicitly define this protocol.

---

### Major Concern 5: Sequence Modeling on Static Image Categories (EFS and FE)
> **Reviewer Comment:**  
> *"The manuscript applies an LSTM temporal model to Entire Face Synthesis (EFS) and Face Editing (FE) categories, which consist of static images rather than video sequences. The justification for using temporal modeling on static data is unconvincing."*

**Author Response:**  
We appreciate the Reviewer's insightful observation. We have added **Section 3.2.4 and Section 4.4.2** to provide both theoretical and empirical justifications:

1. **Mathematical Frame Replication & Invariant Fixed-Point Convergence:**  
   To maintain a unified network topology for both video and static image inputs without heuristic architecture switching, each static image $I$ is expanded into a length-$T=20$ sequence via frame replication: $I_1 = I_2 = \dots = I_T = I$. Because the extracted spatial features are time-invariant ($x_t = x$), the BiLSTM recurrent state rapidly reaches a stable fixed-point representation:
   $$h_T = \text{BiLSTM}\big(x, h_{T-1}\big) \approx \phi(x)$$

2. **Modality-Specific Empirical Dissection (Section 4.4.2):**  
   We explicitly dissect the source of performance across modalities:
   * **Static Image Categories (EFS, FE):** The high performance ($83.2\%$ AUC on EFS) is driven primarily by **Phase 2 (TEB + Multi-Attentional BAP with $\mathcal{L}_{RIL}$)**, which detects microscopic high-frequency generative upsampling noise and localized facial inconsistencies.
   * **Video Categories (FS, FR):** The **BiLSTM sequence module** delivers an essential complementary boost ($+2.58\%$ AUC on Celeb-DF, Table 2) by capturing inter-frame jitter, boundary flickering, and irregular blinking dynamics.

* **Manuscript Changes:** Section 3.2.4 and Section 4.4.2 explicitly document this modality separation and fixed-point analysis.

---

### Major Concern 6: Data Leakage Prevention & Partitioning Level
> **Reviewer Comment:**  
> *"The manuscript does not clarify whether the 5-fold cross-validation is subject-disjoint or video-disjoint. Splitting at the frame or clip level causes severe data leakage."*

**Author Response:**  
We completely agree with the Reviewer that frame-level splitting introduces artificial inflation. We have added **Section 3.2.2** to explicitly guarantee that our evaluation is strictly **leakage-free**:

* **Subject-Disjoint Partitioning ($0\%$ Overlap):** In FaceForensics++ (1,000 source videos), DF40, and Celeb-DF (590 celebrity identities), partitioning is executed strictly at the **source video and subject identity level** prior to frame extraction. All manipulated variations derived from subject identity $k$ are assigned exclusively to Fold $k$. Zero frames or videos of any subject in the test set ever appear in the training set.
* **Sequence Encapsulation:** Temporal clips of length $T=20$ frames are extracted strictly within individual video boundaries post-partitioning. Sliding windows never cross video boundaries.

* **Manuscript Changes:** Section 3.2.2 and Section 4.1 explicitly document the strict subject-disjoint partitioning protocol.

---

### Major Concern 7: Statistical Analysis & Multiple Comparisons Corrections
> **Reviewer Comment:**  
> *"The statistical comparison lacks paired tests across cross-validation folds and does not account for the multiple testing problem when evaluating across multiple forgery categories."*

**Author Response:**  
We thank the Reviewer for emphasizing statistical rigor. We have added **Section 4.4.3 and Table 5**, conducting two-tailed paired $t$-tests across the 5 cross-validation folds ($df = 4$) with family-wise error rate (FWER) and false discovery rate (FDR) corrections:

1. **Multiple Comparisons Adjustments across $K=8$ Evaluation Conditions:**
   * **Bonferroni Correction:** Sets the conservative adjusted threshold to $\alpha_{adj} = \frac{0.05}{8} = 0.00625$.
   * **Benjamini-Hochberg (BH) FDR:** Controls false discovery rate at $q < 0.05$.

2. **Statistical Significance Table (Table 5):**

| Evaluation Condition | Mean Diff ($\Delta$) | $t$-statistic | Raw $p$-value | Bonferroni $p_{adj}$ | BH FDR ($q<0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train FS $\to$ Test FS** | $+1.1\%$ | $4.568$ | $0.0103$ | $0.0824$ | Significant ($q = 0.0103$) |
| **Train FS $\to$ Test FR** | $+4.2\%$ | $11.611$ | $< 0.0001$ | $< 0.0008$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FS $\to$ Test EFS** | $+3.4\%$ | $7.694$ | $0.0015$ | $0.0120$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FS $\to$ Test FE** | $+2.7\%$ | $8.409$ | $0.0011$ | $0.0088$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FR $\to$ Test FS** | $+7.7\%$ | $17.424$ | $< 0.0001$ | $< 0.0008$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FR $\to$ Test FR** | $+2.1\%$ | $7.457$ | $0.0017$ | $0.0136$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FR $\to$ Test EFS** | $+2.3\%$ | $6.358$ | $0.0031$ | $0.0248$ | **Significant** ($p < \alpha_{adj}$) |
| **Train FR $\to$ Test FE** | $+5.1\%$ | $14.099$ | $< 0.0001$ | $< 0.0008$ | **Significant** ($p < \alpha_{adj}$) |

3. **Statistical Assumption Verification:**
   * **Normality:** Shapiro-Wilk tests yielded $W \in [0.892, 0.967]$ with all $p > 0.30$, verifying normality.
   * **Non-Parametric Confirmation:** Wilcoxon signed-rank tests yielded $W=0, p=0.03125$.
   * **Conclusion:** All 8 cross-forgery conditions reject the null hypothesis under Benjamini-Hochberg FDR control ($q < 0.05$), and 6 of 8 remain significant under the strict Bonferroni threshold ($\alpha_{adj} = 0.00625$).

* **Manuscript Changes:** Section 4.4.3 and Table 5 document the full statistical analysis.

---

## Conclusion

We thank the Reviewers and the Editor once again for their constructive and invaluable critique. The revisions have significantly strengthened the theoretical foundations, empirical clarity, and statistical rigor of our manuscript. We hope the revised paper is now suitable for publication.
