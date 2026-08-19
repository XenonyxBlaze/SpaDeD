Review Report
Manuscript ID:1637
Recommendation: Major Revision
This manuscript addresses an important problem in deepfake detection and
proposes a promising combination of ResNeXt50, texture enhancement, multi-
attentional feature extraction, and LSTM-based temporal modeling. The overall
research direction is relevant; however, several methodological issues need to be
resolved before the experimental claims can be considered reliable.
Major Concerns:
1. The formulation of the Regional Independence Loss (LRIL) is questionable.
The manuscript claims that LRIL penalizes overlapping attention maps, but the
proposed loss is actually computed from the cosine similarity between regional
feature vectors (Vi) and (Vj). This does not directly measure spatial overlap between
attention maps. Therefore, the claim that the loss “forces each attention head to
cover a distinct facial region” is not sufficiently supported by the presented
formulation.
2. The definition of Bilinear Attention Pooling (BAP) is unclear.
The mathematical formulation in Eq. (4) appears to be an attention-weighted feature
aggregation rather than a clearly defined bilinear pooling operation. The manuscript
should clarify the exact BAP implementation and how texture and semantic features
are combined before entering the LSTM.
3. The CNN–LSTM feature pipeline is insufficiently specified.
The manuscript does not clearly explain how the outputs of multiple attention heads
are fused into the temporal feature (xt). Consequently, the dimensionality and exact
input representation of the LSTM cannot be reproduced from the current description.
4. The DF40 evaluation protocol requires clarification.
The manuscript describes five-fold cross-validation while also claiming cross-forgery
evaluation on DF40. The exact training/testing split at the manipulation-method level
is not sufficiently specified. In particular, the authors should clarify what constitutes
an “unseen” generator or manipulation method.
5. The use of an LSTM for image-based DF40 categories is unclear.
Since the proposed framework requires a sequence of frames, the manuscript should
explicitly explain how image-based EFS/FE samples are converted into temporal
inputs. Otherwise, the claimed spatiotemporal advantage in these cross-category
experiments is difficult to interpret.
6. Potential data leakage is not adequately addressed.
The manuscript reports five-fold cross-validation but does not clearly state whether
the split is performed at the video/source/identity level before frame and sequence
generation. This is critical for preventing highly similar frames from the same video
from appearing in both training and validation sets.
7. The statistical analysis should be reconsidered.
The paper uses paired t-tests based on five folds and reports multiple significance
tests. The statistical justification and treatment of multiple comparisons should be
clarified before the claims of statistical significance are accepted.
Recommendation:
The manuscript has a potentially valuable research idea, but the above issues
concern the core methodology and experimental validity, rather than merely
presentation. I therefore recommend Major Revision. The authors should first
clarify/correct the LRIL and BAP formulations, fully specify the CNN–LSTM feature
flow and data-splitting protocol, and clarify the DF40 temporal evaluation procedure.
After these issues are addressed, the reported performance improvements can be
assessed more reliably.