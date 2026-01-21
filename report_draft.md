## I. Introduction

Room Impulse Responses (RIRs) play a crucial role in spatial audio rendering, auralization, and speech processing applications. Energy Decay Curves (EDCs) are widely used to characterize reverberation behavior and estimate perceptually relevant acoustic parameters such as reverberation time and clarity.

Recent work has demonstrated that Long Short-Term Memory (LSTM) networks can accurately predict full-band EDCs directly from room geometric and material properties. However, human perception of reverberation is frequency-dependent, and modeling all frequencies jointly may overlook perceptually important spectral characteristics.

In this project, the baseline LSTM-based EDC prediction framework is extended to a multi-band formulation. By predicting frequency-dependent EDCs, the proposed approach aims to improve perceptual quality metrics such as clarity and spectral envelope similarity in reconstructed RIRs.
