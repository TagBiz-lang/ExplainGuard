# ExplainGuard
Cryptographically verifiable activation fingerprints for ML explainability in network intrusion detection


ExplainGuard is a framework that cryptographically binds per-feature activation 
fingerprints to their ML inference event at the moment of generation, providing 
tamper evidence and non-repudiation for explainable AI pipelines in 
security critical systems.

Built for network intrusion detection, ExplainGuard captures activation magnitudes 
during the forward pass, chains them with the input and inference metadata via 
SHA-256, and issues an ECDSA-signed certificate (NIST P-256)  without requiring 
model re-access at verification time.

Evaluated on CICDDoS2019, CICIDS2017, BCC-Cpacket, and CIC-IoT2023 across 
Transformer and CNN-LSTM architectures. Achieves 94% empirical attack detection, 
1.0 ms mean certification latency, and 100% hash-chain integrity across 431,731 
issued certificates.

This repository contains the code, evaluation scripts, and datasets used in the 
paper: "ExplainGuard: Cryptographically Verifiable Explanations for Network 
Intrusion Detection" (ACM CCS 2026 (on review)).
