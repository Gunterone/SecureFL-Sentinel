# SecureFL-Sentinel
Secure Federated Learning Framework with Cryptographic Verification.

This project implements a secure Federated Learning framework based on Random Forest, designed to improve robustness against malicious participants while preserving data privacy.

The system combines cryptographic primitives and statistical defenses to ensure the integrity and auditability of model updates, including:
  - L2 clipping to limit the impact of anomalous updates
  - Pedersen commitments and Merkle trees to guarantee authenticity and immutability
  - Zero-Knowledge Proofs (ZKP) to verify correctness without revealing local data

A sentinel module evaluates each client update on a public validation dataset and filters out contributions that degrade the macro-F1 score in real time.

The framework was tested against multiple adversarial scenarios, including Sign-Flip, Byzantine, and stealthy data poisoning attacks, demonstrating strong resilience with limited computational overhead and no external infrastructure requirements.
