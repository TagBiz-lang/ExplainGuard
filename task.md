# Task: Address Reviewer Comments

## 1. Fix Explanation Method
- [ ] Add `SHAPExplainer` class to `explainer.py`
- [ ] Add `ExplainerFactory` to `explainer.py`
- [ ] Remove duplicate `IntrinsicExplainer` from `explainers.py`
- [ ] Update config with explainer type settings

## 2. Fix Fake RQ Evaluations
- [ ] Rewrite `evaluate_rq1_integrity()` with real tampering tests
- [ ] Remove duplicate `evaluate_rq2_adversarial()`, implement real adversarial testing
- [ ] Rewrite `evaluate_rq3_overhead()` with real timing measurements
- [ ] Update `generate_certificates()` to pass needed data for real RQ evaluation

## 3. Strengthen Certificate Verification
- [ ] Add `tamper()` method to Certificate
- [ ] Add `verify_hash_chain()` method to Certificate
- [ ] Add timestamp and model version to certificate data

## 4. Multi-Domain Dataset Support
- [ ] Add domain field to dataset config
- [ ] Add domain-aware logging in dataset loader

## 5. Clean Up Code Duplication
- [ ] Consolidate `components.py` to import from canonical modules

## 6. Verification
- [ ] Run existing tests
- [ ] Verify imports work
- [ ] Check output JSON for real values
