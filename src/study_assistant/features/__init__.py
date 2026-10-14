"""Feature engineering and text transformation modules.

CRITICAL POLICY:
- rationale and mark_scheme must NEVER be used as features (target leakage barrier).
- Vectorizers, scalers, and encoders must only be fitted on training splits.
"""
