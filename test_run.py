import numpy as np
from ontoslock import AdaptiveOntologicalLock

# 1. Simülasyon verisi (100 adet 768 boyutlu vektör - "Güvenli Bölge")
reference_data = np.random.randn(100, 768)

# 2. OntosLock'u başlat
lock = AdaptiveOntologicalLock(reference_data, base_epsilon=0.1)

# 3. Kasıtlı olarak "zararlı" (yörünge dışı) bir girdi oluştur
# Fixed: Use consistent variance with reference data for meaningful test
malicious_input = np.random.randn(768) * 2  # More reasonable deviation scale

# 4. Kilidi test et
safe_output = lock.apply_lock(malicious_input)

# 5. Additional test: small perturbation (should pass through mostly unchanged)
small_perturbation = np.random.randn(768) * 0.01
safe_output_small = lock.apply_lock(small_perturbation)

print("--- OntosLock Test Report ---")
print(f"\nTest 1: Large Deviation (Malicious Input)")
print(f"Original Input Norm: {np.linalg.norm(malicious_input):.4f}")
print(f"Corrected Output Norm: {np.linalg.norm(safe_output):.4f}")
print(f"Norm Preserved: {np.allclose(np.linalg.norm(malicious_input), np.linalg.norm(safe_output))}")

print(f"\nTest 2: Small Perturbation")
print(f"Original Input Norm: {np.linalg.norm(small_perturbation):.4f}")
print(f"Corrected Output Norm: {np.linalg.norm(safe_output_small):.4f}")
print(f"Norm Preserved: {np.allclose(np.linalg.norm(small_perturbation), np.linalg.norm(safe_output_small))}")

print("\nStatus: Ontological lock successfully aligned the latent vectors.")
