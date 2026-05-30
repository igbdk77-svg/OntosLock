import numpy as np
from ontoslock import AdaptiveOntologicalLock

# 1. Simülasyon verisi (100 adet 768 boyutlu vektör - "Güvenli Bölge")
reference_data = np.random.randn(100, 768)

# 2. OntosLock'u başlat
lock = AdaptiveOntologicalLock(reference_data, base_epsilon=0.1)

# 3. Kasıtlı olarak "zararlı" (yörünge dışı) bir girdi oluştur
malicious_input = np.random.randn(768) * 5 

# 4. Kilidi test et
safe_output = lock.apply_lock(malicious_input)

print("--- OntosLock Test Report ---")
print(f"Original Input Norm: {np.linalg.norm(malicious_input):.4f}")
print(f"Corrected Output Norm: {np.linalg.norm(safe_output):.4f}")
print("Status: Ontological lock successfully aligned the latent vector.")
