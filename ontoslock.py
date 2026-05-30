import numpy as np
from scipy.spatial import KDTree

class AdaptiveOntologicalLock:
    """
    OntosLock: A high-performance, non-invasive cognitive control architecture.
    Aligns latent activations to a reference manifold using cubic damping.
    """
    def __init__(self, reference_matrix, base_epsilon=0.05, smoothing=0.1, k=3, temperature=0.1):
        """
        Initialize the lock with a reference matrix of 'safe' latent states.
        """
        self.drm = self._normalize(np.array(reference_matrix))
        self.tree = KDTree(self.drm)
        self.base_epsilon = base_epsilon
        self.smoothing = smoothing
        self.k = k
        self.temperature = temperature

    def _normalize(self, x):
        """Helper to normalize vectors to unit length."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-9)

    def get_dynamic_target(self, h_curr):
        """Finds the optimal safe target in the DRM using k-NN."""
        h_norm = self._normalize(h_curr.reshape(1, -1)).flatten()
        dists, idxs = self.tree.query(h_norm, k=self.k)
        
        # Calculate RBF weights
        sigma = np.mean(dists) + 1e-9
        weights = np.exp(-(dists**2) / (2 * (sigma * self.temperature)**2))
        weights /= (weights.sum() + 1e-9)
        
        return np.average(self.drm[idxs], axis=0, weights=weights)

    def apply_lock(self, h_curr):
        """
        Main inference function: Applies cubic damping to drift inputs.
        """
        h_norm = self._normalize(h_curr.reshape(1, -1)).flatten()
        h_target = self.get_dynamic_target(h_norm)
        
        dist = np.linalg.norm(h_norm - h_target)
        
        # Apply Cubic Damping if deviation exceeds epsilon
        if dist > self.base_epsilon:
            raw_ratio = ((dist - self.base_epsilon)**3) / (dist + 1e-9)
            ratio = np.clip(raw_ratio, 0, 1) 
            h_corrected = h_norm - ratio * (h_norm - h_target)
            
            # Apply EMA smoothing for stability
            h_final = (1 - self.smoothing) * h_norm + self.smoothing * h_corrected
            return h_final * np.linalg.norm(h_curr)
        
        return h_curr
          
