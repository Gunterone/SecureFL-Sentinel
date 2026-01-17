# attacks.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Dict, Any
import math
import copy 

class BaseAttack(ABC):
    """
    Interfaccia base per gli attacchi.
    """
    def __init__(self, seed: int = 42, **kwargs):
        self.rng = np.random.RandomState(seed)
        self.params = kwargs

    @abstractmethod
    def produce_trees(self, X: np.ndarray, y: np.ndarray, K: int):
        raise NotImplementedError

def _train_tree(X, y, max_depth: Optional[int], random_state: int):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    return tree

def _sample_indices(rng: np.random.RandomState, n, frac):
    cnt = max(1, int(math.ceil(frac * n)))
    return rng.choice(n, size=cnt, replace=False)


# Sign-Flip attack
class SignFlipAttack(BaseAttack):
    """
    - Allena K alberi normalmente.
    - Calcola feature_importances_ per ogni albero.
    - Per una frazione flip_frac degli alberi sovrascrive feature_importances_
      con il valore invertito e scalato: new_fi = -signflip_scale * old_fi.
    Questo fa sì che il client, quando costruisce local_weights concatenando
    tree.feature_importances_, ottenga un vettore con segno invertito rispetto
    al comportamento benigno.
    """
    def __init__(self, seed=42, **kwargs):
        super().__init__(seed, **kwargs)
        self.signflip_scale = float(self.params.get("signflip_scale", 1.0))
        self.flip_frac = float(self.params.get("flip_frac", 1.0))
        self.max_depth = self.params.get("random_model_depth", None)

    def produce_trees(self, X, y, K):
        trees = []
        # Allenamento k alberi normale
        for i in range(K):
            rs = self.rng.randint(0, 2**31 - 1)
            t = _train_tree(X, y, max_depth=self.max_depth, random_state=rs)
            trees.append(t)

        # numero alberi da flippare
        n_flip = max(1, int(math.ceil(self.flip_frac * K)))
        flip_idx = set(self.rng.choice(range(K), size=n_flip, replace=False))

        # raccoglie feature_importances originali per diagnostica
        fi_list = []
        for t in trees:
            fi = getattr(t, "feature_importances_", None)
            if fi is None:
                # se non disponibile, proviamo a ricavare via predict su colonne (fallback)
                fi = np.zeros(X.shape[1], dtype=float)
            fi_list.append(np.array(fi, dtype=float))

        # applica flip/sign change su alberi selezionati
        flipped = 0
        # norma totale prima e dopo
        total_before = np.linalg.norm(np.concatenate(fi_list)) if fi_list else 0.0

        for idx in flip_idx:
            t = trees[idx]
            fi = fi_list[idx]
            # se fi all-zero: skip
            if fi.sum() == 0:
                # piccola importanza distribuita uniformemente
                fi = np.ones_like(fi, dtype=float) * (1.0 / max(1, len(fi)))
            # costruzione nuova feature_importances_ con flip e scala
            new_fi = - self.signflip_scale * fi
            try:
                setattr(t, "_fake_feature_importances_", np.asarray(new_fi, dtype=float))
                flipped += 1
                fi_list[idx] = new_fi
            except Exception as e:
                print(f"[SIGNFLIP] non ho potuto impostare feature_importances_ su tree idx={idx}: {e}")
                continue

        total_after = np.linalg.norm(np.concatenate(fi_list)) if fi_list else 0.0
        print(f"[SIGNFLIP] flipped {flipped}/{n_flip} trees; flip_frac={self.flip_frac}; signflip_scale={self.signflip_scale}")
        return trees




# Byzantine attack
class ByzantineAttack(BaseAttack):
    """
    Produce alberi "malformati" addestrati su rumore/etichette casuali.
    """
    def __init__(self, seed=42, **kwargs):
        super().__init__(seed, **kwargs)
        self.max_depth = self.params.get("random_model_depth", None)
        self.noise_size = int(self.params.get("noise_size", 50))
        self.mode = str(self.params.get("mode", "random_labels"))

    def produce_trees(self, X, y, K):
        trees = []
        n_features = X.shape[1]
        for i in range(K):
            rs = self.rng.randint(0, 2**31 - 1)
            if self.mode == "random_labels":
                # usa gli stessi X ma con etichette casuali
                rand_labels = self.rng.randint(np.min(y), np.max(y) + 1, size=len(y))
                t = _train_tree(X, rand_labels, max_depth=self.max_depth, random_state=rs)
            elif self.mode == "random_features":
                # genera dati completamente casuali
                Xn = self.rng.normal(size=(self.noise_size, n_features))
                yn = self.rng.randint(np.min(y), np.max(y) + 1, size=self.noise_size)
                t = _train_tree(Xn, yn, max_depth=self.max_depth, random_state=rs)
            else:
                # predice sempre una classe
                Xc = np.zeros((1, n_features))
                yc = np.array([self.rng.randint(np.min(y), np.max(y) + 1)])
                t = _train_tree(Xc, yc, max_depth=1, random_state=rs)
            trees.append(t)
        return trees


# Stealthy gradual poisoning
class StealthyGradualPoisoningAttack(BaseAttack):
    """
    Avvelenamento graduale e stealthy:
      - ogni round aumenta la frazione di campioni avvelenati (fino a poison_frac)
      - invia una miscela di alberi puliti + alcuni alberi avvelenati per ridurre
        la probabilità che lo server riconosca il client come malizioso.
    """
    def __init__(self, seed=42, **kwargs):
        super().__init__(seed, **kwargs)
        self.poison_frac = float(self.params.get("poison_frac", 0.15))
        self.steps_to_reach = int(self.params.get("steps_to_reach", 5))
        self.target_class = int(self.params.get("target_class", 1))
        self.malicious_share = float(self.params.get("malicious_share", 0.85))
        self.max_depth = self.params.get("random_model_depth", None)
        self.label_flip_mapping: Optional[Dict[int, int]] = self.params.get("label_flip_mapping", None)
        self.round_counter = 0

    def _compute_current_frac(self):
        # aumento la frazione
        r = min(self.round_counter + 1, self.steps_to_reach)
        return (r / self.steps_to_reach) * self.poison_frac

    def produce_trees(self, X, y, K):
        self.round_counter += 1
        n = len(y)
        curr_frac = self._compute_current_frac()
        n_poison = max(1, int(math.ceil(curr_frac * n)))

        # seleziona indici da avvelenare
        poison_idx = _sample_indices(self.rng, n, curr_frac)

        # crea dataset avvelenato
        y_poisoned = y.copy()
        if self.label_flip_mapping:
            # mapping
            for src, dst in self.label_flip_mapping.items():
                mask = (y_poisoned == src)
                y_poisoned[mask] = dst
        else:
            y_poisoned[poison_idx] = self.target_class

        trees = []
        n_malicious = int(math.ceil(self.malicious_share * K))
        n_clean = K - n_malicious

        # Addestra alberi puliti 
        for _ in range(n_clean):
            rs = self.rng.randint(0, 2**31 - 1)
            t = _train_tree(X, y, max_depth=None, random_state=rs)
            trees.append(t)

        # Addestra alberi maligni 
        for _ in range(n_malicious):
            rs = self.rng.randint(0, 2**31 - 1)
            t = _train_tree(X, y_poisoned, max_depth=self.max_depth, random_state=rs)
            trees.append(t)

        # Mischia ordine 
        self.rng.shuffle(trees)
        return trees

def create_attack(mode: str, *, seed: int = 42, **kwargs) -> Optional[BaseAttack]:
    if mode is None:
        return None
    m = str(mode).lower()
    common = {"seed": seed}
    common.update(kwargs)
    if m == "":
        return None
    if m == "sign_flip":
        return SignFlipAttack(**common)
    if m == "byzantine":
        return ByzantineAttack(**common)
    if m == "stealthy_poison":
        return StealthyGradualPoisoningAttack(**common)
    return None
