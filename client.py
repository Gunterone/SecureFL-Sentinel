#!/usr/bin/env python3
from __future__ import annotations
import pickle
import json
import secrets
import hashlib
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from helper import load_and_preprocess_csv
from attacks import create_attack, BaseAttack
from zkp_common import zkp_client_utils as client_zkp 
from zkp_common import zkp_common as zk_common


# Util
def pack_params(W: np.ndarray, b: float):
    return [W.astype(np.float32), np.array([b], dtype=np.float32)]

def pack_payload(obj: dict) -> np.ndarray:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(b, dtype=np.uint8)

def _hash_trees(lst):
    """Calcola hash cumulativo deterministico degli alberi serializzati."""
    h = hashlib.sha256()
    for tb in lst:
        h.update(hashlib.sha256(tb).digest())
    return h.hexdigest()

class RFBaggingClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        *,
        label_col: str = "marker",
        apply_smote: bool = False,
        smote_sampling_strategy: str | float | dict = "auto",
        smote_k_neighbors: int = 5,
        coerce_numeric: bool = True,
        cast_float32: bool = True,
        seed: int = 42,
        # Attack cfg 
        attack_mode: str = "none",
        poison_frac: float = 0.15,
        trigger_delta: float = 8.0,
        trigger_cols: list[int] | None = None,
        target_class: int = 1,
        random_model_depth: int = 6,
        label_flip_mapping: dict[int, int] | None = None,
        is_malicious: bool = False,
        n_features: int = 20,
    ):
        self.client_id = client_id
        self.seed = seed
        self.is_malicious = is_malicious
        self.attack_mode = attack_mode.lower()
        self.n_features = n_features

        # ZKP init
        self.zkid = client_zkp.load_or_create_zkid(client_id)
        self.zkp_sk, self.zkp_pk = client_zkp.load_or_create_keypair(client_id)

        # Data preprocessing
        data_csv = f"data/{client_id}/data_{client_id}.csv"
        self.X, self.y, _ = load_and_preprocess_csv(
            data_csv,
            label_col=label_col,
            apply_smote=apply_smote,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            coerce_numeric=coerce_numeric,
            cast_float32=cast_float32,
            random_state=seed,
        )
        n = len(self.y)
        self.sample_weight = np.ones(n, dtype=float)

        # model
        self.rf = RandomForestClassifier(
            n_estimators=0,
            warm_start=True,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=seed,
            max_features="sqrt",
        )
        self.trees_built = 0
        self._last_signed_msg_hex: str | None = None

        # attack
        self.attack: BaseAttack | None = create_attack(
            self.attack_mode,
            seed=seed,
            poison_frac=poison_frac,
            trigger_delta=trigger_delta,
            trigger_cols=trigger_cols,
            target_class=target_class,
            random_model_depth=random_model_depth,
            label_flip_mapping=label_flip_mapping,
        )


    def get_properties(self, config):
        cfg = config or {}
        if cfg.get("zk_request", False):
            out = {
                "zk_client_id": self.zkid,
                "zk_pubkey_int": str(int(self.zkp_pk)), 
            }
            # opzionale: pre-auth
            if cfg.get("zk_phase") == "auth":
                rnd = int(cfg.get("zk_round", -1))
                nonce_hex = str(cfg.get("zk_challenge", ""))
                if rnd >= 0 and nonce_hex:
                    msg = (nonce_hex + str(rnd)).encode()
                    sig = client_zkp.schnorr_sign(self.zkp_sk, msg)
                    out["zk_proof_R"] = str(sig[0])
                    out["zk_proof_s"] = str(sig[1])
            return out
        return {}

    def get_parameters(self, config):
        return pack_params(np.zeros((self.n_features,), dtype=np.float32), 0.0)

    def fit(self, parameters, config):
        K = int(config.get("trees_per_round", 25))
        oob_lambda = float(config.get("oob_lambda", 1.0))
        ema = float(config.get("oob_ema", 0.9))
        round_id = int(config.get("zk_round", 0)) if config else 0

        # Verifica ZKP round precedente
        if config and "zkp_verify_payload" in config:
            try:
                z = json.loads(config["zkp_verify_payload"])
                transcript = z["transcript"]
                accepted_hex = z["accepted_msgs"]
                sb = float(z.get("sentinel_before", 0.0))
                sa = float(z.get("sentinel_after", 0.0))
                tau = float(z.get("sentinel_tau", 0.0))

                # Verifica firma transcript
                core = {k: transcript[k] for k in ["round_seed","merkle_root","commitments","proofs","delta_q","scale"]}
                core_msg = json.dumps(core, separators=(",", ":"), sort_keys=True).encode()
                agg_pk = int(transcript["aggregator_pk"])
                sig = tuple(transcript["aggregator_sig"])
                ok_sig = client_zkp.schnorr_verify(agg_pk, core_msg, sig)

                # Merkle root e inclusione del proprio msg
                accepted_msgs = [bytes.fromhex(h) for h in accepted_hex]
                merkle_ok = (zk_common.merkle_root(accepted_msgs).hex() == transcript["merkle_root"])
                inclusion_ok = True
                if self._last_signed_msg_hex is not None:
                    inclusion_ok = (self._last_signed_msg_hex in accepted_hex)

                # Verifica prove di apertura Pedersen
                C_list = transcript["commitments"]
                pr_list = transcript["proofs"]
                rseed = bytes.fromhex(transcript["round_seed"])
                updates = [json.loads(m.decode())["update"] for m in accepted_msgs]
                dim = len(updates[0]) if updates else 0
                m_sum = 0
                for k in range(dim):
                    m_sum += sum(u[k] for u in updates)
                pedersen_ok = True
                for j, Cj in enumerate(C_list):
                    rj = zk_common.deterministic_blind(rseed, "r", j)
                    if not zk_common.schnorr_verify_opening(Cj, m_sum, tuple(pr_list[j])):
                        pedersen_ok = False
                        break

                # 4) Ricomposizione Δq
                delta_calc = []
                if updates:
                    n = len(updates)
                    for k in range(dim):
                        s = sum(u[k] for u in updates)
                        delta_calc.append(int(s // n))
                delta_ok = (delta_calc == transcript["delta_q"])

                # 5) Verifica sentinella
                sentinel_ok = ((sb - sa) <= tau)

                all_ok = (ok_sig and merkle_ok and inclusion_ok and pedersen_ok and delta_ok and sentinel_ok)
                print(f"[{self.client_id}] Round {round_id} verifica transcript")
                print(f" - transcript signature {'ok' if ok_sig else 'FAIL'}")
                print(f" - merkle/inclusion {'ok' if (merkle_ok and inclusion_ok) else 'FAIL'}")
                print(f" - commitments opening {'ok' if pedersen_ok else 'FAIL'}")
                print(f" - Δq match {'ok' if delta_ok else 'FAIL'}")
                print(f" - sentinel metric {'ok' if sentinel_ok else 'FAIL'}")

            except Exception as e:
                print(f"[{self.client_id}] Round {round_id} verifica transcript: EXCEPTION {e}")


        # Train modello locale
        self.rf.n_estimators += K
        self.rf.fit(self.X, self.y, sample_weight=self.sample_weight)
        self.trees_built += K

        # OOB re-weighting
        if hasattr(self.rf, "oob_decision_function_") and self.rf.oob_decision_function_ is not None:
            proba = self.rf.oob_decision_function_
            y_pred = np.argmax(proba, axis=1)
            err = (y_pred != self.y).astype(float)
            new_w = 1.0 + oob_lambda * err
            self.sample_weight = ema * self.sample_weight + (1 - ema) * new_w
            self.sample_weight /= (self.sample_weight.mean() + 1e-12)

        # Attacco + update
        if self.is_malicious and self.attack is not None:
            new_estimators = self.attack.produce_trees(self.X, self.y, K)
            attack_used = self.attack_mode
            print(f"[{self.client_id}] Attacco attivo: {attack_used}")
        else:
            new_estimators = self.rf.estimators_[-K:]
            attack_used = "none"

        # Genera un vettore di update basato sui nuovi alberi
        local_weights = []
        for tree in new_estimators:
            fi = getattr(tree, "_fake_feature_importances_", None)
            if fi is None and hasattr(tree, "feature_importances_"):
                fi = tree.feature_importances_
            if fi is not None:
                local_weights.extend(fi)
        if not local_weights:
            local_weights = np.mean(self.X, axis=0).tolist()

        local_weights = np.array(local_weights, dtype=float)
        local_weights -= np.mean(local_weights)
        local_weights += np.random.normal(0, 0.02, size=local_weights.shape)

        if self.is_malicious:
            drift_factor = 1.0 + 0.25 * round_id
            local_weights *= drift_factor
            print(f"[{self.client_id}] Attacco attivo: drift_factor={drift_factor:.2f}")

        # Quantizzazione per ZKP audit
        scale = 1000
        q_update = client_zkp.quantize_vector(local_weights.tolist(), scale=scale)

        # Firma ZKP + payload
        trees_bytes = [pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL) for t in new_estimators]
        trees_hash = _hash_trees(trees_bytes)

        signed_payload = client_zkp.produce_signed_update(
            client_id=self.client_id,
            sk=self.zkp_sk,
            q_update=q_update,
            scale=scale,
            round_id=round_id,
            extra_data=trees_hash.encode(),  #hash alberi inclusi nella firma
        )

        try:
            self._last_signed_msg_hex = signed_payload["msg"].hex()
        except Exception:
            self._last_signed_msg_hex = None

        payload = {
            "client_id": self.client_id,
            "n_samples": len(self.y),
            "trees": trees_bytes,
            "attack_mode": self.attack_mode,
            "is_malicious": bool(self.is_malicious),
        }

        metrics = {
            "oob_score": float(getattr(self.rf, "oob_score_", np.nan)),
            "trees_sent": K,
            "total_trees_local": self.trees_built,
            "attack_mode": self.attack_mode,
            "is_malicious": int(self.is_malicious),

            # ZKP + audit
            "zk_client_id": self.zkid,
            "zk_pubkey_int": str(int(self.zkp_pk)),
            "zk_signature": json.dumps({"R": signed_payload["sig"][0], "s": signed_payload["sig"][1]}),
            "q_update": json.dumps(q_update),
            "scale": scale,
            "client_id": self.client_id,
            "zk_round": int(round_id),
            "trees_hash": trees_hash,  #hash incluso per verifica lato server
        }

        arr = pack_payload(payload)
        return [arr], len(self.y), metrics

    def evaluate(self, parameters, config):
        return 0.0, len(self.y), {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8081")
    parser.add_argument("--client_id", required=True)
    parser.add_argument("--is_malicious", action="store_true")
    parser.add_argument("--attack_mode", type=str, default="none",
                        choices=["none", "sign_flip", "byzantine", "stealthy_poison"])
    parser.add_argument("--poison_frac", type=float, default=0.15)
    parser.add_argument("--trigger_delta", type=float, default=8.0)
    parser.add_argument("--trigger_cols", type=str, default="")
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--random_model_depth", type=int, default=6)

    args = parser.parse_args()
    trigger_cols = [int(t) for t in args.trigger_cols.split(",") if t.strip().isdigit()] if args.trigger_cols else None

    client = RFBaggingClient(
        client_id=args.client_id,
        apply_smote=False,
        smote_sampling_strategy="auto",
        smote_k_neighbors=5,
        attack_mode=args.attack_mode,
        poison_frac=args.poison_frac,
        trigger_delta=args.trigger_delta,
        trigger_cols=trigger_cols,
        target_class=args.target_class,
        random_model_depth=args.random_model_depth,
        is_malicious=bool(args.is_malicious),
    )

    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
