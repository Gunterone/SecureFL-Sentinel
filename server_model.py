# server_model.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import parameters_to_ndarrays, EvaluateIns, FitIns, Metrics, GetPropertiesIns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import warnings
import json
import secrets
import hashlib
from typing import Dict, Optional, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helper import load_and_preprocess_csv
from zkp_common.zkp_model_utils import ZKPRegistry, collect_client_pubkeys, save_server_json
from zkp_common.aggregator_audit import compute_commitments_and_proofs, build_transcript
from zkp_common import zkp_client_utils as client_zkp

warnings.filterwarnings("ignore", category=UserWarning, message=r"X does not have valid feature names")

GLOBAL_CLASSES = np.array([0, 1, 2, 3], dtype=int)
CLIENTS_DIR = Path("zkp_common/clients_zkp")
SERVER_JSON = Path("zkp_common/server_zkp/pubkeys.json")

# -------------------- utils --------------------
def _load_eval_csv(path: str, label_col: str = "marker"):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Manca la colonna label '{label_col}' in {path}")
    y = df[label_col].astype(int).to_numpy()
    Xdf = df.drop(columns=[label_col]).replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.select_dtypes(include=["number"]).copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    bad = [c for c in Xdf.columns if Xdf[c].isna().all()]
    if bad:
        Xdf = Xdf.drop(columns=bad)
    X = Xdf.astype(np.float32).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("X_eval contiene NaN/Inf dopo la pulizia.")
    return X, y


def _load_trigger_csv(path: str, label_col: str = "marker"):
    df = pd.read_csv(path)
    if label_col in df.columns:
        Xdf = df.drop(columns=[label_col]).replace([np.inf, -np.inf], np.nan)
    else:
        Xdf = df.replace([np.inf, -np.inf], np.nan)

    Xdf = Xdf.select_dtypes(include=["number"]).copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    bad = [c for c in Xdf.columns if Xdf[c].isna().all()]
    if bad:
        Xdf = Xdf.drop(columns=bad)

    X = Xdf.astype(np.float32).to_numpy()
    if X.size == 0:
        raise ValueError("Trigger CSV non contiene colonne numeriche valide dopo la pulizia.")
    if not np.isfinite(X).all():
        raise ValueError("X_trigger contiene NaN/Inf dopo la pulizia.")
    return X


def _parse_boost(s: str) -> np.ndarray:
    """Parsa stringa tipo '1:1.25,2:1.25,3:0.95' in un vettore di pesi per classe."""
    v = np.ones(len(GLOBAL_CLASSES), dtype=np.float32)
    if not s:
        return v
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, val = kv.split(":")
        v[int(k)] = float(val)
    return v


def _tpr_fpr_from_confusion(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
    return (
        float(np.nanmean(tpr)),
        float(np.nanmean(fpr)),
        {int(c): float(v) for c, v in zip(GLOBAL_CLASSES, tpr)},
        {int(c): float(v) for c, v in zip(GLOBAL_CLASSES, fpr)},
    )


def _save_confusion_and_report(cm: np.ndarray, y: np.ndarray, y_pred: np.ndarray, out_dir="data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_counts = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in GLOBAL_CLASSES],
        columns=[f"pred_{c}" for c in GLOBAL_CLASSES],
    )
    df_rowpct = (df_counts.div(df_counts.sum(axis=1).replace(0, 1), axis=0) * 100).round(2)
    cls_rep = classification_report(y, y_pred, labels=GLOBAL_CLASSES, zero_division=0, digits=4)

    (Path(out_dir) / "confusion_matrix_counts.csv").write_text(df_counts.to_csv(index=True))
    (Path(out_dir) / "confusion_matrix_rowpct.csv").write_text(df_rowpct.to_csv(index=True))
    with open(Path(out_dir) / "classification_report.txt", "w") as f:
        f.write(cls_rep)
    return df_counts, df_rowpct, cls_rep


def _save_roc_plot(proba: np.ndarray, y: np.ndarray, out_dir="data"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for j, cls in enumerate(GLOBAL_CLASSES):
        y_bin = (y == cls).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, proba[:, j])
            auc_val = roc_auc_score(y_bin, proba[:, j])
            plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={auc_val:.3f})")
        except Exception:
            continue

    try:
        auc_micro = roc_auc_score(y, proba, average="micro", multi_class="ovr")
        auc_macro = roc_auc_score(y, proba, average="macro", multi_class="ovr")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title(f"ROC Curves (micro={auc_micro:.3f}, macro={auc_macro:.3f})")
    except Exception:
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title("ROC Curves")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    out_path = Path(out_dir) / "roc_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ROC] salvata in {out_path}")


def _score_tree_macro_f1(estimator, X_eval, y_eval) -> float:
    try:
        if hasattr(estimator, "predict_proba"):
            y_pred = np.argmax(estimator.predict_proba(X_eval), axis=1)
        else:
            y_pred = estimator.predict(X_eval)
        return float(f1_score(y_eval, y_pred, average="macro"))
    except Exception:
        return 0.0


# -------------------- foresta globale --------------------
class GlobalForest:
    def __init__(self, estimators: list):
        self.trees = list(estimators)
        self.classes_ = GLOBAL_CLASSES.copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            return np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        acc = None
        for est in self.trees:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
            else:
                y_h = est.predict(X)
                p = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
                for j, c in enumerate(self.classes_):
                    p[:, j] = (y_h == c).astype(float)
            acc = p if acc is None else (acc + p)
        return acc / max(len(self.trees), 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# -------------------- strategia federata --------------------
class FederatedForestStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        X_eval,
        y_eval,
        trees_per_round: int,
        pool_max_trees: int,
        min_fit_clients: int,
        min_available_clients: int,
        boost_mode: str = "off",  # "off" | "fixed" | "adaptive"
        boost_vec: np.ndarray | None = None,
        target_tpr: float = 0.65,
        boost_k: float = 0.7,
        boost_cap: float = 1.35,
        trigger_X: np.ndarray | None = None,
        backdoor_target: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.trees_per_round = int(trees_per_round)
        self.pool_max_trees = int(pool_max_trees)
        self.required_min_fit = int(min_fit_clients)
        self.required_min_avail = int(min_available_clients)

        self.pool_trees: list = []

        # trigger/backdoor (opzionali)
        self.trigger_X = trigger_X
        self.backdoor_target = int(backdoor_target) if backdoor_target is not None else None

        # boosting
        self.boost_mode = boost_mode
        self.fixed_boost = boost_vec if boost_vec is not None else np.ones(len(GLOBAL_CLASSES), dtype=float)
        self.target_tpr = float(target_tpr)
        self.boost_k = float(boost_k)
        self.boost_cap = float(boost_cap)
        self.prev_tpr_by_class: dict[int, float] | None = None

        # soglia clipping per l2 degli update firmati
        self.clip_bound = float("inf")

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=self.required_min_fit,
            min_num_clients=self.required_min_avail,
        )
        cfg = {
            "server_round": server_round,
            "trees_per_round": self.trees_per_round,
        }
        fitins = fl.server.client_proxy.FitIns(parameters=parameters, config=cfg)
        return [(cp, fitins) for cp in clients]

    def _extract_trees_from_fitres(self, fitres) -> list:
        trees = []
        arr_list = parameters_to_ndarrays(fitres.parameters)
        if not arr_list:
            return trees
        blob = arr_list[0].tobytes()
        try:
            obj = pickle.loads(blob)
        except Exception:
            return trees

        if hasattr(obj, "estimators_"):
            trees.extend(obj.estimators_)
            return trees

        if isinstance(obj, dict) and "trees" in obj:
            for tb in obj.get("trees", []):
                try:
                    t = pickle.loads(tb)
                    trees.append(t)
                except Exception:
                    continue
        return trees

    def aggregate_fit(self, server_round, results, failures):
        new_trees = []
        for _, fit_res in results:
            new_trees.extend(self._extract_trees_from_fitres(fit_res))
        self.pool_trees.extend(new_trees)

        if len(self.pool_trees) > self.pool_max_trees:
            scores = [(_score_tree_macro_f1(t, self.X_eval, self.y_eval), i) for i, t in enumerate(self.pool_trees)]
            scores.sort(reverse=True)
            keep_idx = {i for _, i in scores[: self.pool_max_trees]}
            self.pool_trees = [t for i, t in enumerate(self.pool_trees) if i in keep_idx]

        return (results[0][1].parameters if results else None), {
            "n_trees_new": int(len(new_trees)),
            "n_trees_pool": int(len(self.pool_trees)),
        }

    def _boost_from_prev_tpr(self) -> np.ndarray:
        v = np.ones(len(GLOBAL_CLASSES), dtype=float)
        if not self.prev_tpr_by_class:
            return v
        for j, cls in enumerate(GLOBAL_CLASSES):
            tpr_c = float(self.prev_tpr_by_class.get(int(cls), 1.0))
            gap = max(0.0, self.target_tpr - tpr_c)
            v[j] = min(1.0 + self.boost_k * gap, self.boost_cap)
        total_excess = float(np.sum(np.maximum(v - 1.0, 0.0)))
        if total_excess > 0:
            j_dom = len(GLOBAL_CLASSES) - 1
            v[j_dom] = max(0.85, 1.0 - 0.15 * total_excess)
        return v

    def _choose_boost_vec(self) -> np.ndarray:
        if self.boost_mode == "off":
            return np.ones(len(GLOBAL_CLASSES), dtype=float)
        if self.boost_mode == "fixed":
            return self.fixed_boost.astype(float)
        if self.boost_mode == "adaptive":
            return self._boost_from_prev_tpr()
        return np.ones(len(GLOBAL_CLASSES), dtype=float)

    def evaluate(self, server_round, parameters):
        if not self.pool_trees:
            return 0.0, {"note": "no_trees_yet"}

        global_forest = GlobalForest(self.pool_trees)
        proba_raw = global_forest.predict_proba(self.X_eval)

        boost_vec = self._choose_boost_vec().reshape(1, -1)
        proba = proba_raw * boost_vec

        y_pred = np.argmax(proba, axis=1)
        acc = float(accuracy_score(self.y_eval, y_pred))
        cm = confusion_matrix(self.y_eval, y_pred, labels=GLOBAL_CLASSES)
        tpr_macro, fpr_macro, tpr_by_class, fpr_by_class = _tpr_fpr_from_confusion(cm)

        cls_rep_dict = classification_report(
            self.y_eval,
            y_pred,
            labels=GLOBAL_CLASSES,
            zero_division=0,
            output_dict=True,
        )
        precision_macro = float(cls_rep_dict["macro avg"]["precision"])
        recall_macro = float(cls_rep_dict["macro avg"]["recall"])
        f1_macro = float(cls_rep_dict["macro avg"]["f1-score"])

        try:
            auc_ovr = float(roc_auc_score(self.y_eval, proba, multi_class="ovr", labels=GLOBAL_CLASSES))
        except Exception:
            auc_ovr = float("nan")
        try:
            auc_ovo = float(roc_auc_score(self.y_eval, proba, multi_class="ovo", labels=GLOBAL_CLASSES))
        except Exception:
            auc_ovo = float("nan")

        asr_val = None
        if (self.trigger_X is not None) and (self.backdoor_target is not None):
            try:
                proba_trigger = global_forest.predict_proba(self.trigger_X)
                y_trig_pred = np.argmax(proba_trigger, axis=1)
                asr_val = float(np.mean(y_trig_pred == int(self.backdoor_target)))
            except Exception as e:
                print(f"[ASR] errore calcolo ASR: {e}")
                asr_val = None

        out_dir = "data"
        df_counts, df_rowpct, cls_rep = _save_confusion_and_report(cm, self.y_eval, y_pred, out_dir=out_dir)
        _save_roc_plot(proba, self.y_eval, out_dir=out_dir)

        asr_str = f"{asr_val:.4f}" if asr_val is not None else "N/A"
        print(
            f"\n[SERVER] Round {server_round} — GLOBAL FOREST:"
            f"\n Accuracy_FL={acc:.4f} | MacroF1={f1_macro:.4f}"
            f"\n Precision_macro={precision_macro:.4f} Recall_macro={recall_macro:.4f}"
            f"\n TPR_macro={tpr_macro:.4f} FPR_macro={fpr_macro:.4f}"
            f"\n ROC-AUC OvR={auc_ovr if not np.isnan(auc_ovr) else 'nan'} OvO={auc_ovo if not np.isnan(auc_ovo) else 'nan'}"
            f"\n Pool trees={len(self.pool_trees)} | boost_vec={boost_vec.ravel().tolist()} (mode={self.boost_mode})"
        )
        print(f" ASR (backdoor target={self.backdoor_target}) = {asr_str}")
        print("[CONFUSION] TPR per classe:", tpr_by_class)
        print("[CONFUSION] FPR per classe:", fpr_by_class)
        print("[CONFUSION] Matrice di confusione (conteggi):\n", df_counts)
        print("[CONFUSION] Matrice di confusione (% per riga):\n", df_rowpct)
        print("[CONFUSION] Classification report:\n", cls_rep)
        print("-" * 80)

        self.prev_tpr_by_class = tpr_by_class

        metrics = {
            "acc": acc,
            "macro_f1": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "tpr_macro": tpr_macro,
            "fpr_macro": fpr_macro,
            "roc_auc_ovr": auc_ovr,
            "roc_auc_ovo": auc_ovo,
            "tpr_by_class": tpr_by_class,
            "fpr_by_class": fpr_by_class,
            "n_eval": int(len(self.y_eval)),
            "n_trees_pool": int(len(self.pool_trees)),
            "boost_vec": boost_vec.ravel().tolist(),
            "boost_mode": self.boost_mode,
            "asr": asr_val,
        }

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / f"metrics_round_{server_round}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return float(1.0 - acc), metrics


# Strategy ZKP wrapper
class ZKPStrategy(fl.server.strategy.Strategy):
    """Wrapper: verifica firme client e integra audit/commitments/transcript."""

    def __init__(
        self,
        base: fl.server.strategy.Strategy,
        logger=None,
        zkid2pub_hex: Dict[str, str] | None = None,
        autoreg: bool = False,
        preauth: bool = False,
    ) -> None:
        self.base = base
        self.zkp = ZKPRegistry(zkid2pub_hex, autoreg=autoreg)
        self.LOG = logger or logging.getLogger("SERVER")
        self.preauth = bool(preauth)
        self.aggregator_sk, self.aggregator_pk = client_zkp.keygen()
        self._cur_round_seed: Optional[bytes] = None

    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)

    def configure_parameters(self, server_round, parameters, client_manager):
        fn = getattr(self.base, "configure_parameters", None)
        return fn(server_round, parameters, client_manager) if fn else None

    def evaluate(self, server_round, parameters):
        return self.base.evaluate(server_round, parameters)

    def on_evaluate_config_fn(self):
        fn = getattr(self.base, "on_evaluate_config_fn", None)
        return fn() if fn else None

    def on_fit_config_fn(self):
        fn = getattr(self.base, "on_fit_config_fn", None)
        return fn() if fn else None

    # utilità ZKP
    def _ensure_props(self, clients: list) -> list:
        """
        Registra solo la corrispondenza cid↔zkid.
        """
        valid_clients = []
        for c in clients:
            try:
                ins = GetPropertiesIns(config={"zk_request": True})
                try:
                    res = c.get_properties(ins, 30.0, "zkp_props")
                except TypeError:
                    res = c.get_properties(ins)
                props = dict(res.properties or {})
                zkid = str(props.get("zk_client_id", "")).strip()
                if not zkid:
                    self.LOG.warning(f"[ZKP][SERVER] client {c.cid} senza zkid, skip")
                    continue

                # registra solo il binding cid↔zkid (idempotente)
                try:
                    self.zkp.register(c.cid, zkid)
                except RuntimeError as e:
                    self.LOG.warning(f"[ZKP][SERVER] {e}; skip {c.cid}")
                    continue

                pub_value = props.get("zk_pubkey_int") or props.get("zk_pubkey") or props.get("pub_hex") or ""
                if pub_value and not self.zkp.has_pubkey(zkid):
                    try:
                        pk_val = int(pub_value)
                        self.zkp.zkid2pub[str(zkid)] = pk_val
                        self.LOG.info(f"[ZKP][SERVER] Registrata pubkey per zkid={zkid} (autoreg)")
                    except Exception:
                        pass

                valid_clients.append(c)
            except Exception as e:
                self.LOG.warning(f"[ZKP][SERVER] errore properties da {c.cid}: {e}")

        self.LOG.info(f"[ZKP][SERVER] Client validi dopo ensure_props: {len(valid_clients)}/{len(clients)}")
        return valid_clients

    def _pre_auth_round(self, client: fl.server.client_proxy.ClientProxy, zkid: str, rnd: int) -> bool:
        """Pre-auth opzionale: il client firma una challenge e la verifichiamo."""
        if not self.preauth:
            return True
        if not self.zkp.has_pubkey(zkid):
            return False

        challenge = self.zkp.get_or_make_nonce("auth", rnd, zkid)
        ins = GetPropertiesIns(config={
            "zk_request": True,
            "zk_phase": "auth",
            "zk_round": int(rnd),
            "zk_challenge": challenge
        })
        try:
            res = client.get_properties(ins, 30.0, "zkp_auth")
        except TypeError:
            res = client.get_properties(ins)
        props = dict(res.properties or {})
        sig_R = props.get("zk_proof_R")
        sig_s = props.get("zk_proof_s")
        pk = self.zkp.pub_for(zkid)

        if not (sig_R and sig_s and pk):
            self.LOG.warning(f"[ZKP][SERVER] pre-auth dati mancanti per cid={client.cid}")
            return False

        try:
            R_int = int(sig_R, 16) if isinstance(sig_R, str) and sig_R.startswith("0x") else int(sig_R)
            s_int = int(sig_s, 16) if isinstance(sig_s, str) and sig_s.startswith("0x") else int(sig_s)
            ok = client_zkp.schnorr_verify(pk, str(challenge).encode(), (R_int, s_int))
            if not ok:
                self.LOG.warning(f"[ZKP][SERVER] pre-auth signature invalid for cid={client.cid}")
            return ok
        except Exception:
            return False

    def _inject_zk_into_cfg(self, ins_list, server_round: int, phase: str):
        """
        Inserisce metadati zk nella config. Non richiede più la pubkey nel registry
        per far partire il task (la firma si verifica dopo con la chiave inviata nelle metrics).
        Se --zk-preauth è attivo, allora serve la pubkey nel registry.
        """
        clients = [c for c, _ in ins_list]
        valid_clients = self._ensure_props(clients)

        new_list, failures = [], []

        # seed unico per round
        self._cur_round_seed = secrets.token_bytes(16)
        seed_hex = self._cur_round_seed.hex()

        for c, ins in ins_list:
            if c not in valid_clients:
                self.LOG.warning(f"[ZKP][SERVER] Excluding {c.cid} from {phase}: not validated")
                failures.append((c, RuntimeError("ZKP_NOT_VALIDATED")))
                continue

            zkid = self.zkp.zkid_for(c.cid)

            # SOLO se pre-auth è abilitata: serve la pubkey nel registry
            if self.preauth:
                if not zkid or not self.zkp.has_pubkey(zkid):
                    self.LOG.warning(f"[ZKP] preauth attiva, ma no pubkey per cid={c.cid} (zkid={zkid}); skip {phase}")
                    failures.append((c, RuntimeError("ZKP_NO_PUBKEY_PREAUTH")))
                    continue
                if not self._pre_auth_round(c, zkid, server_round):
                    self.LOG.warning(f"[ZKP] skip {phase} per {c.cid}: pre-auth fallita")
                    failures.append((c, RuntimeError("ZKP_PREAUTH_FAILED")))
                    continue

            cfg = dict(ins.config or {})
            cfg.update({
                "zk_required": True,
                "zk_phase": phase,
                "zk_round": int(server_round),
                "zk_round_seed": seed_hex,
            })
            if isinstance(ins, FitIns):
                new_list.append((c, FitIns(ins.parameters, cfg)))
            else:
                new_list.append((c, EvaluateIns(ins.parameters, cfg)))

        self.LOG.info(f"[ZKP] {phase.upper()} sending {len(new_list)}/{len(ins_list)} tasks after ZK filtering")
        return new_list, failures

    
    #  Verifica ZKP + hash alberi + guardrail sentinella
    def _filter_on_zkp(self, results, server_round: int, phase: str):
        """Verifica firme Schnorr sui payload client e filtra i client non validi o fuori bound L2."""
        ok, failures = [], []
        self._accepted_submissions = []
        clip_bound = getattr(self.base, "clip_bound", float("inf"))
        self.LOG.info(f"[ZKP] Round {server_round} – clip_bound={clip_bound}")

        # --- hash helper ---
        def _hash_trees(lst):
            h = hashlib.sha256()
            for tb in lst:
                h.update(hashlib.sha256(tb).digest())
            return h.hexdigest()

        for cp, res in results:
            try:
                m: Metrics = res.metrics or {}
                zkid = str(m.get("zk_client_id", "")).strip()
                pk_raw = m.get("zk_pubkey_int") or m.get("zk_pubkey") or None
                sig_json = m.get("zk_signature") or None

                if not (zkid and pk_raw and sig_json):
                    self.LOG.warning(f"[ZKP][SERVER] {cp.cid}: campi mancanti; excluded")
                    failures.append((cp, RuntimeError("ZKP_MISSING_FIELDS")))
                    continue

                try:
                    pk_val = int(str(pk_raw))
                    sig_dict = json.loads(sig_json)
                    sig = (int(sig_dict["R"]), int(sig_dict["s"]))
                    q_update = [int(v) for v in json.loads(m.get("q_update", "[]"))]
                    scale = int(str(m.get("scale", 1000)))
                    client_id = str(m.get("client_id", ""))
                    round_id = int(str(m.get("zk_round", server_round)))
                    trees_hash_client = str(m.get("trees_hash", "")).strip()
                except Exception as e:
                    self.LOG.warning(f"[ZKP][SERVER] {cp.cid}: parsing error: {e}")
                    failures.append((cp, RuntimeError("ZKP_PARSE_ERROR")))
                    continue

                # calcola hash alberi realmente ricevuti
                trees_bytes = []
                try:
                    arr_list = parameters_to_ndarrays(res.parameters)
                    if arr_list:
                        blob = arr_list[0].tobytes()
                        obj = pickle.loads(blob)
                        if isinstance(obj, dict) and "trees" in obj:
                            trees_bytes = obj["trees"]
                except Exception:
                    pass

                trees_hash_server = _hash_trees(trees_bytes)

                # messaggio base + hash alberi
                msg_bytes = client_zkp.serialize_update_message(
                    client_id=client_id,
                    round_id=round_id,
                    scale=scale,
                    q_update=q_update,
                )
                msg_full = msg_bytes + trees_hash_server.encode()

                # verifica firma Schnorr
                try:
                    if not client_zkp.schnorr_verify(pk_val, msg_full, sig):
                        self.LOG.warning(f"[ZKP][SERVER] firma invalida (hash mismatch?) da {cp.cid}")
                        failures.append((cp, RuntimeError("ZKP_SIG_INVALID")))
                        continue
                except Exception as e:
                    self.LOG.warning(f"[ZKP][SERVER] errore verifica firma {cp.cid}: {e}")
                    failures.append((cp, RuntimeError("ZKP_SIG_ERROR")))
                    continue

                # verifica coerenza hash alberi
                if trees_hash_client and trees_hash_client != trees_hash_server:
                    self.LOG.warning(f"[ZKP][SERVER] hash alberi non combacia per {client_id}")
                    failures.append((cp, RuntimeError("ZKP_TREES_HASH_MISMATCH")))
                    continue

                # controllo L2 clipping
                try:
                    vec = [v / scale for v in q_update]
                    norm_val = float(np.linalg.norm(vec))
                    if norm_val > clip_bound:
                        self.LOG.warning(
                            f"[ZKP][SERVER] client {client_id} escluso per L2={norm_val:.4f} > bound={clip_bound}"
                        )
                        failures.append((cp, RuntimeError("ZKP_L2_TOO_HIGH")))
                        continue
                except Exception as e:
                    self.LOG.warning(f"[ZKP][SERVER] errore calcolo L2 per {client_id}: {e}")
                    failures.append((cp, RuntimeError("ZKP_L2_ERROR")))
                    continue

                # guard-rail sentinella pubblico S
                try:
                    test_X = getattr(self.base, "X_eval", None)
                    test_y = getattr(self.base, "y_eval", None)
                
                    if test_X is not None and test_y is not None:
                        trees_temp = self.base._extract_trees_from_fitres(res)
                
                        if trees_temp:
                            # F1 del modello prima del contributo
                            forest_before = GlobalForest(list(self.base.pool_trees))
                            f1_before = float(f1_score(test_y,
                                        forest_before.predict(test_X),
                                        average="macro"))
                
                            # F1 dopo aver aggiunto solo gli alberi del client
                            forest_after = GlobalForest(list(self.base.pool_trees) + trees_temp)
                            f1_after = float(f1_score(test_y,
                                        forest_after.predict(test_X),
                                        average="macro"))
                
                            delta = f1_after - f1_before
                            threshold = 1#0.0015 
                
                            self.LOG.info(
                                f"[DEBUG][SENTINEL] client {client_id}: "
                                f"F1_before={f1_before:.4f}  F1_after={f1_after:.4f}  Δ={delta:.4f}"
                            )
                
                            # se il client degrada il modello: scarto
                            if delta < -threshold:
                                self.LOG.warning(
                                    f"[ZKP][SERVER] client {client_id} SCARTATO: degrado sentinella Δ={delta:.4f}"
                                )
                                failures.append((cp, RuntimeError("ZKP_SENTINEL_FAIL")))
                                continue
                except Exception as e:
                    self.LOG.warning(f"[ZKP][SERVER] errore guard-rail sentinella per {client_id}: {e}")
                    failures.append((cp, RuntimeError("ZKP_SENTINEL_ERROR")))
                    continue

                except Exception as e:
                    self.LOG.warning(f"[ZKP][SERVER] errore guard-rail sentinella per {client_id}: {e}")
                    failures.append((cp, RuntimeError("ZKP_SENTINEL_ERROR")))
                    continue

                self._accepted_submissions.append({
                    "client_id": client_id,
                    "pk": pk_val,
                    "q_update": q_update,
                    "scale": scale,
                    "sig": sig,
                    "msg": msg_bytes,
                    "round": round_id,
                    "_cp": cp,
                    "_res": res,
                })
                ok.append((cp, res))

            except Exception as e:
                self.LOG.warning(f"[ZKP][SERVER] errore generale filter {cp.cid}: {e}")
                failures.append((cp, RuntimeError("ZKP_FILTER_EXCEPTION")))
                continue

        self.zkp.gc(current_round=server_round)
        self.LOG.info(
            f"[ZKP] Round {server_round}: accettati {len(ok)} / {len(results)} (ZKP+clip+sentinella)"
        )
        return ok, failures

    # ----- hooks -----
    def configure_fit(self, server_round, parameters, client_manager):
        ins_list = self.base.configure_fit(server_round, parameters, client_manager)
        if not ins_list:
            return ins_list
        new_list, failures = self._inject_zk_into_cfg(ins_list, server_round, "fit")
        self._last_inject_failures_fit = failures

        if hasattr(self, "_last_transcript_payload"):
            for cp, ins in new_list:
                cfg = dict(ins.config) 
                cfg["zkp_verify_payload"] = self._last_transcript_payload
                ins.config = cfg


        return new_list

    def configure_evaluate(self, server_round, parameters, client_manager):
        ins_list = self.base.configure_evaluate(server_round, parameters, client_manager)
        if not ins_list:
            return ins_list
        new_list, failures = self._inject_zk_into_cfg(ins_list, server_round, "eval")
        self._last_inject_failures_eval = failures
        return new_list

    def aggregate_fit(self, server_round, results, failures):
        """Applica ZKP e clipping; esclude client non validi o fuori bound prima dell'aggregazione."""
        # Verifica firme e raccoglie submissions valide
        filtered, zkp_failures = self._filter_on_zkp(results, server_round, "fit")
    
        clip_bound = getattr(self.base, "clip_bound", float("inf"))
        self.LOG.info(f"[ZKP] Round {server_round} – clip_bound={clip_bound}")
    
        # Recupera i submissions validi
        accepted_subs = getattr(self, "_accepted_submissions", [])
        accepted_cids = {s["_cp"].cid for s in accepted_subs}
        results_final = [(cp, res) for (cp, res) in filtered if cp.cid in accepted_cids]
    
        # Estrae alberi solo dai client validi
        new_trees = []
        for _, fit_res in results_final:
            new_trees.extend(self.base._extract_trees_from_fitres(fit_res))
    
        prev_len = len(self.base.pool_trees)
        self.base.pool_trees.extend(new_trees)
        added = len(self.base.pool_trees) - prev_len
    
        kept_new = added
        if len(self.base.pool_trees) > self.base.pool_max_trees:
            scores = [
                (_score_tree_macro_f1(t, self.base.X_eval, self.base.y_eval), i)
                for i, t in enumerate(self.base.pool_trees)
            ]
            scores.sort(reverse=True)
            keep_idx = {i for _, i in scores[: self.base.pool_max_trees]}
            kept_new = sum(1 for i in range(prev_len, prev_len + added) if i in keep_idx)
            self.base.pool_trees = [t for i, t in enumerate(self.base.pool_trees) if i in keep_idx]
    
        print(
            f"[POOL] round={server_round} new_trees={added} kept_after_prune={kept_new} "
            f"pool_size={len(self.base.pool_trees)}/{self.base.pool_max_trees}"
        )

        print(f"[ZKP] Round {server_round}: accepted {len(results_final)} / {len(results)} clients after ZKP+L2 filtering")
    
        # Costruzione Merkle / transcript audit
        if accepted_subs:
            try:
                dim = len(accepted_subs[0]["q_update"])
                sums = [0] * dim
                for sub in accepted_subs:
                    for j, v in enumerate(sub["q_update"]):
                        sums[j] += v
                nacc = len(accepted_subs)
                delta_q = [s // nacc for s in sums]
    
                round_seed = self._cur_round_seed or secrets.token_bytes(16)
                scale_used = accepted_subs[0]["scale"]
                commitments, proofs, _ = compute_commitments_and_proofs(
                    accepted_subs, round_seed, scale=scale_used, chunk_size=None
                )
                transcript = build_transcript(
                    round_seed,
                    accepted_subs,
                    commitments,
                    proofs,
                    delta_q,
                    scale_used,
                    self.aggregator_sk,
                    self.aggregator_pk,
                )
                print(f"[ZKP] Transcript published with merkle {transcript['merkle_root']}")
            except Exception as e:
                print(f"[ZKP] Errore durante audit transcript: {e}")

            # Payload da mandare al client per round
            try:
                # Prepara dati per verifica lato client
                accepted_msgs = [sub["msg"].hex() for sub in accepted_subs]
            
                # Metriche sentinella
                sentinel_before = getattr(self.base, "_sentinel_before", None)
                sentinel_after = getattr(self.base, "_sentinel_after", None)
                sentinel_tau = getattr(self.base, "_sentinel_tau", 0.001)
            
                transcript_payload = {
                    "transcript": transcript,
                    "accepted_msgs": accepted_msgs,
                    "sentinel_before": float(sentinel_before) if sentinel_before is not None else 0.0,
                    "sentinel_after": float(sentinel_after) if sentinel_after is not None else 0.0,
                    "sentinel_tau": float(sentinel_tau),
                }
            
                # payload nel registry per mandarlo ai client al prossimo round
                self._last_transcript_payload = json.dumps(transcript_payload)
            
            except Exception as e:
                print(f"[ZKP][SERVER] Errore preparazione payload ZKP client: {e}")

    
        # Unione di tutte le failure
        all_failures = failures + zkp_failures + getattr(self, "_last_inject_failures_fit", [])
        return self.base.aggregate_fit(server_round, results_final, all_failures)



    def aggregate_evaluate(self, server_round, results, failures):
        filtered, zkp_failures = self._filter_on_zkp(results, server_round, "eval")
        all_failures = failures + zkp_failures + getattr(self, "_last_inject_failures_eval", [])
        return self.base.aggregate_evaluate(server_round, filtered, all_failures)


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:8083")
    parser.add_argument("--server", dest="address", help="Alias di --address")
    parser.add_argument("--eval_csv", default="data/public_eval.csv")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--trees_per_round", type=int, default=30)
    parser.add_argument("--pool_max_trees", type=int, default=1000)
    parser.add_argument("--min_fit_clients", type=int, default=6)
    parser.add_argument("--min_available_clients", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)

    # Boost
    parser.add_argument(
        "--boost_mode",
        choices=["off", "fixed", "adaptive"],
        default="fixed",
        help="off=falso, fixed=usa --boost, adaptive=calcola da TPR round-1",
    )
    parser.add_argument(
        "--boost",
        default="1:1.50,2:1.50",
        help='Esempio: "1:1.25,2:1.25,3:0.95" (usato solo con --boost_mode fixed)',
    )
    parser.add_argument("--target_tpr", type=float, default=0.65, help="Target recall per boost adattivo")
    parser.add_argument("--boost_k", type=float, default=0.7, help="Sensibilità boost adattivo")
    parser.add_argument("--boost_cap", type=float, default=1.35, help="Limite massimo del boost per classe")

    # ASR / backdoor options
    parser.add_argument("--trigger_csv", default=None, help="CSV con campioni trigger (opzionale) per calcolo ASR")
    parser.add_argument(
        "--backdoor_target",
        type=int,
        default=1,
        help="Classe target del backdoor (es. 0). Se None ASR non verrà calcolata",
    )

    # ZKP
    parser.add_argument(
        "--zk-autoreg",
        action="store_false",
        help="Dev only: accetta pubkey inviata dal client al primo contatto."
    )
    parser.add_argument(
        "--zk-preauth",
        action="store_true",
        help="Se true, esegue pre-auth (firma challenge) prima di inviare fit/eval."
    )
    parser.add_argument(
        "--clip-bound",
        type=float,
        default= 2.5,#1e9,
        help="L2 clipping bound per gli update dei client (default: no clipping)"
    )

    args = parser.parse_args()
    LOG = logging.getLogger("SERVER")

    # Carica dataset di valutazione
    X_eval, y_eval, _ = load_and_preprocess_csv(args.eval_csv, label_col="marker")

    fixed_boost_vec = _parse_boost(args.boost)

    # carica trigger
    trigger_X = None
    if args.trigger_csv is not None:
        try:
            trigger_X = _load_trigger_csv(args.trigger_csv)
            print(f"[ASR] Caricati {trigger_X.shape[0]} campioni trigger da {args.trigger_csv}")
        except Exception as e:
            print(f"[ASR] Impossibile caricare trigger_csv {args.trigger_csv}: {e}")
            trigger_X = None

    # Carica mapping pubkeys
    zkid_map: Dict[str, str] = {}
    try:
        with open(SERVER_JSON, "r") as f:
            zkid_map = json.load(f)
    except Exception as e:
        LOG.warning(f"")

    if not zkid_map:
        collected = collect_client_pubkeys(CLIENTS_DIR)
        if collected:
            save_server_json(collected, SERVER_JSON)
            zkid_map = collected
        else:
            if SERVER_JSON.exists():
                try:
                    with open(SERVER_JSON, "r") as f:
                        zkid_map = json.load(f)
                except Exception:
                    pass

    base = FederatedForestStrategy(
        X_eval=X_eval,
        y_eval=y_eval,
        trees_per_round=args.trees_per_round,
        pool_max_trees=args.pool_max_trees,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        boost_mode=args.boost_mode,
        boost_vec=fixed_boost_vec,
        target_tpr=args.target_tpr,
        boost_k=args.boost_k,
        boost_cap=args.boost_cap,
        trigger_X=trigger_X,
        backdoor_target=args.backdoor_target,
    )

    # Imposta il clip bound
    base.clip_bound = float(args.clip_bound) if hasattr(args, "clip_bound") else 10.0
    LOG.info(f"[DEBUG] base.clip_bound set to {base.clip_bound}")

    strategy = ZKPStrategy(
        base,
        logger=LOG,
        zkid2pub_hex=zkid_map,
        autoreg=args.zk_autoreg,
        preauth=args.zk_preauth,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
