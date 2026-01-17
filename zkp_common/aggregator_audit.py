# aggregator_audit.py

from __future__ import annotations
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path
import secrets

from zkp_common import zkp_common as zk_common  
from zkp_common import zkp_client_utils as client_zkp 

TRANSCRIPTS_DIR = Path("data/zkp_transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

def build_leaves_from_accepted(accepted_submissions: List[Dict[str, Any]]) -> List[bytes]:
    leaves: List[bytes] = []
    for sub in accepted_submissions:
        # serializzazione canonica per computare hash delle foglie, include firma e msg originale
        msg = sub.get("msg")
        if isinstance(msg, bytes):
            leaf = msg
        else:
            # fallback
            leaf = json.dumps({
                "client_id": sub.get("client_id"),
                "round": str(sub.get("round", "")),
                "scale": int(sub.get("scale", 1000)),
                "update": sub.get("q_update"),
                "pk": int(sub.get("pk")),
                "sig": sub.get("sig")
            }, separators=(",", ":"), sort_keys=True).encode()
        leaves.append(leaf)
    return leaves

def compute_commitments_and_proofs(accepted: List[Dict[str, Any]], round_seed: bytes, scale: int = 1000, chunk_size: int | None = None) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    """
    Computa commitments C_j e prove di apertura Schnorr per ogni coordinata.
    Ritorna (commitments, proofs, sums_mj)
    """
    if not accepted:
        return [], [], []

    dim = len(accepted[0]["q_update"])
    # chunking
    if chunk_size is None or chunk_size <= 0:
        chunk_size = dim
    n_chunks = (dim + chunk_size - 1) // chunk_size

    commitments: List[int] = []
    proofs: List[Tuple[int, int]] = []
    sums_mj: List[int] = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(dim, start + chunk_size)
        m_chunk = 0
        for j in range(start, end):
            m_j = sum(sub["q_update"][j] for sub in accepted)
            m_chunk += m_j
        r_j = zk_common.deterministic_blind(round_seed, "r", chunk_idx)
        C_j = client_zkp.pedersen_commit(m_chunk, r_j)
        prf = zk_common.schnorr_prove_opening(C_j, m_chunk, r_j)
        commitments.append(int(C_j))
        proofs.append((int(prf[0]), int(prf[1])))
        sums_mj.append(int(m_chunk))

    return commitments, proofs, sums_mj

def build_transcript(round_seed: bytes, accepted_submissions: List[Dict[str, Any]], commitments: List[int], proofs: List[Tuple[int, int]], delta_q: List[int], scale: int, aggregator_sk: int, aggregator_pk: int) -> Dict[str, Any]:
    leaves = build_leaves_from_accepted(accepted_submissions)
    merkle = zk_common.merkle_root(leaves).hex()
    # signed body
    core = {
        "round_seed": round_seed.hex(),
        "merkle_root": merkle,
        "commitments": commitments,
        "proofs": proofs,
        "delta_q": delta_q,
        "scale": int(scale)
    }
    core_msg = json.dumps(core, separators=(",", ":"), sort_keys=True).encode()
    sig = client_zkp.schnorr_sign(aggregator_sk, core_msg)
    transcript = dict(core)
    transcript["aggregator_pk"] = int(aggregator_pk)
    transcript["aggregator_sig"] = (int(sig[0]), int(sig[1]))
    # salvataggio
    idx = secrets.token_hex(8)
    out_fp = TRANSCRIPTS_DIR / f"transcript_{idx}.json"
    with out_fp.open("w") as f:
        json.dump(transcript, f, indent=2)
    return transcript

def verify_transcript(transcript: Dict[str, Any], accepted_msgs: List[bytes]) -> bool:
    # valida firma aggregatore e coerenza firma e albero
    agg_pk = int(transcript["aggregator_pk"])
    core = {k: transcript[k] for k in ["round_seed", "merkle_root", "commitments", "proofs", "delta_q", "scale"]}
    core_msg = json.dumps(core, separators=(",", ":"), sort_keys=True).encode()
    if not client_zkp.schnorr_verify(agg_pk, core_msg, tuple(transcript["aggregator_sig"])):
        return False
    # verify merkle root
    if zk_common.merkle_root(accepted_msgs).hex() != transcript["merkle_root"]:
        return False
    # verify openings
    r_seed = bytes.fromhex(transcript["round_seed"])
    commitments = transcript["commitments"]
    proofs = transcript["proofs"]
    updates = [json.loads(m.decode())["update"] for m in accepted_msgs]
    if updates:
        n = len(updates)
    else:
        n = 1
    # reconstruct sums per chunk same as compute_commitments_and_proofs
    dim = len(updates[0]) if updates else 0
    # assume chunk_size equals dim
    for j in range(len(commitments)):
        # compute m_chunk
        m_chunk = 0
        # for assume chunk covers entire dim or proportional;
        for coord in range(dim):
            m_chunk += sum(u[coord] for u in updates)
        r_j = zk_common.deterministic_blind(r_seed, "r", j)
        if not zk_common.schnorr_verify_opening(commitments[j], m_chunk, proofs[j]):
            return False
    # verify delta_q coherence
    target = []
    for coord in range(dim):
        s = sum(u[coord] for u in updates)
        avg = s // n
        target.append(int(avg))
    return target == transcript["delta_q"]
