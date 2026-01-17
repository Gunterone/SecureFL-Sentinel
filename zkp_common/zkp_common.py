# Server-side utilities: ZKP registry, Merkle, Pedersen verify/opening, deterministic blinding.
from __future__ import annotations
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict
from zkp_common import zkp_client_utils as client_zkp

P = client_zkp.P
Q = client_zkp.Q
G = client_zkp.G
HGEN = client_zkp.HGEN
H_int = client_zkp.H_int
H_bytes = client_zkp.H_bytes

# Merkle helpers
def merkle_root(leaves: List[bytes]) -> bytes:
    if not leaves:
        return b"\x00" * 32
    nodes = [hashlib.sha256(l).digest() for l in leaves]
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nxt = []
        for i in range(0, len(nodes), 2):
            nxt.append(hashlib.sha256(nodes[i] + nodes[i + 1]).digest())
        nodes = nxt
    return nodes[0]

def merkle_proof_for_index(leaves: List[bytes], idx: int) -> List[bytes]:
    # ritorna lista di siblings hashes per prova
    if not leaves:
        return []
    nodes = [hashlib.sha256(l).digest() for l in leaves]
    proof = []
    pos = idx
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        next_nodes = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1]
            next_nodes.append(hashlib.sha256(left + right).digest())
            if i == pos ^ 1 or i == pos - 1:
                # determa sibling
                pass
        sibling_index = pos ^ 1
        sibling = nodes[sibling_index] if sibling_index < len(nodes) else nodes[-1]
        proof.append(sibling)
        pos = pos // 2
        nodes = [hashlib.sha256(nodes[i] + nodes[i + 1]).digest() for i in range(0, len(nodes), 2)]
    return proof

# Pedersen verify/opening
def pedersen_commit(m: int, r: int) -> int:
    return (pow(G, m % Q, P) * pow(HGEN, r % Q, P)) % P

def schnorr_prove_opening(C: int, m: int, r: int) -> Tuple[int, int]:
    """
    Produce Schnorr PoK per provare la conoscenza di r in modo che C = g^m * h^r.
    """
    Cprime = (C * pow(G, (-m) % Q, P)) % P
    k = secrets.randbelow(Q)
    R = pow(HGEN, k, P)
    e = H_int(R.to_bytes(64, "big"), Cprime.to_bytes(64, "big"))
    s = (k + e * r) % Q
    return (R, s)

def schnorr_verify_opening(C: int, m: int, proof: Tuple[int, int]) -> bool:
    R, s = proof
    Cprime = (C * pow(G, (-m) % Q, P)) % P
    e = H_int(R.to_bytes(64, "big"), Cprime.to_bytes(64, "big"))
    return pow(HGEN, s, P) == (R * pow(Cprime, e, P)) % P

# deterministic blinding derivation
def deterministic_blind(round_seed: bytes, tag: str, j: int) -> int:
    data = b"agg" + round_seed + tag.encode() + j.to_bytes(4, "big")
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % Q

# Registry (mapping cid <-> zkid & pubkeys)
class ZKPRegistry:
    def __init__(self, zkid2pub: Dict[str, int] | None = None, autoreg: bool = False):
        self.cid2zkid: Dict[str, str] = {}
        self.zkid2cid: Dict[str, str] = {}
        self.zkid2pk: Dict[str, int] = {}
        self.autoreg = bool(autoreg)
        if zkid2pub:
            # zkid2pub mapping
            for zkid, pk in zkid2pub.items():
                self.zkid2pk[str(zkid)] = int(pk)

    def register(self, cid: str, zkid: str) -> None:
        prev_for_cid = self.cid2zkid.get(cid)
        prev_for_zkid = self.zkid2cid.get(zkid)
        if prev_for_cid and prev_for_cid != zkid:
            raise RuntimeError(f"CID {cid} already bound to zkid {prev_for_cid}")
        if prev_for_zkid and prev_for_zkid != cid:
            raise RuntimeError(f"ZKID {zkid} already bound to cid {prev_for_zkid}")
        self.cid2zkid[cid] = zkid
        self.zkid2cid[zkid] = cid

    def zkid_for(self, cid: str) -> Optional[str]:
        return self.cid2zkid.get(cid)

    def has_pubkey(self, zkid: str) -> bool:
        return zkid in self.zkid2pk

    def set_pubkey_if_allowed(self, zkid: str, pk: Optional[int]) -> None:
        if self.has_pubkey(zkid) or not self.autoreg or pk is None:
            return
        self.zkid2pk[zkid] = int(pk)

    def pub_for(self, zkid: str) -> Optional[int]:
        return self.zkid2pk.get(zkid)

    def verify_client_signature(self, pk: int, msg: bytes, sig: Tuple[int, int]) -> bool:
        return client_zkp.schnorr_verify(pk, msg, sig)
