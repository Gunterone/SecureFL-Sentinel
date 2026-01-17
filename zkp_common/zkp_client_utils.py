# Utilities ZKP client: keygen, Schnorr sign/verify, quantize helpers, message packing.

from __future__ import annotations
import os
import json
import math
import hashlib
import secrets
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Gruppo sicuro
# Primo sicuro fisso
def gen_demo_safe_prime():
    q = int("C196BA0D5F06C3E5B60A5C4B9E19F32F9C5E9E9D6F5B0F1E6D9F4B2E7A3D1C7", 16)
    p = 2 * q + 1
    return p, q

P, Q = gen_demo_safe_prime()

# Generatore deterministico del sottogruppo di ordine Q:
G_FULL = 2
G = pow(G_FULL, 2, P)  # deterministico e uguale ovunque

# Generatore H per Pedersen commitment deterministico (deriva t dall’hash costante)
def _H_to_Zq(*parts: bytes) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return (int.from_bytes(h.digest(), "big") % Q) or 1

def hash_to_subgroup(tag: bytes) -> int:
    # Mappa 'tag' in Z_p^*, poi proietta nel sottogruppo di ordine q
    x = int.from_bytes(hashlib.sha256(tag).digest(), "big") % P
    if x in (0, 1):  # evita 0 e 1
        x = 2
    # proiezione nel sottogruppo di ordine q 
    h = pow(x, (P - 1) // Q, P)
    if h == 1:
        # cambia leggermente il tag e riprova
        x = (x + 1) % P
        h = pow(x, (P - 1) // Q, P)
        if h == 1:
            raise RuntimeError("hash_to_subgroup failed to find non-identity")
    return h

HGEN = hash_to_subgroup(b"pedersen-h-generator-v1")

# Hash helpers
def H_int(*parts: bytes) -> int:
    m = hashlib.sha256()
    for part in parts:
        m.update(part)
    return int.from_bytes(m.digest(), "big") % Q

def H_bytes(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for part in parts:
        m.update(part)
    return m.digest()

# Schnorr
def keygen() -> Tuple[int, int]:
    sk = secrets.randbelow(Q - 1) + 1
    pk = pow(G, sk, P)
    return sk, pk

def schnorr_sign(sk: int, msg: bytes) -> Tuple[int, int]:
    k = secrets.randbelow(Q - 1) + 1
    R = pow(G, k, P)
    # fissa la lunghezza in byte di R in base a P
    R_bytes_len = (P.bit_length() + 7) // 8
    e = H_int(R.to_bytes(R_bytes_len, "big"), msg)
    s = (k + e * sk) % Q
    return (R, s)

def schnorr_verify(pk: int, msg: bytes, sig: Tuple[int, int]) -> bool:
    R, s = sig
    R_bytes_len = (P.bit_length() + 7) // 8
    e = H_int(R.to_bytes(R_bytes_len, "big"), msg)
    lhs = pow(G, s, P)
    rhs = (R * pow(pk, e, P)) % P
    return lhs == rhs

# Pedersen commit
def pedersen_commit(m: int, r: int) -> int:
    return (pow(G, m % Q, P) * pow(HGEN, r % Q, P)) % P

# Quantizzazione
def quantize_vector(vec: List[float], scale: int = 1000) -> List[int]:
    return [int(round(float(v) * scale)) for v in vec]

def dequantize_vector(vec_q: List[int], scale: int = 1000) -> List[float]:
    return [v / scale for v in vec_q]

def l2norm(vec: List[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))

# Messaggio canonico
def pack_update_message(client_id: str, q_update: List[int], scale: int, round_id: int | str) -> bytes:
    """
    Serializza il payload firmato (client_id, round, scale, update) in JSON canonico:
    - sort_keys=True
    - separators=(",", ":")
    """
    payload = {"client_id": client_id, "round": str(round_id), "scale": int(scale), "update": [int(v) for v in q_update]}
    msg = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return msg

# usato dal client/server aggiornati
serialize_update_message = pack_update_message

# Key storage
def zk_dir(client_id: str) -> str:
    p = os.path.join("zkp_common", "clients_zkp", client_id)
    os.makedirs(p, exist_ok=True)
    return p

def load_or_create_zkid(client_id: str) -> str:
    folder = zk_dir(client_id)
    fp = Path(folder) / "public.json"
    if fp.exists():
        try:
            with fp.open("r") as f:
                data = json.load(f)
            zkid = str(data.get("client_id", "")).strip()
            if zkid:
                return zkid
        except Exception:
            pass
    zkid = secrets.token_hex(16)
    with fp.open("w") as f:
        json.dump({"client_id": zkid}, f)
    return zkid

def load_or_create_keypair(client_id: str) -> Tuple[int, int]:
    """
    Crea o carica (sk, pk). pk = G^sk mod P con G deterministico.
    """
    folder = zk_dir(client_id)
    sk_fp = Path(folder) / "secret.json"
    pk_fp = Path(folder) / "public.json"
    zkid = load_or_create_zkid(client_id)

    if sk_fp.exists():
        try:
            with sk_fp.open("r") as f:
                data = json.load(f)
            sk = int(data.get("sk"))
            pk = pow(G, sk, P)
            with pk_fp.open("w") as f:
                json.dump({"client_id": zkid, "pk": pk}, f)
            return sk, pk
        except Exception:
            pass

    sk, pk = keygen()
    with sk_fp.open("w") as f:
        json.dump({"sk": sk}, f)
    with pk_fp.open("w") as f:
        json.dump({"client_id": zkid, "pk": pk}, f)
    return sk, pk

def produce_signed_update(
    client_id: str,
    sk: int,
    q_update: List[int],
    scale: int,
    round_id: int | str,
    extra_data: bytes | None = None,
) -> Dict[str, Any]:
    """
    Genera un messaggio firmato Schnorr che include hash cumulativo degli alberi.
    """
    #  Messaggio base
    msg = pack_update_message(client_id, q_update, scale, round_id)

    #  Se c'è extra_data, concatena nel messaggio firmato
    msg_full = msg + (extra_data or b"")

    #  Firma Schnorr
    sig = schnorr_sign(sk, msg_full)

    return {
        "client_id": client_id,
        "pk": pow(G, sk, P),
        "q_update": [int(v) for v in q_update],
        "scale": int(scale),
        "sig": (int(sig[0]), int(sig[1])),
        "msg": msg,
        "round": str(round_id),
        "extra_data_hex": extra_data.hex() if extra_data else None,
    }

