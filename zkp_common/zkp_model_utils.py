
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

try:
    from ecdsa import SECP256k1, ellipticcurve
except Exception as e:
    raise SystemExit(
        "Missing dependency 'ecdsa'. Install with:  pip install ecdsa"
    ) from e

CURVE = SECP256k1
G = CURVE.generator
N = CURVE.order

# Utility: raccolta pubkeys

def collect_client_pubkeys(clients_dir: Path) -> dict:
    """Scansiona cartelle client e costruisce mapping {zkid: pub_hex} (solo per test iniziali)."""
    all_pubkeys = {}
    if not clients_dir.exists():
        return all_pubkeys
    for client_folder in clients_dir.iterdir():
        if client_folder.is_dir():
            public_json = client_folder / "public.json"
            schnorr_pk_json = client_folder / "schnorr_pk.json"
            try:
                if public_json.exists():
                    with open(public_json, "r") as f:
                        client_data = json.load(f)
                    zkid = client_data.get("client_id")
                else:
                    continue
                if schnorr_pk_json.exists():
                    with open(schnorr_pk_json, "r") as f:
                        pk_data = json.load(f)
                    pubkey = pk_data.get("pub_hex")
                else:
                    continue
                if zkid and pubkey:
                    all_pubkeys[str(zkid)] = str(pubkey)
            except Exception:
                pass
    return all_pubkeys


def save_server_json(pubkeys: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pubkeys, f, indent=2)


# Funzioni di base per Schnorr

def _hash_challenge(R_bytes: bytes, X_bytes: bytes, phase: str, rnd: int, nonce_hex: str) -> int:
    """H = SHA256(R || X || phase || round || nonce_bytes) mod N"""
    h = hashlib.sha256()
    h.update(R_bytes)
    h.update(X_bytes)
    h.update(phase.encode("utf-8"))
    h.update(rnd.to_bytes(8, "big", signed=False))
    try:
        nonce_bytes = bytes.fromhex(nonce_hex)
    except ValueError:
        nonce_bytes = nonce_hex.encode("utf-8")
    h.update(nonce_bytes)
    return int.from_bytes(h.digest(), "big") % N


def _point_eq(P: ellipticcurve.Point, Q: ellipticcurve.Point) -> bool:
    return (P.x(), P.y()) == (Q.x(), Q.y())


def _point_from_compressed_hex(hx: str) -> ellipticcurve.Point:
    b = bytes.fromhex(hx)
    if len(b) != 33:
        raise ValueError(f"Pubkey non valida, lunghezza {len(b)} != 33")
    prefix = b[0]
    x_bytes = b[1:]
    x = int.from_bytes(x_bytes, "big")

    curve = SECP256k1.curve
    p = curve.p()
    a = curve.a()
    b_curve = curve.b()

    y2 = (x**3 + a*x + b_curve) % p
    y = pow(y2, (p + 1) // 4, p)
    if (y % 2 == 0 and prefix == 3) or (y % 2 == 1 and prefix == 2):
        y = p - y
    return ellipticcurve.Point(curve, x, y)


def _point_to_compressed_hex(P: ellipticcurve.Point) -> str:
    prefix = 2 + (P.y() & 1)
    x_bytes = int(P.x()).to_bytes(32, "big")
    return bytes([prefix]).hex() + x_bytes.hex()

class ZKPRegistry:
    """Mantiene mapping cid↔zkid, zkid→pubkey(Point) e nonces per round."""

    def __init__(self, zkid2pub_hex: Dict[str, str] | None = None, autoreg: bool = False) -> None:
        self.cid2zkid: Dict[str, str] = {}
        self.zkid2cid: Dict[str, str] = {}
        self.zkid2pub: Dict[str, ellipticcurve.Point] = {}
        self.fit_nonces: Dict[int, Dict[str, str]] = defaultdict(dict)
        self.eval_nonces: Dict[int, Dict[str, str]] = defaultdict(dict)
        self.auth_nonces: Dict[int, Dict[str, str]] = defaultdict(dict)
        self.autoreg = bool(autoreg)

        if zkid2pub_hex:
            zmap = zkid2pub_hex.get("clients", zkid2pub_hex)
            for zkid, hx in zmap.items():
                try:
                    self.zkid2pub[str(zkid)] = _point_from_compressed_hex(str(hx))
                except Exception:
                    pass

    # binding client ↔ zkid
    def register(self, cid: str, zkid: str) -> None:
        prev_for_cid = self.cid2zkid.get(cid)
        prev_for_zkid = self.zkid2cid.get(zkid)
        if prev_for_cid and prev_for_cid != zkid:
            raise RuntimeError(f"CID {cid} già legato a zkid {prev_for_cid}, non a {zkid}")
        if prev_for_zkid and prev_for_zkid != cid:
            raise RuntimeError(f"ZKID {zkid} già legato a cid {prev_for_zkid}, non a {cid}")
        self.cid2zkid[cid] = zkid
        self.zkid2cid[zkid] = cid

    def zkid_for(self, cid: str) -> Optional[str]:
        return self.cid2zkid.get(cid)

    def has_pubkey(self, zkid: str) -> bool:
        return zkid in self.zkid2pub

    def set_pubkey_if_allowed(self, zkid: str, pub_hex: Optional[str]) -> None:
        if self.has_pubkey(zkid) or not self.autoreg or not pub_hex:
            return
        try:
            self.zkid2pub[zkid] = _point_from_compressed_hex(pub_hex)
        except Exception:
            pass

    def pub_for(self, zkid: str) -> Optional[ellipticcurve.Point]:
        return self.zkid2pub.get(zkid)

    def pub_matches(self, zkid: str, pub_hex: str) -> bool:
        try:
            reg = self.zkid2pub.get(zkid)
            cur = _point_from_compressed_hex(pub_hex) if pub_hex else None
            return (reg is not None) and (cur is not None) and _point_eq(reg, cur)
        except Exception:
            return False

    def get_or_make_nonce(self, phase: str, rnd: int, zkid: str) -> str:
        import secrets
        table = (
            self.fit_nonces if phase == "fit" else
            self.eval_nonces if phase == "eval" else
            self.auth_nonces
        )
        n = table[rnd].get(zkid)
        if n is None:
            n = secrets.token_hex(16)
            table[rnd][zkid] = n
        return n

    def expected_nonce(self, phase: str, rnd: int, zkid: str) -> Optional[str]:
        table = (
            self.fit_nonces if phase == "fit" else
            self.eval_nonces if phase == "eval" else
            self.auth_nonces
        )
        return table.get(rnd, {}).get(zkid)

    def verify_schnorr(self, phase: str, rnd: int, zkid: str, R_hex: str, s_hex: str) -> bool:
        nonce = self.expected_nonce(phase, rnd, zkid)
        X = self.pub_for(zkid)
        if not nonce or X is None:
            return False
        try:
            R = _point_from_compressed_hex(R_hex)
            s = int(s_hex, 16) % N
        except Exception:
            return False
        c = _hash_challenge(bytes.fromhex(_point_to_compressed_hex(R)),
                            bytes.fromhex(_point_to_compressed_hex(X)),
                            phase, rnd, nonce)
        left = s * G
        right = R + c * X
        return _point_eq(left, right)

    def gc(self, current_round: int, keep_last: int = 1) -> None:
        for table in (self.fit_nonces, self.eval_nonces, self.auth_nonces):
            for r in list(table.keys()):
                if r < current_round - keep_last:
                    table.pop(r, None)
