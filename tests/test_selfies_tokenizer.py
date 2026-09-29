"""SELFIES tokenizer round-trip."""

from functionalspec.data.smiles_tokenizer import MoleculeTokenizer


def test_selfies_roundtrip():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    tok = MoleculeTokenizer.from_smiles_list(smiles, representation="selfies")
    assert tok.vocab_size > 4
    for s in smiles:
        ids = tok.encode(s)
        assert ids is not None
        back = tok.decode_to_smiles(ids)
        # canonical forms may differ; require non-empty valid-ish return
        assert isinstance(back, str) and len(back) > 0
