import torch

from functionalspec.data.motifs import MotifVocabulary, PAD_MOTIF
from functionalspec.models.generator_b import LearnedMotifDecoder


def test_brics_motif_targets_are_fixed_length():
    vocab = MotifVocabulary.from_smiles(["CCO", "c1ccccc1O"], max_size=16, max_motifs=4)
    encoded = vocab.encode("CCO")
    assert len(encoded) == 4
    assert all(isinstance(value, int) for value in encoded)
    assert encoded[-1] == PAD_MOTIF


def test_arm_b_forward_backward():
    model = LearnedMotifDecoder(
        cond_dim=12,
        motif_vocab_size=10,
        atom_vocab_size=16,
        motif_codebook=8,
        motif_dim=6,
        max_motifs=3,
        atom_hidden=12,
        atom_layers=1,
    )
    flat_s = torch.randn(2, 12)
    tokens = torch.randint(0, 16, (2, 5))
    out = model(flat_s, tokens)
    assert out["fragment_logits"].shape == (2, 3, 10)
    assert out["atom_logits"].shape == (2, 5, 16)
    loss = out["atom_logits"].mean() + out["fragment_logits"].mean() + out["motif_vq_loss"]
    loss.backward()
    assert model.query.grad is not None
    sampled, motif_ids = model.sample(flat_s, max_len=4)
    assert sampled.shape[0] == 2
    assert motif_ids.shape == (2, 3)
