"""Axis observability (Table 3) matches configs/acoustic_targets.yaml."""
from sparc_als.utils.io import load_yaml


def test_axis_observability_matches_table_3():
    cfg = load_yaml("configs/acoustic_targets.yaml")
    axes = cfg["axes"]
    # Phonatory stability: ✓ / ✓ / --
    assert axes["phonatory_stability"]["observed"]["voc_als"] is True
    assert axes["phonatory_stability"]["observed"]["minsk2020"] is True
    assert axes["phonatory_stability"]["observed"]["torgo"] is False
    # Articulatory precision: ✓ / ✓^a (distance only) / ✓^b (aligned)
    assert axes["articulatory_precision"]["observed"]["voc_als"] is True
    assert axes["articulatory_precision"]["observed"]["minsk2020"] == "distance_only"
    assert axes["articulatory_precision"]["observed"]["torgo"] == "aligned"
    # Prosodic / timing: ✓ / -- / --
    assert axes["prosodic_timing"]["observed"]["voc_als"] is True
    assert axes["prosodic_timing"]["observed"]["minsk2020"] is False
    assert axes["prosodic_timing"]["observed"]["torgo"] is False
    # Respiratory support: ✓ / ✓ / --
    assert axes["respiratory_support"]["observed"]["voc_als"] is True
    assert axes["respiratory_support"]["observed"]["minsk2020"] is True
    assert axes["respiratory_support"]["observed"]["torgo"] is False
