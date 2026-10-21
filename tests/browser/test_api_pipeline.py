"""Real-browser integrity checks; scripted responses here are NOT experiment results."""
import pytest

from turnitover.api_pipeline.fixture import controlled_cases, reference_program
from turnitover.api_pipeline.scoring import quality, replay_witnesses

pytestmark = pytest.mark.browser


def test_controlled_faults_and_independent_witness_replay(render_config, views):
    reference = reference_program()
    cases = controlled_cases(reference)
    assert quality(cases['c00'], reference, render_config, views)['code_correct'] is True
    assert quality(cases['c01'], reference, render_config, views)['code_correct'] is False
    assert quality(cases['c02'], reference, render_config, views)['code_correct'] is False
    # The hidden door-axis fault has no rest-pose mismatch; actuation is required.
    trajectory = [
        {'step': 0, 'action': {'__type__': 'RequestView', 'view_id': 'front'}, 'image_ref': '0.png'},
        {'step': 1, 'action': {'__type__': 'ActuateJoint', 'joint_id': 'door_hinge', 'detent': 'limit'},
         'image_ref': '1.png'},
    ]
    witnesses = replay_witnesses(cases['c02'], reference, trajectory, render_config, views)
    assert witnesses[0]['mismatched_parts'] == []
    assert witnesses[1]['mismatched_parts'] == ['door']
