from turnitover.api_pipeline.scoring import diagnosis_metrics


def test_finding_must_match_independent_target_and_acquired_geometric_witness():
    target = [{'defect_id': 'kinematics.joint_axis', 'parts': ['door'], 'joint_id': 'door_hinge'}]
    finding = {**target[0], 'evidence_steps': [0]}
    result = {'status': 'complete', 'verdict': {'status': 'fail', 'findings': [finding]},
              'spent': 1, 'trajectory': [], 'model_calls': []}
    witness = {'step': 0, 'image_acquired': True, 'mismatched_parts': []}
    score = diagnosis_metrics(result, target, [witness])
    assert score['detection_correct'] and score['type_exact']
    assert score['part_localization'] == {'tp': 1, 'fp': 0, 'fn': 0}
    assert score['citations_exist'] and score['geometric_witness_consistent_findings'] == 0
    witness['mismatched_parts'] = ['door']
    assert diagnosis_metrics(result, target, [witness])['geometric_witness_consistent_findings'] == 1
    witness['image_acquired'] = False
    assert diagnosis_metrics(result, target, [witness])['geometric_witness_consistent_findings'] == 0


def test_general_generation_prompt_uses_only_supplied_public_task():
    from turnitover.api_pipeline.fixture import generation_prompt
    prompt = generation_prompt('public ABI', 'Build a red kettle.')
    assert 'Build a red kettle.' in prompt and 'public ABI' in prompt
    assert 'drawer_slide' not in prompt and 'door_hinge' not in prompt
    assert 'BoxGeometry' not in prompt
