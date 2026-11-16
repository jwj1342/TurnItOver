"""Private, scope-bound clean certificates for the three-tier v1.1 release."""
import hashlib
import json

POLICY = 'three-tier-900-clean-pass-v1.1'


def observation_digest(record):
    value = {k: record[k] for k in ('episode_id', 'images', 'image_sha256', 'data_identity')}
    value['user'] = record['messages'][0]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_clean_pass(record, annotation):
    from turnitover.oracle.supervision_validation import public_payload
    cert = record.get('pass_certificate', {})
    identity = record['data_identity']
    payload = public_payload(record)
    if (record.get('label_policy') != POLICY or annotation.get('label_policy') != POLICY
            or record.get('is_clean') is not True or annotation.get('is_clean') is not True
            or annotation['labels'] or record.get('admission') != 'eligible'
            or annotation.get('admission') != 'eligible'
            or not identity.get('reference_sha')
            or identity['reference_sha'] != identity.get('candidate_sha')
            or annotation['reference_sha'] != identity['reference_sha']
            or annotation['candidate_sha'] != identity['candidate_sha']
            or annotation.get('pass_certificate') != cert
            or cert.get('policy') != POLICY
            or cert.get('basis') != 'published_reference_identity_and_complete_frozen_observation'
            or cert.get('scope') != 'published_structure_kinematics_fixed_observation_task'
            or cert.get('observation_sha256') != observation_digest(record)
            or cert.get('case_id') != record['episode_id']
            or cert.get('reference_sha') != identity['reference_sha']
            or not record.get('terminal') or payload.get('remaining') != 0
            or not payload['history'] or len(record['images']) < 2
            or record.get('evidence', {}).get('covered_label_indices')):
        raise ValueError('These exports do not establish positive pass evidence')


def clean_verdict():
    return {'verdict': {
        'status': 'pass', 'confidence': 1.0, 'findings': [],
        'summary': 'The candidate passes the prescribed reference comparison within the observed task scope.',
        'limitations': ['This verdict is limited to the supplied views and states; it does not certify universal correctness.']}}
