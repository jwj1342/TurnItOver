import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.validate_tier_expansion import _baseline_digest, validate
from scripts.prepare_tier_review import prepare
from scripts.import_tier_review import import_return, digest


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    contract = {
        "status": "candidate_contract",
        "baseline_reuse_allowlist": {
            "allowed_tier_pairs": [["easy", "medium"]],
            "expected_shared_identities": 0,
            "identity_set_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "baseline_counts": {
            "easy": {"objects": 1, "cases": 2, "inventory_sha256": ""},
            "medium": {"objects": 0, "cases": 0, "inventory_sha256": ""},
            "hard": {"objects": 0, "cases": 0, "inventory_sha256": ""},
        },
        "frozen_evaluation_exclusion": {
            "objects": 0,
            "source_identity_set_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "tier_minima": {
            tier: {"objects": 2, "cases": 4, "train": 1, "validation": 1, "test": 0}
            for tier in ("easy", "medium", "hard")
        },
        "global_minima": {"independent_source_objects": 6},
    }
    objects = []
    index = 0
    for tier in ("easy", "medium", "hard"):
        for split in ("train", "validation"):
            source = f"PartNet-Mobility:{index}"
            membership = "new" if index else "baseline"
            cases = []
            for kind in ("clean", "defect"):
                case = {"case_id": f"{tier}-{index}-{kind}", "kind": kind,
                        "defect_id": None if kind == "clean" else "kinematics.joint_axis",
                        "membership": "added" if membership == "new" else "baseline"}
                if case["membership"] == "added":
                    case.update(
                        automatic_status="passed",
                        automatic_evidence_sha256="a" * 64,
                        difficulty_review={
                            "status": "passed",
                            "reviewer": "reviewer-1",
                            "reviewed_at": "2026-11-04",
                            "verdict": kind,
                            "difficulty": "clean" if kind == "clean" else tier,
                        },
                    )
                cases.append(case)
            objects.append({
                "tier": tier, "source_identity": source,
                "geometry_fingerprint": f"{index:064x}", "split": split,
                "original_split": split if membership == "baseline" else None,
                "membership": membership, "cases": cases,
            })
            index += 1
    def write_evidence(name, value):
        payload = json.dumps(value).encode()
        (tmp_path / name).write_bytes(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    for row in objects:
        for case in row["cases"]:
            if case["membership"] != "added":
                continue
            cid = case["case_id"]
            image = tmp_path / f'{cid}.png'
            Image.new('RGB', (256, 256), 'white').save(image)
            media = [{'path': image.name, 'sha256': digest(image)}]
            binding = {key: row[key] for key in ("source_identity", "geometry_fingerprint", "tier")}
            binding.update({key: case.get(key) for key in ("case_id", "kind", "defect_id")})
            automatic = write_evidence(f"{cid}.automatic.json", {
                "version": 1, "status": "passed", "binding": binding, "media": media})
            intake = tmp_path / f'{cid}.intake.json'
            intake.write_text(json.dumps([{'source_case_id': cid, 'bundle': str(tmp_path),
                'primary': [{'path': str(image), 'caption': '参考 / 初始视角 / rest'}]}]))
            pack, custody = tmp_path / f'{cid}.pack', tmp_path / f'{cid}.custody'
            result = prepare(intake, pack, custody, 'synthetic-test')
            opaque = json.loads((custody / 'mapping.json').read_text())['mapping'][0]['case_id']
            raw = tmp_path / f'{cid}.raw.jsonl'
            raw.write_text(json.dumps({'case_id': opaque, 'reviewer': 'reviewer-1',
                'decision': 'clean' if case['kind'] == 'clean' else row['tier'],
                'notes': 'Synthetic fixture observation', 'reviewed_at': '2026-11-04',
                'review_package_sha256': result['package_sha256']}) + '\n')
            metadata = tmp_path / f'{cid}.metadata.json'
            metadata.write_text(json.dumps([{'case_id': cid, 'bundle': str(tmp_path),
                'tier': row['tier'], 'kind': case['kind']}]))
            imported = tmp_path / f'{cid}.imported'
            import_return(raw, digest(raw), pack, result['package_sha256'],
                custody / 'mapping.json', digest(custody / 'mapping.json'), metadata,
                digest(metadata), imported)
            package = write_evidence(f"{cid}.package.json", {
                "version": 1, "case_id": cid, "media": media,
                'blind_package_sha256': result['package_sha256']})
            returned_path = imported / 'return-0000.json'
            returned = {'path': returned_path.relative_to(tmp_path).as_posix(), 'sha256': digest(returned_path)}
            case["automatic_evidence_sha256"] = automatic["sha256"]
            case["evidence"] = {"automatic_report": automatic, "review_package": package,
                                "review_return": returned}

    for tier in ("easy", "medium", "hard"):
        contract["baseline_counts"][tier]["inventory_sha256"] = _baseline_digest(
            [row for row in objects if row["tier"] == tier]
        )
    contract_path, manifest_path = tmp_path / "contract.json", tmp_path / "manifest.json"
    frozen_root = tmp_path / 'frozen-v2'
    frozen_root.mkdir()
    source_bindings, baseline_objects = {}, {}
    for row in objects:
        if row['membership'] == 'baseline':
            baseline_objects.setdefault(row['tier'], {})[row['source_identity']] = {
                'geometry_fingerprint': row['geometry_fingerprint']}
        else:
            spec = write_evidence(row['source_identity'].replace(':', '-') + '.spec.json', {'synthetic_source': row['source_identity']})
            archive = write_evidence(row['source_identity'].replace(':', '-') + '.zip', {'synthetic_archive': row['source_identity']})
            row['original_spec'] = spec
            source_bindings[row['source_identity']] = [{
                'spec_path': str(tmp_path / spec['path']), 'spec_sha256': spec['sha256'],
                'archive_path': str(tmp_path / archive['path']), 'archive_sha256': archive['sha256'],
                'geometry_fingerprint': row['geometry_fingerprint']}]
    pins = write_evidence('custody-pins.json', {
        'version': 1, 'algorithm': 'conversion-scale-audit-unique-rounded-world-vertices-float64-v1',
        'protected_files': [], 'frozen_root': str(frozen_root), 'frozen_files': {},
        'baseline_archives': [], 'frozen_objects': [], 'source_bindings': source_bindings,
        'baseline_objects': baseline_objects})
    contract['custody_pins_sha256'] = pins['sha256']
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "version": 1, "objects": objects, "frozen_evaluation_source_identities": [],
        "custody_pins": pins,
    }), encoding="utf-8")
    return manifest_path, contract_path


def test_accepts_pairing_split_minima_and_approved_additions(tmp_path):
    manifest, contract = _fixture(tmp_path)
    report = validate(manifest, contract)
    assert report["status"] == "passed"
    assert report["independent_source_objects"] == 6
    assert report["added_cases_with_automatic_and_human_pass"] == 10


@pytest.mark.parametrize("mutation, message", [
    ("missing_clean", "lacks a Clean or defect"),
    ("unreviewed", "lacks human difficulty approval"),
    ("duplicate_source", "Source identity crosses dataset splits"),
])
def test_rejects_unpaired_unreviewed_or_reused_additions(tmp_path, mutation, message):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    if mutation == "missing_clean":
        payload["objects"][1]["cases"] = payload["objects"][1]["cases"][1:]
    elif mutation == "unreviewed":
        payload["objects"][1]["cases"][0]["difficulty_review"] = "pending"
    else:
        payload["objects"][1]["source_identity"] = payload["objects"][0]["source_identity"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate(manifest, contract)


def test_rejects_modified_immutable_baseline(tmp_path):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["objects"][0]["cases"][0]["case_id"] = "rewritten-baseline-case"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable baseline inventory changed"):
        validate(manifest, contract)


def test_rejects_duplicate_geometry_across_new_objects(tmp_path):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["objects"][2]["geometry_fingerprint"] = payload["objects"][1]["geometry_fingerprint"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate geometry fingerprint"):
        validate(manifest, contract)


def test_rejects_new_source_in_frozen_v2_exclusion(tmp_path):
    import hashlib

    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    excluded = [payload["objects"][1]["source_identity"]]
    payload["frozen_evaluation_source_identities"] = excluded
    pinned = {
        "objects": 1,
        "source_identity_set_sha256": hashlib.sha256((excluded[0] + "\n").encode()).hexdigest(),
    }
    contract_data = json.loads(contract.read_text())
    contract_data["frozen_evaluation_exclusion"] = pinned
    contract.write_text(json.dumps(contract_data), encoding="utf-8")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps frozen evaluation v2"):
        validate(manifest, contract)


@pytest.mark.parametrize('mutation', ['missing_return', 'changed_media', 'wrong_geometry',
                                     'wrong_package', 'different_judgment', 'escaping_path'])
def test_rejects_unbound_or_modified_local_evidence(tmp_path, mutation):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    case = payload['objects'][1]['cases'][0]
    evidence = case['evidence']
    if mutation == 'missing_return':
        (tmp_path / evidence['review_return']['path']).unlink()
    elif mutation == 'changed_media':
        automatic = json.loads((tmp_path / evidence['automatic_report']['path']).read_text())
        (tmp_path / automatic['media'][0]['path']).write_text('changed')
    elif mutation == 'escaping_path':
        evidence['review_return']['path'] = '../external-return.json'
    else:
        reference = evidence['automatic_report' if mutation == 'wrong_geometry' else 'review_return']
        path = tmp_path / reference['path']
        document = json.loads(path.read_text())
        if mutation == 'wrong_geometry':
            document['binding']['geometry_fingerprint'] = 'f' * 64
        elif mutation == 'wrong_package':
            document['review_package_sha256'] = 'f' * 64
        else:
            document['difficulty_review']['reviewer'] = 'different-reviewer'
        content = json.dumps(document).encode()
        path.write_bytes(content)
        reference['sha256'] = hashlib.sha256(content).hexdigest()
        if mutation == 'wrong_geometry':
            case['automatic_evidence_sha256'] = reference['sha256']
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate(manifest, contract)


@pytest.mark.parametrize('mutation', ['missing_pins', 'modified_pins', 'original_spec',
                                     'source_archive', 'frozen_extra_file'])
def test_rejects_changed_custody_inputs(tmp_path, mutation):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    if mutation == 'missing_pins':
        del payload['custody_pins']
    elif mutation == 'modified_pins':
        (tmp_path / 'custody-pins.json').write_text('{}')
    elif mutation == 'original_spec':
        (tmp_path / payload['objects'][1]['original_spec']['path']).write_text('changed')
    elif mutation == 'source_archive':
        pins = json.loads((tmp_path / 'custody-pins.json').read_text())
        Path(next(iter(pins['source_bindings'].values()))[0]['archive_path']).write_text('changed')
    else:
        (tmp_path / 'frozen-v2' / 'new-file').write_text('added')
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate(manifest, contract)


@pytest.mark.parametrize('baseline', [True, False])
def test_rejects_added_case_on_frozen_geometry_even_with_another_identity(tmp_path, baseline):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    target = payload['objects'][0 if baseline else 1]
    if baseline:
        # Preserve original cases and add a case to the baseline object.
        added = dict(payload['objects'][1]['cases'][0])
        added['case_id'] = 'baseline-added-clean'
        target['cases'].append(added)
    pins_path = tmp_path / 'custody-pins.json'
    pins = json.loads(pins_path.read_text())
    pins['frozen_objects'] = [{'source_identity': 'PartNet-Mobility:different-id',
                               'geometry_fingerprint': target['geometry_fingerprint']}]
    content = json.dumps(pins).encode()
    pins_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    payload['custody_pins']['sha256'] = digest
    document = json.loads(contract.read_text())
    document['custody_pins_sha256'] = digest
    contract.write_text(json.dumps(document))
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='reuses frozen evaluated v2'):
        validate(manifest, contract)


@pytest.mark.parametrize('name', ['baseline-data.zip', 'frozen-v2/manifest.json'])
def test_rejects_rewritten_original_baseline_or_v2_bytes(tmp_path, name):
    manifest, contract = _fixture(tmp_path)
    original = tmp_path / name
    original.write_bytes(b'original frozen bytes')
    pin_path = tmp_path / 'custody-pins.json'
    pins = json.loads(pin_path.read_text())
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    pins['protected_files'].append({'path': str(original), 'sha256': digest})
    if name.startswith('frozen-v2/'):
        pins['frozen_files']['manifest.json'] = digest
    content = json.dumps(pins).encode()
    pin_path.write_bytes(content)
    pin_hash = hashlib.sha256(content).hexdigest()
    payload = json.loads(manifest.read_text())
    payload['custody_pins']['sha256'] = pin_hash
    manifest.write_text(json.dumps(payload))
    document = json.loads(contract.read_text())
    document['custody_pins_sha256'] = pin_hash
    contract.write_text(json.dumps(document))
    original.write_bytes(b'changed frozen bytes')
    with pytest.raises(ValueError, match='Protected baseline/v2 bytes changed'):
        validate(manifest, contract)


def test_rejects_cross_split_baseline_reuse_even_when_tier_pair_allowed(tmp_path):
    manifest, contract = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    original = payload['objects'][0]
    duplicate = payload['objects'][3]
    duplicate.update(source_identity=original['source_identity'],
                     geometry_fingerprint=original['geometry_fingerprint'],
                     membership='baseline', original_split='validation')
    for case in duplicate['cases']:
        case['membership'] = 'baseline'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Source identity crosses dataset splits'):
        validate(manifest, contract)


def test_explicit_missing_source_wave_fails_before_custody_scan(tmp_path):
    from scripts.build_tier_expansion_pins import build

    with pytest.raises(ValueError, match='Explicit download manifest is missing'):
        build(tmp_path, tmp_path / 'pins.json', download_manifests=[tmp_path / 'missing.json'])
    with pytest.raises(ValueError, match='Explicit conversion catalog is missing'):
        build(tmp_path, tmp_path / 'pins.json', catalog_roots=[tmp_path / 'missing-catalog'])
