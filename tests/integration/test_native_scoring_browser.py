"""Analytic smoke programs on the existing RP29 browser harness, no model API."""
import hashlib
import math
import os
from pathlib import Path

import pytest

from scripts.rp29_native_replay import replay_role
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.render.session import ObservationSession, RenderConfig
from turnitover.render.views import load_views
from turnitover.repair.native_evaluator import measure_replay

pytestmark = pytest.mark.browser


@pytest.fixture
def native_session(monkeypatch):
    root_value = os.environ.get('TIO_NATIVE_SCORING_ROOT')
    if not root_value:
        pytest.skip('Set TIO_NATIVE_SCORING_ROOT to the existing native RP29 checkout/dist')
    root = Path(root_value)
    executable = os.environ.get('TIO_BROWSER_EXECUTABLE')
    if executable:
        from playwright.sync_api import BrowserType
        launch = BrowserType.launch
        monkeypatch.setattr(BrowserType, 'launch',
                            lambda self, **kw: launch(self, **{**kw, 'executable_path': executable}))
    esbuild = Path(os.environ.get('TIO_ESBUILD', root / 'web/node_modules/.bin/esbuild'))
    # The older local session already owns browser transport and rendering.
    # These test-only compatibility calls use the native harness methods;
    # no alternative scene executor or production protocol is introduced.
    class NativeSession(ObservationSession):
        def advance_time(self, time):
            return self._call('advanceTime', {'time_seconds': time})
        def dispose_program(self):
            return self._call('disposeProgram')
    with NativeSession(RenderConfig(root/'web/dist', esbuild, width=128, height=128),
                       load_views(root/'configs/views.yaml')) as session:
        yield session


def program(*, metalness=0., scale=1., other=1., dispose=True, tick=None, duplicate=False,
            segments=1, uv_scale=1.):
    setter = "if(v>0)wasMoved=true;if(v===0&&wasMoved&&!clone){clone=new THREE.Mesh(geometry,material);clone.name='late#mesh';part.add(clone);}"
    timing = {'absolute': 'part.position.z=t;', 'phase': 'part.position.z=t*2;',
              'delta': 'part.position.z+=t;'}
    tick_code = ',tick:(t:number)=>{' + timing[tick] + '}' if tick else ''
    dispose_code = ',dispose:()=>{geometry.dispose();material.dispose();texture.dispose();}' if dispose else ''
    return f'''export default function(THREE:any) {{
      const root=new THREE.Group(),part=new THREE.Group(),other=new THREE.Group();
      part.name='part';other.name='other';other.position.x={other};root.add(part,other);
      const geometry=new THREE.BoxGeometry(1,1,1,{segments},{segments},{segments});
      geometry.scale({scale},1,1);
      const uv=geometry.getAttribute('uv');
      for(let i=0;i<uv.count;i++)uv.setXY(i,uv.getX(i)*{uv_scale},uv.getY(i));
      const texture=new THREE.DataTexture(new Uint8Array([128,128,255,255]),1,1);
      texture.needsUpdate=true;
      const material=new THREE.MeshStandardMaterial({{roughness:.5,metalness:{metalness},normalMap:texture}});
      const mesh=new THREE.Mesh(geometry,material);mesh.name='part#mesh';part.add(mesh);
      const mesh2=new THREE.Mesh(geometry,material);mesh2.name='other#mesh';other.add(mesh2);
      let wasMoved=false,clone:any=null;
      const joints={{joint:{{type:'revolute',part:'part',limits:[0,1],
        set:(v:number)=>{{{setter if duplicate else ''}}}}}}};
      return {{root,joints{dispose_code}{tick_code}}};
    }}'''


def replay(session, tmp_path, reference, candidate, **kwargs):
    result = {'runtime': [], 'private_evidence': {}, 'gate': {'passed': True},
        'program_sha256': {role: hashlib.sha256(source.encode()).hexdigest()
                          for role, source in [('reference', reference), ('candidate', candidate)]}}
    case = {'view': 'front', 'commands': [[state, {'joint': value}] for state, value
                in zip(('zero', 'mid', 'limit'), (0., .5, 1.))], **kwargs}
    for role, source in [('reference', reference), ('candidate', candidate)]:
        replay_role(session, ObjectProgram(source), Framing(center=(0., 0., 0.), radius=2.),
                    case, role, tmp_path, result)
    return result


def score(evidence, kind, target='part', **kwargs):
    labels = [{'defect_id': kind, 'parts': [target]}]
    scope = {'defect_id': kind, 'reference_program_sha256': evidence['program_sha256']['reference'],
        'provenance': [{'path': 'analytic-browser-fixture', 'sha256': 'b' * 64}],
        'target': {'part': target}, **kwargs}
    return measure_replay(evidence, labels, {'max_triangles': 1000, 'max_draw_calls': 20}, [scope])


def test_real_gpu_material_readback_and_lifecycle(native_session, tmp_path):
    evidence = replay(native_session, tmp_path, program(), program(metalness=1.))
    result = score(evidence, 'material.metalness')
    assert result['measurements'][0]['passed'] is False
    assert result['checks']['material.properties'] is False
    assert result['checks']['runtime.lifecycle'] is True
    import json
    ref = json.loads(Path(evidence['private_evidence']['reference']['path']).read_text())
    gpu = ref['states']['zero']['readback']['meshes'][0]['materials'][0]['gpu']
    assert gpu['program_present'] and gpu['uses_normalmap']
    assert 'positions_b64' not in json.dumps(evidence['runtime'])


def test_noop_dispose_is_not_mistaken_for_a_present_callback(native_session, tmp_path):
    bad = program().replace('geometry.dispose();material.dispose();texture.dispose();', '')
    result = score(replay(native_session, tmp_path, program(), bad), 'runtime.missing_dispose')
    assert result['measurements'][0]['passed'] is False


@pytest.mark.parametrize('uv_scale,passed', [(1., True), (2., False)])
def test_remeshed_repair_through_complete_native_scoring(native_session, tmp_path, uv_scale, passed):
    import json
    result = score(replay(native_session, tmp_path, program(), program(segments=2, uv_scale=uv_scale)),
                   'geometry.thin_sheet', tolerance_m=0.)
    assert result['measurements'][0]['passed'] is True
    assert result['checks']['material.properties'] is passed
    if passed:
        assert all(value is True for value in result['checks'].values()), result['checks']
    assert json.loads(json.dumps(result))['checks'] == result['checks']


@pytest.mark.parametrize('kind,kwargs', [
    ('geometry.thin_sheet', {'scale': .02}),
    ('structure.floating', {'other': 1.3}),
    ('structure.interpenetration', {'other': .7}),
])
def test_current_native_geometry_defects(native_session, tmp_path, kind, kwargs):
    result = score(replay(native_session, tmp_path, program(), program(**kwargs)), kind,
                   support={'part': 'other'}, tolerance_m=0)
    assert result['measurements'][0]['passed'] is False


@pytest.mark.parametrize('mode,passed', [('absolute', True), ('phase', False), ('delta', False), (None, False)])
def test_native_tick_phase_and_absolute_history(native_session, tmp_path, mode, passed):
    result = score(replay(native_session, tmp_path, program(tick='absolute'), program(tick=mode),
                          time_sequence=[0., .5, 1.]), 'runtime.tick_missing_or_phase')
    assert result['measurements'][0]['passed'] is passed


def test_out_and_back_duplicate_uses_private_current_geometry(native_session, tmp_path):
    result = score(replay(native_session, tmp_path, program(), program(duplicate=True)),
                   'geometry.duplicate_part')
    assert result['measurements'][0]['passed'] is False


@pytest.mark.parametrize('case_id', [
    'rp29-fa857b20c5745d6e223f', 'rp29-9985185022d4233e5c02',
    'rp29-balanced-98e750bf18c5601fc11d', 'rp29-balanced-62bf1b7393ca638c873c',
])
def test_frozen_validation_source_corruption_and_reference_restoration(native_session, tmp_path, case_id):
    """Engineering source check, not a repair/model run or a paper metric."""
    import json
    from turnitover.repair.native_evaluator import _mesh_export
    from turnitover.repair.native_geometry import measure_geometry
    from turnitover.repair.native_scopes import construction_scopes
    annotations = os.environ.get('TIO_NATIVE_ANNOTATIONS')
    cache = os.environ.get('TIO_NATIVE_SOURCE_CACHE')
    if not annotations or not cache:
        pytest.skip('Set explicit Train/Validation annotations and source cache for source smoke')
    rows = [json.loads(line) for line in Path(annotations).read_text(encoding='utf-8').splitlines()]
    row = next(r for r in rows if r.get('source_roster_row', {}).get('split') == 'validation'
               and r['id'] == case_id)
    prefix = row['source_roster_row']['report_path'].split('/output/', 1)[0] + '/'
    def resolve(path):
        if not path.startswith(prefix):
            raise ValueError('Source smoke path outside the explicitly selected native cache')
        return Path(cache) / path[len(prefix):]
    sources = {}
    for role, key in [('reference', 'case_reference'), ('candidate', 'case_candidate')]:
        item = row['programs'][key]
        raw = resolve(item['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        sources[role] = raw.decode('utf-8')
    native_session.load(ObjectProgram(sources['reference']), None)
    frame = native_session.info.framing
    semantics = native_session._page.evaluate(
        '()=>window.harness.object.root.userData.nativeJointSemantics?.joints ?? {}')
    commands = [[state, {key: ((1 - fraction) * joint['limits'][0] + fraction * joint['limits'][1]
                              if semantics.get(key, {}).get('native_type') != 'continuous'
                              and all(math.isfinite(x) for x in joint['limits']) else 0.)
                        for key, joint in native_session.info.joints.items()}]
                for state, fraction in [('zero', 0.), ('mid', .5), ('limit', 1.)]]
    native_session.dispose_program()
    # Detents are engineering smoke coordinates only; formal runs retain the
    # separately hash-frozen authored observation recipe.
    case = {'view': 'front', 'commands': commands}
    evidence = {'runtime': [], 'private_evidence': {}, 'gate': {'passed': True},
        'program_sha256': {role: hashlib.sha256(code.encode()).hexdigest() for role, code in sources.items()}}
    docs = {}
    for role in ('reference', 'candidate'):
        replay_role(native_session, ObjectProgram(sources[role]), frame, case, role, tmp_path, evidence)
        docs[role] = json.loads(Path(evidence['private_evidence'][role]['path']).read_text())
    bound = construction_scopes(row, evidence['program_sha256']['reference'], resolve=resolve)
    histories = {role: {state: _mesh_export(snapshot['geometry']) for state, snapshot in doc['states'].items()}
                 for role, doc in docs.items()}
    label, source_scope = bound['labels'][0], bound['scopes'][0]
    assert source_scope is not None
    broken = measure_geometry(label['defect_id'], source_scope, histories, evidence['program_sha256'])
    assert broken['passed'] is False, broken
    restored = measure_geometry(label['defect_id'], source_scope,
        {**histories, 'candidate': histories['reference']}, evidence['program_sha256'])
    assert restored['passed'] is True, restored
