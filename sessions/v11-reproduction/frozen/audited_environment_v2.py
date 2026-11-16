"""Preserve the audit while explicitly rejecting nonfinite candidate geometry."""
import numpy as np
from audited_environment import RepairEnvironment as PreviousEnvironment

class RepairEnvironment(PreviousEnvironment):
    def audit(self,source):
        try:return super().audit(source)
        except ValueError as exc:
            if str(exc)!='Probabilities contain NaN':raise
            trace=exc.__traceback__;evidence=None
            while trace:
                code=trace.tb_frame.f_code
                if code.co_filename.replace('\\','/').endswith('/turnitover/checkers/geometry.py') and code.co_name=='check':
                    evidence=trace.tb_frame.f_locals.get('evidence')
                trace=trace.tb_next
            if evidence is None or not any(not np.isfinite(mesh.vertices).all() for state in evidence.geometry.values() for mesh in state.values()):
                raise
            return {'usable':False,'passed':False,'score':0.,'error':'NonfiniteCandidateGeometry',
                    'scope':'configured_states_geometry_and_runtime',
                    'detail':'Candidate exports nonfinite world-space mesh coordinates.'}
