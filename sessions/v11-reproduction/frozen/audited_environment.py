"""Turn malformed exported mesh arrays into an explicit failed private audit."""
import re
from turnitover.repair.environment import RepairEnvironment as BaseEnvironment

class RepairEnvironment(BaseEnvironment):
    def audit(self,source):
        try:
            return super().audit(source)
        except ValueError as exc:
            trace=exc.__traceback__;frames=[]
            while trace:
                frames.append((trace.tb_frame.f_code.co_filename.replace('\\','/'),trace.tb_frame.f_code.co_name))
                trace=trace.tb_next
            recognized=any(path.endswith('/turnitover/render/protocol.py') and name in ('decode_indices','decode_positions') for path,name in frames)
            if not recognized or not re.fullmatch(r'cannot reshape array of size \d+ into shape \(3\)',str(exc)):
                raise
            return {'usable':False,'passed':False,'score':0.,'error':'MalformedGeometryArray',
                    'scope':'configured_states_geometry_and_runtime',
                    'detail':'Mesh export cannot form complete 3D vertices or triangle index triples.'}
