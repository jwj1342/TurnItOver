# Hard full-motion formal v1.1

`hard-full-motion-formal-v1.1` preserves the immutable 75-object
`single_motion_v1` baseline and adds 75 independently sourced
`compositional_v1_1` objects.  The two tracks are recorded per case and must
be reported separately.

Each compositional case has exactly two declared field faults on one moving
part.  Supported signatures are the six pairs of `joint_axis`,
`joint_anchor`, `part_offset`, and `part_rotation`.  Each fault has an
Easy/Medium-range parameter; Hard difficulty is the need to diagnose both,
not a large-motion source gate.  The combined candidate must change exactly
the two declared fields, retain source authenticity, and have complete normal
and candidate checker/render evidence.  Individual atom screening is not an
admission requirement.

The candidate bundle has 150 source-unique objects and 300 cases (one Clean
control and one defect candidate per object), split 100/20/30 for
Train/Validation/Test.  All cases remain `pending_human_review` and the train
manifest remains empty until review and a separate eligibility freeze.
