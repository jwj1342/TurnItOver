# Interpretation limits fixed for this collection

The run tests active observation and stopping within each human-reviewed object's
normal-selected joint and fixed camera. It does not test free camera search or
discovery among arbitrary joints. The selected joint, moving part, normal joint
type and reference state labels are public, identical across variants of an
object, and not evidence of the injected defect. Correct joint/part fields here
do not establish unrestricted localization capability.

The task text calls its declared scope `easy_selected_joint_motion_v1`. It asks
the teacher to compare candidate/reference rest, mid and limit, uses the full
taxonomy, and does not tell it that a defect exists. However, it does not
explicitly give static installation alignment equal emphasis with motion. Some
recorded pass explanations focus on motion consistency. Thus offset misses are
strict failures relative to the requested Gold task, with scope wording as a
confound; they cannot by themselves isolate a visual-perception limitation.
The prompt and action space remain unchanged throughout the eighteen cases.

The existing independent evidence rule accepts only cited observations which
show a perceptible discrepancy from a compatible reference. An unchanged rest
frame can provide baseline context, but citing it inside an axis finding fails
that strict rule. No cited steps are silently pruned to improve acceptance.

The original auditor recognizes static/runtime pass scopes. The local adapter
adds a complete selected-joint motion scope for genuinely clean cases. A wrong
pass on a defective case still contains the legacy secondary
`pass_scope_not_supported` reason alongside `pass_contradicts_objective_defect`.
When all three states are actually covered, the substantive failure is the
wrong pass, not a lack of observation. This redundant legacy reason is preserved
in raw audit output and must not be counted as a separate failed Episode.

Action supervision is qualified separately from diagnosis: a legal, new,
frozen-bound, reference-comparable observation can survive an incorrect or
missing terminal diagnosis. Such rows demonstrate valid acquisition behavior,
not optimal action choice or a proven advantage over a fixed script. All
teacher/student prompt and image prefixes are checked for exact parity.

Provider output tokens include hidden reasoning. If the provider returns no
text after consuming the limit, the original error can still contain usage.
The accounting summary recovers only that retained usage and preserves the
original Episode and call metadata. Unknown usage or billing is not assumed zero.
