"""Reference-free, bounded CPU relation inference and conservative fusion.

This runs after Homr has exited, using the same photograph and segmentation.
No Torch, reference MusicXML, renderer or scorer is needed at inference.
"""
import hashlib,json,sys,time
from pathlib import Path
import xml.etree.ElementTree as E
import cv2
from route_trials.runtime_inference import save
from route_trials.testset3_photo_cache import cached_photo_reads
from route_trials.relation_page_inputs import cached_relation_inputs, geometry_json, parse_score, photo_digest
from route_trials.relation_session_reuse import scoped_cpu_sessions, validated_manifest_for


def sha(path):return photo_digest(path)


def validate_models(models):
    models=Path(models);manifest=json.loads((models/'manifest.json').read_text(encoding='utf8'))
    required={'symbol.onnx','stroke.onnx','edge.onnx','grace.onnx','symbol_scores.json','stroke_scores.json','edge_scores.json','grace_training.json'}
    files=manifest['files']
    if {f['path'] for f in files}!=required:raise ValueError('Incomplete relation model manifest')
    for f in files:
        path=models/f['path']
        if not path.is_file() or path.stat().st_size!=f['bytes'] or sha(path)!=f['sha256']:raise ValueError('Relation model integrity failed: '+f['path'])
    return manifest


def decode_and_fuse(photo,staff_geometry,production,cache,models,output,local_accidental_alignment=True,partial_pitch_alignment=True,independent_chord_anchor=True,local_tuplet_spans=True,tuplet_crossings=True,geometry_tuplet_alignment=True,conflicting_tuplet_ratio=True,opposing_voice_partition=True,global_ottava_indices=True,expanded_rhythm=True,ink_backed_voice_heads=True,joint_staff_rhythm=True,coverage_repairs=True,tuplet_scope=True,identity_repairs=True,measure_joint=True,expanded_joint_candidates=True,structural_decode=True,recognized_headers=None):
    """Only predicted image geometry/graph and candidate XML are inputs."""
    from route_trials.relation_v2 import redecode
    from route_trials.relation_ensemble import arbitrate
    from route_trials.refine_candidate import refine
    models=Path(models);output=Path(output);photo=Path(photo)
    structural_decode=structural_decode and measure_joint and expanded_joint_candidates
    start=time.perf_counter();validate_models(models)
    payload=json.loads(Path(cache).read_text(encoding='utf8'))
    if payload.get('image_sha256')!=sha(photo):raise ValueError('Relation graph does not certify this photograph')
    proposals=output.with_suffix('.proposals.json')
    save(proposals,{'image_sha256':sha(photo),'coordinate_scale':16/19.8,'predictions':payload.get('head_proposals',[]),'reference_supplied_to_recognizer':False})
    relation=output.with_suffix('.relation.musicxml')
    redecode(photo,Path(staff_geometry),Path(recognized_headers or production),Path(cache),relation,'all',root=models,backend='onnx',
             refinement={'rescue':proposals,'attach_graces':True,'voice_continuity':True,'join_grace_chords':True,'visual_tuplets':True,'local_tuplet_spans':local_tuplet_spans,'tuplet_crossings':tuplet_crossings,'expanded_rhythm':expanded_rhythm,'identity_repairs':identity_repairs})
    arbitration=arbitrate(Path(production),relation,relation.with_suffix('.graph.json'),photo,Path(staff_geometry),output)
    base=parse_score(output).getroot();other=parse_score(relation).getroot();graph=json.loads(relation.with_suffix('.graph.json').read_text(encoding='utf8'))
    # Voice/grace merging must retain every ordinary event on its time axis.
    from route_trials.refine_candidate import records
    def clocks(root):return [Counter(r['key'] for r in row if not r['grace']) for row in records(root)]
    from collections import Counter
    before=clocks(base);report=refine(base,other,graph)
    if before!=clocks(base):raise ValueError('Relation refinement changed ordinary clocks')
    from route_trials.relation_tuplets import fuse as fuse_tuplets
    report['tuplets']=fuse_tuplets(base,graph,photo,geometry_json(staff_geometry),geometry_alignment=geometry_tuplet_alignment,replace_conflicting_ratio=conflicting_tuplet_ratio,expanded_rhythm=expanded_rhythm)
    report['expanded_rhythm']={'enabled':expanded_rhythm,'maximum_beam_layers':2,'minimum_template_score':.86,'minimum_class_margin':.08,'weak_prefix_proposal_requires_rest_and_complete_unique_voice':True}
    report['tuplet_digit_crossings']={'enabled':tuplet_crossings,'mask_kernel_pixels':[5,3],'minimum_template_score':.86,'minimum_class_margin':.08}
    report['tuplets_geometry_alignment']={'enabled':geometry_tuplet_alignment,'maximum_staff_step_residual':.65,'existing_pitch_retained':True,'maximum_groups_per_lane':64}
    report['conflicting_tuplet_ratio']={'enabled':conflicting_tuplet_ratio,'requires_witnessed_conflicting_ratio_and_complete_printed_meter':True}
    before_partition=clocks(base)
    from route_trials.relation_voice_partition import fuse as fuse_voices
    report['opposing_voice_partition']=fuse_voices(base,graph,photo,ink_backed_heads=ink_backed_voice_heads,stable_secondary_voice=measure_joint and expanded_joint_candidates) if opposing_voice_partition else []
    report['secondary_voice_identity']={'stable_support_enabled':measure_joint and expanded_joint_candidates,'multiple_labels_require_unique_largest_ordinary_note_support':True,'minimum_multiple_label_support':4}
    report['ink_backed_voice_heads']={'enabled':ink_backed_voice_heads,'minimum_detector_confidence':.7,'minimum_low_confidence_filled_core':.95,'minimum_low_confidence_stem_confidence':.97}
    if before_partition!=clocks(base):raise ValueError('Voice partition changed ordinary clocks')
    report['voice_partition_clocks_preserved']=True
    report['local_tuplet_spans']={'enabled':local_tuplet_spans,'maximum_windows_per_chain':128,
        'overlapping_interpretations':'abstain','proof':'individual_printed_number_continuous_beam_and_complete_matching_voice'}
    from omr_event_records import records as event_records
    def identities(root):return [Counter((r['staff'],r['voice'],r['midi'],r['grace']) for r in row) for row in event_records(root)]
    identities_before=identities(base)
    from route_trials.relation_joint_rhythm import fuse as fuse_joint_rhythm
    report['joint_staff_rhythm']=fuse_joint_rhythm(base,graph,photo,geometry_json(staff_geometry)) if joint_staff_rhythm else []
    report['joint_staff_rhythm_limits']={'enabled':joint_staff_rhythm,'maximum_staves':2,'maximum_groups_per_lane':64,'maximum_events_per_measure':128,'maximum_jobs':128,'maximum_divisions_factor':12,'minimum_rest_pair_cells':3,'maximum_target_beam_gap_spaces':4,'maximum_partner_beam_gap_spaces':8,'partner_beam_samples':7,'maximum_cross_staff_column_residual_spaces':.3,'requires_unique_complete_partner_and_full_ink_backed_binding':True}
    if identities_before!=identities(base):raise ValueError('Joint rhythm changed pitch, staff, voice or cardinality')
    def time_voices(root):return [Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['grace']) for r in row) for row in event_records(root)]
    temporal_before=time_voices(base)
    from route_trials.relation_pitch import fuse_low_octaves
    report['pitch_repairs']=fuse_low_octaves(base,graph,photo,geometry_json(staff_geometry),partial_measures=partial_pitch_alignment,independent_partner=independent_chord_anchor)
    report['partial_pitch_alignment']={'enabled':partial_pitch_alignment,'maximum_columns':64,
        'extension_requires_matching_onset_duration':True,'legacy_repairs_retained':True,
        'proof':'unchanged_octave_partner_staff_track_head_core_and_continuous_ink_stem'}
    report['independent_chord_anchor']={'enabled':independent_chord_anchor,
        'proof':'retained_production_anchor_two_staff_positions_two_ink_heads_and_shared_stem',
        'target_minimum_confidence':.92,'anchor_minimum_confidence':.8,
        'time_voice_and_cardinality_preserved':True,'legacy_repairs_retained':True}
    if temporal_before!=time_voices(base):raise ValueError('Pitch refinement changed time or voice')
    from route_trials.relation_notation import fuse_ottavas
    from route_trials.relation_accidentals import fuse_accidentals
    geometry=geometry_json(staff_geometry)
    report['ottava_repairs']=fuse_ottavas(base,graph,photo,geometry,global_group_indices=global_ottava_indices)
    report['ottava_group_indices']={'global_row_indices_enabled':global_ottava_indices}
    report['accidental_repairs']=fuse_accidentals(base,graph,photo,geometry,local_alignment=local_accidental_alignment)
    report['local_accidental_alignment']={'enabled':local_accidental_alignment,'maximum_columns':64,
        'ambiguous_pairs':'abstain','state_across_unmatched_columns':'cleared',
        'incomplete_graph_seed':'requires_existing_pitch_agreement'}
    if temporal_before!=time_voices(base):raise ValueError('Notation refinement changed time or voice')
    from route_trials.relation_coverage import fuse_key_context,fuse_written_low_octaves,fuse_isolated_melody
    report['coverage_repairs']={}
    if coverage_repairs:
        report['coverage_repairs']['key_context']=fuse_key_context(base,graph,photo,geometry)
        report['coverage_repairs']['written_low_octaves']=fuse_written_low_octaves(base,graph,photo,geometry)
        report['coverage_repairs']['key_after_written']=fuse_key_context(base,graph,photo,geometry)
        if temporal_before!=time_voices(base):raise ValueError('Coverage pitch repair changed time or voice')
        report['coverage_repairs']['isolated_melody']=fuse_isolated_melody(base,graph,photo,geometry)
    report['coverage_limits']={'enabled':coverage_repairs,'maximum_systems':64,'maximum_groups_per_row':2048,'maximum_columns':64,'maximum_events_per_measure':128,'maximum_recovery_jobs':128,'maximum_melody_groups':8,'photo_only_key_and_staff_proof':True,'unchanged_rest_clock_checked':True,'no_additional_model_or_PL':True}
    from route_trials.relation_tuplet_scope import fuse as fuse_tuplet_scope
    scope_identity_before=identities(base)
    report['tuplet_scope']=fuse_tuplet_scope(base,graph,photo,geometry) if coverage_repairs and tuplet_scope else []
    if scope_identity_before!=identities(base):raise ValueError('Tuplet scope changed pitch, staff, voice or cardinality')
    report['tuplet_scope_limits']={'enabled':coverage_repairs and tuplet_scope,'maximum_false_prefix_groups':2,'maximum_jobs':128,'maximum_divisions_factor':12,'proof':'exclusive_printed_three_continuous_double_beam_identical_staff_positions_and_conserved_meter'}
    report['identity_repairs']={'enabled':identity_repairs,'key_state':[],'chord_heads':[],'sextuplets':[],'attachments':sum(len(r.get('chord_attachment_repairs',[])) for r in graph['systems'])}
    if identity_repairs:
        from route_trials.relation_key_state import fuse as fuse_key_state
        from route_trials.relation_chord_coverage import fuse as fuse_chord_heads
        before_key=time_voices(base)
        report['identity_repairs']['key_state']=fuse_key_state(base,graph,photo,geometry)
        if before_key!=time_voices(base):raise ValueError('Per-pitch key state changed time, staff or voice')
        def all_events(root):return [Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in row) for row in event_records(root)]
        before_heads=all_events(base)
        report['identity_repairs']['chord_heads']=fuse_chord_heads(base,graph,photo,geometry)
        after_heads=all_events(base)
        if any(a-b for a,b in zip(before_heads,after_heads)) or sum(sum((b-a).values()) for a,b in zip(before_heads,after_heads))!=len(report['identity_repairs']['chord_heads']):raise ValueError('Missing chord head repair altered an existing event')
        from route_trials.relation_sextuplets import fuse as fuse_sextuplets
        before_six=identities(base)
        report['identity_repairs']['sextuplets']=fuse_sextuplets(base,graph,photo,geometry)
        if before_six!=identities(base):raise ValueError('Sextuplet repair changed pitch, staff, voice or cardinality')
    report['identity_limits']={'enabled':identity_repairs,'maximum_columns':64,'maximum_chord_heads':8,'minimum_orphan_filled_core':.95,'minimum_shared_stem_confidence':.97,'signature_maximum_residual_steps':.4,'signature_maximum_residual_spread':.12,'isolated_natural_two_staggered_bars_required':True,'no_additional_model_or_PL':True}
    report['measure_joint']={'enabled':measure_joint,'pitch_chords':[],'clocks':[]}
    if measure_joint:
        from route_trials.relation_measure_joint import fuse as fuse_measure_joint
        def temporal(root):return [Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['grace']) for r in row) for row in event_records(root)]
        before_joint=temporal(base)
        report['measure_joint']['pitch_chords']=fuse_measure_joint(base,graph,photo,geometry)
        after_joint=temporal(base)
        added=sum(c['method'] in ('joint_partial_chord_shared_stem_ink_head','joint_chromatic_unison_two_signs_opposite_ink_lobes') for c in report['measure_joint']['pitch_chords'])
        if any(a-b for a,b in zip(before_joint,after_joint)) or sum(sum((b-a).values()) for a,b in zip(before_joint,after_joint))!=added:raise ValueError('Measure joint pitch/chord decoder changed an existing clock or voice')
        from route_trials.relation_joint_clock import fuse as fuse_measure_clock
        before_clock=[Counter((r['staff'],r['midi'],r['grace']) for r in row) for row in event_records(base)]
        report['measure_joint']['clocks']=fuse_measure_clock(base,graph,photo,geometry)
        after_clock=[Counter((r['staff'],r['midi'],r['grace']) for r in row) for row in event_records(base)]
        if any(a-b for a,b in zip(before_clock,after_clock)) or sum(sum((b-a).values()) for a,b in zip(before_clock,after_clock))!=len(report['measure_joint']['clocks']):raise ValueError('Measure joint clock decoder changed a pitch, staff or original cardinality')
    report['measure_joint_limits']={'enabled':measure_joint,'maximum_columns':64,'maximum_chord_heads':8,'maximum_assignment_states':64,'overflow_assignment':'abstain','maximum_pitch_chord_jobs':128,'maximum_triplet_notes_per_lane':24,'maximum_clock_jobs':128,'maximum_divisions_factor':12,'no_additional_model_or_PL':True,'reference_supplied':False}
    report['expanded_joint_candidates']={'enabled':measure_joint and expanded_joint_candidates,'pitch_chords':[],'clocks':[]}
    if measure_joint and expanded_joint_candidates:
        before_candidates=temporal(base)
        report['expanded_joint_candidates']['pitch_chords']=fuse_measure_joint(base,graph,photo,geometry,expanded_candidates=True)
        after_candidates=temporal(base)
        added=sum(c['method'] in ('joint_partial_chord_shared_stem_ink_head','joint_chromatic_unison_two_signs_opposite_ink_lobes','joint_two_detected_unison_heads_two_isolated_signs') for c in report['expanded_joint_candidates']['pitch_chords'])
        if any(a-b for a,b in zip(before_candidates,after_candidates)) or sum(sum((b-a).values()) for a,b in zip(before_candidates,after_candidates))!=added:raise ValueError('Expanded joint candidates changed an existing clock, staff or voice')
        from route_trials.relation_joint_lane_clock import fuse as fuse_lane_clock
        before_lane=identities(base)
        report['expanded_joint_candidates']['clocks']=fuse_lane_clock(base,graph,photo,geometry)
        if before_lane!=identities(base):raise ValueError('Complete lane clock repair changed pitch, staff, voice or cardinality')
    report['expanded_joint_limits']={'enabled':measure_joint and expanded_joint_candidates,'maximum_heads_per_assignment':8,'maximum_glyphs_per_assignment':8,'maximum_exact_masks':256,'maximum_assignment_numeric_bytes':36864,'assignment_ambiguity_margin':.15,'maximum_columns':64,'maximum_events_per_measure':128,'maximum_jobs_per_phase':128,'maximum_tuplet_columns':24,'maximum_divisions_factor':12,'independent_isolated_glyph_shape_required_for_weak_proposals':True,'no_additional_model_or_PL':True,'reference_supplied':False}
    report['structural_decode']={'enabled':structural_decode,'changes':[]}
    if structural_decode:
        from route_trials.relation_structural_decode import fuse as fuse_structural
        report['structural_decode']['changes'],report['structural_decode']['diagnostics']=fuse_structural(base,other,graph,photo,geometry)
    report['structural_limits']={'maximum_shift_bars':2,'minimum_distinct_shifted_patterns':2,'maximum_jobs':128,'maximum_events_per_measure':128,'maximum_stems':4096,'maximum_staff_rows':128,'maximum_tuplet_columns':24,'maximum_divisions_factor':12,'beam_ink_samples':7,'hollow_head_requires_enclosed_white_hole':True,'no_additional_model_or_PL':True}
    E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
    report.update({'arbitration':arbitration,'ordinary_clock_preserved_after_arbitration':before==clocks(base),
                   'ordinary_clock_preserved_during_voice_grace_refinement':True,'backend':'ONNX Runtime CPU',
                   'pitch_refinement_preserved_time_voice':True,
                   'notation_refinement_preserved_time_voice':True,
                   'torch_loaded':'torch' in sys.modules,'image_sha256':sha(photo),'model_manifest_sha256':sha(models/'manifest.json'),
                   'elapsed_s':time.perf_counter()-start,'physical_arm64_verified':False})
    if report['torch_loaded']:raise ValueError('Deployment inference unexpectedly loaded Torch')
    save(output.with_suffix('.merge.json'),report)
    return report


@scoped_cpu_sessions(validate_models)
@cached_relation_inputs
@cached_photo_reads
def enhance(photo,homr_geometry,production,models,output,staff_scale=None,supplementary_staff_scale=None,bar_count_repair=True,joint_event_grouping=True,joint_dependencies=True,electronic_event_clocks=True,joint_stream_clocks=True,classical_edge_gate=True,write_merge_report=True):
    from route_trials.map_staffs import map_photo
    from route_trials.relation_decode import recognize
    output=Path(output);models=Path(models);manifest=validated_manifest_for(models) or validate_models(models)
    raw=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
    if raw is None:raise ValueError('Unreadable photograph')
    h,w=raw.shape;limits=manifest['limits']
    if h*w>min(16000000,limits['maximum_photo_pixels']) or max(h,w)>min(8192,limits['maximum_photo_dimension']):raise ValueError('Photograph exceeds bounded relation image size')
    del raw
    staff_geometry=output.with_suffix('.staff.json');save(staff_geometry,map_photo(Path(photo),Path(homr_geometry)))
    frontend=output.with_suffix('.frontend.musicxml')
    recognize(Path(photo),staff_geometry,Path(production),models/'symbol.onnx',models/'symbol_scores.json',
              models/'stroke.onnx',models/'stroke_scores.json',models/'edge.onnx',models/'edge_scores.json',frontend,backend='onnx',staff_scale=staff_scale,classical_edge_gate=classical_edge_gate,export_xml=False)
    report=decode_and_fuse(photo,staff_geometry,production,frontend.with_suffix('.graph.json'),models,output)
    frontend_run=json.loads(frontend.with_suffix('.run.json').read_text(encoding='utf8'))
    report['inference_profile']=frontend_run['inference_profile']
    report['classical_edge_gate']=frontend_run.get('classical_edge_gate',{'enabled':False})
    report['model_dispatch']={'traditional_first_scope':'head-to-stem links',
        'uncertain_links_query_original_edge_model':bool(classical_edge_gate),
        'symbol_and_stroke_models_still_scan_image':True}
    count_relation=output.with_suffix('.relation.musicxml')
    if supplementary_staff_scale is not None:
        # The recognized headers precede count recovery. Keep them as the
        # semantic context for both passes; repaired XML may have lost its
        # original system markers or have a different number of measures.
        supplemental=output.with_suffix('.scale_frontend.musicxml')
        candidate=output.with_suffix('.scale_candidate.musicxml')
        try:
            from route_trials.meter_header_consensus import repair as repair_meters,homr_head_rows
            meter_base=parse_score(output).getroot();meter_headers=parse_score(production).getroot()
            meter_geometry=geometry_json(staff_geometry)
            meter_graph=homr_head_rows(geometry_json(homr_geometry),meter_geometry)
            meter_changes=repair_meters(meter_base,meter_headers,meter_graph,photo,meter_geometry)
            supplement_base=output;supplement_headers=Path(production)
            if meter_changes:
                supplement_base=output.with_suffix('.meter_base.musicxml');supplement_headers=output.with_suffix('.meter_headers.musicxml')
                E.ElementTree(meter_base).write(supplement_base,encoding='utf8',xml_declaration=True)
                E.ElementTree(meter_headers).write(supplement_headers,encoding='utf8',xml_declaration=True)
            recognize(Path(photo),staff_geometry,supplement_headers,models/'symbol.onnx',models/'symbol_scores.json',
                      models/'stroke.onnx',models/'stroke_scores.json',models/'edge.onnx',models/'edge_scores.json',supplemental,backend='onnx',staff_scale=supplementary_staff_scale,classical_edge_gate=classical_edge_gate,export_xml=False)
            second=decode_and_fuse(photo,staff_geometry,supplement_base,supplemental.with_suffix('.graph.json'),models,candidate,recognized_headers=supplement_headers)
            checked=parse_score(candidate).getroot()
            if checked.tag!='score-partwise' or len(checked.findall('part'))!=1 or not checked.findall('part/measure'):
                raise ValueError('Unsupported or empty supplemental MusicXML')
            import shutil
            shutil.copyfile(candidate,output)
            count_relation=candidate.with_suffix('.relation.musicxml')
            supplemental_run=json.loads(supplemental.with_suffix('.run.json').read_text(encoding='utf8'))
            second['inference_profile']=supplemental_run['inference_profile']
            second['classical_edge_gate']=supplemental_run.get('classical_edge_gate',{'enabled':False})
            report['supplementary_staff_scale']={'enabled':True,'report':second,'fallback':False,'header_meter_repairs':meter_changes}
        except Exception as exc:
            report['supplementary_staff_scale']={'enabled':True,'fallback':True,'error':str(exc)}
    if bar_count_repair:
        from route_trials.relation_bar_count import repair as repair_count
        report['local_bar_count']={'enabled':True,'changes':[],'maximum_count_difference':1,'maximum_systems':64,'maximum_bars_per_system':128,'maximum_events_per_inserted_bar':128,'maximum_staff_rows':128,'maximum_stems':4096,'existing_measure_clocks_preserved':True,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot()
            changes=repair_count(base,parse_score(count_relation).getroot(),json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8')),photo,geometry_json(staff_geometry))
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['local_bar_count']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['local_bar_count'].update({'fallback':True,'error':str(exc)})
    from route_trials.relation_complete_clocks import fuse as complete_clocks
    report['complete_staff_clocks']={'changes':[],'maximum_jobs':128,'maximum_events_per_measure':128,'maximum_columns_per_staff':64,'maximum_chord_heads':8,'maximum_meter_beats':16,'maximum_divisions':9600,'pitch_duration_voice_grace_cardinality_preserved':True,'reference_supplied':False,'no_additional_model_or_PL':True}
    try:
        base=parse_score(output).getroot()
        changes=complete_clocks(base,json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8')),photo,geometry_json(staff_geometry))
        if changes:
            E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
        report['complete_staff_clocks']['changes']=changes
    except (ValueError,OSError,KeyError,E.ParseError) as exc:
        report['complete_staff_clocks'].update({'fallback':True,'error':str(exc)})
    if joint_event_grouping:
        from route_trials.relation_joint_events import fuse as augment_events
        from route_trials.relation_joint_overflow import fuse as regroup_overflow
        report['joint_event_grouping']={'enabled':True,'changes':[],'overflow_changes':[],'maximum_jobs':128,'maximum_events_per_measure':128,'maximum_columns':64,'maximum_chord_heads':8,'maximum_orphan_hollows_per_row':128,'maximum_stems':4096,'maximum_native_beam_offsets':97,'native_beam_samples':9,'no_additional_model_or_PL':True,'reference_supplied':False}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes,diagnostics=augment_events(base,graph,photo,geometry)
            overflow=regroup_overflow(base,graph,photo,geometry)
            if changes or overflow:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['joint_event_grouping'].update({'changes':changes,'diagnostics':diagnostics,'overflow_changes':overflow})
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['joint_event_grouping'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_joint_lanes import fuse as joint_lanes
        from route_trials.relation_joint_dependencies import fuse as dependencies
        report['joint_dependencies']={'enabled':True,'changes':[],'maximum_jobs':128,'maximum_events':128,'maximum_columns':64,'maximum_orphans_per_row':128,'maximum_stems':4096,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=joint_lanes(base,graph,photo,geometry);extra,diagnostics=dependencies(base,graph,photo,geometry);changes+=extra
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['joint_dependencies'].update({'changes':changes,'diagnostics':diagnostics})
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['joint_dependencies'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_physical_streams import fuse as physical_streams,hollow_unisons
        from route_trials.relation_multivalue_state import fuse as unison_state
        report['physical_streams']={'enabled':True,'changes':[],'maximum_jobs':128,'maximum_events':128,'maximum_columns':64,'maximum_heads':8,'maximum_orphans_per_row':128,'maximum_stems':4096,'native_slot_offsets':23,'voice_lookback_measures':8,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=physical_streams(base,graph,photo,geometry);changes+=hollow_unisons(base,graph,photo,geometry);changes+=unison_state(base,graph,photo,geometry)
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['physical_streams']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['physical_streams'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_label_rest import fuse as label_rests
        from route_trials.relation_column_binding import fuse as column_state
        report['label_rest_columns']={'enabled':True,'changes':[],'maximum_jobs_per_phase':128,'maximum_events_per_measure':128,'maximum_columns':64,'maximum_chord_heads':8,'maximum_stems':4096,'maximum_meter_beats':16,'maximum_divisions':9600,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=label_rests(base,graph,photo,geometry);changes+=column_state(base,graph,photo,geometry)
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['label_rest_columns']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['label_rest_columns'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_chord_evidence import fuse as chord_evidence
        report['chord_evidence']={'enabled':True,'changes':[],'maximum_jobs':128,'maximum_events':128,'maximum_columns':64,'maximum_heads':8,'maximum_stems':4096,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=chord_evidence(base,graph,photo,geometry)
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['chord_evidence']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['chord_evidence'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_clock_lattice import fuse as clock_lattice
        report['shared_clock_lattice']={'enabled':True,'changes':[],'maximum_jobs':128,'maximum_events':128,'maximum_columns':64,'maximum_heads':8,'maximum_stems':4096,'maximum_divisions_factor':12,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=clock_lattice(base,graph,photo,geometry,geometry_json(homr_geometry))
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['shared_clock_lattice']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['shared_clock_lattice'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_courtesy_boundary import fuse as courtesy_boundary
        report['courtesy_boundary']={'enabled':True,'changes':[],'maximum_systems':64,'maximum_nodes_per_staff':2048,'maximum_stems':4096,'reference_supplied':False,'no_additional_model_or_PL':True}
        try:
            base=parse_score(output).getroot();graph=json.loads(count_relation.with_suffix('.graph.json').read_text(encoding='utf8'));geometry=geometry_json(staff_geometry)
            changes=courtesy_boundary(base,graph,photo,geometry,geometry_json(homr_geometry))
            if changes:
                E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
            report['courtesy_boundary']['changes']=changes
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['courtesy_boundary'].update({'fallback':True,'error':str(exc)})
    if joint_dependencies:
        from route_trials.relation_beam_clock import fuse as beam_clock
        report['physical_beam_clock']={'enabled':True,'changes':[],'maximum_events':128,'maximum_cells':16,'maximum_voice_labels':8,'reference_supplied':False,'experimental':True}
        try:
            import cv2 as beam_cv2
            from homr.autocrop import autocrop
            from homr.resize import resize_image
            from homr.color_adjust import apply_clahe
            base=parse_score(output).getroot();hg=geometry_json(homr_geometry)
            raw=beam_cv2.imread(str(photo))
            if raw is not None and raw.shape[0]*raw.shape[1]<=16000000 and max(raw.shape[:2])<=8192:
                raster=beam_cv2.resize(apply_clahe(resize_image(autocrop(raw))),tuple(hg['preprocessed_size']))
                del raw
                changes=beam_clock(base,hg,raster)
                if changes:
                    E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
                report['physical_beam_clock']['changes']=changes
                from route_trials.relation_group_clock import fuse as local_group_clock
                local_changes=local_group_clock(base,hg,raster)
                if local_changes:
                    E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf8',xml_declaration=True)
                report['local_group_clock']={'enabled':True,'changes':local_changes,'reference_supplied':False,'maximum_events_per_bar':128,'maximum_heads_per_bar':128,'maximum_time_slots':48,'global_meter_invented':False}
                if electronic_event_clocks:
                    from route_trials.relation_electronic_clocks import fuse as electronic_clocks
                    report['electronic_event_clocks']=electronic_clocks(base,hg,raster,photo=photo)
                    if any(s['changes'] for s in report['electronic_event_clocks']['stages'].values()):
                        clock_temp=output.with_name(output.name+'.clock.tmp')
                        try:
                            E.indent(base,space='  ');E.ElementTree(base).write(clock_temp,encoding='utf8',xml_declaration=True)
                            clock_temp.replace(output)
                        finally:clock_temp.unlink(missing_ok=True)
                if joint_stream_clocks:
                    try:
                        from route_trials.relation_joint_decoder import fuse as joint_clocks
                        report['joint_stream_clocks']=joint_clocks(base,hg,raster,photo=photo)
                        if any(s['changes'] for s in report['joint_stream_clocks']['stages'].values()):
                            joint_temp=output.with_name(output.name+'.joint.tmp')
                            try:
                                E.indent(base,space='  ');E.ElementTree(base).write(joint_temp,encoding='utf8',xml_declaration=True)
                                joint_temp.replace(output)
                            finally:joint_temp.unlink(missing_ok=True)
                    except Exception as joint_error:
                        report['joint_stream_clocks']={'enabled':True,'fallback':True,'error':str(joint_error),'stages':{}}
        except (ValueError,OSError,KeyError,E.ParseError) as exc:
            report['physical_beam_clock'].update({'fallback':True,'error':str(exc)})
    # Attach complete audit trails only after the work graph is no longer used.
    if classical_edge_gate:
        report['classical_edge_gate_trace']=json.loads(frontend.with_suffix('.edge_gate.json').read_text(encoding='utf8'))
        supplement=report.get('supplementary_staff_scale',{})
        if supplement.get('enabled') and not supplement.get('fallback'):
            supplement['report']['classical_edge_gate_trace']=json.loads(supplemental.with_suffix('.edge_gate.json').read_text(encoding='utf8'))
    if write_merge_report:save(output.with_suffix('.merge.json'),report)
    return report
