"""Permit five-line rest-only pages after upstream's no-notehead rejection.

The fallback repeats CPU segmentation only on that explicit exception. It
uses the same clef/bar anchors, staff detector and Transformer as Homr, with
staff-line spacing replacing the unavailable notehead-height estimate.
Unpitched/percussion layouts remain unsupported.
"""
from statistics import median


def detect_rest_only(image_path,config):
    import homr.main as main
    predictions,debug=main.load_and_preprocess_predictions(image_path,config.enable_debug,
                                                          config.enable_cache,config.segnet_use_gpu)
    try:
        symbols=main.predict_symbols(debug,predictions)
        heads=main.combine_noteheads_with_stems(symbols.noteheads,symbols.stems_rest)
        if heads:
            raise RuntimeError('rest-only fallback found noteheads; abstaining')
        fragments=main.break_wide_fragments(symbols.staff_fragments)
        preliminary=main.detect_staff(debug,predictions.staff,fragments,symbols.clefs_keys,[])
        if not preliminary:
            raise RuntimeError('no clef-anchored five-line staff on rest-only page')
        spaces=[b-a for s in preliminary for p in s.grid for a,b in zip(p.y,p.y[1:])
                if len(p.y)==5]
        if not spaces or not 5<=median(spaces)<=40:
            raise RuntimeError('invalid staff spacing on rest-only page')
        bars=main.detect_bar_lines(symbols.bar_lines,float(median(spaces)*0.85))
        staffs=main.detect_staff(debug,predictions.staff,fragments,symbols.clefs_keys,bars)
        if not staffs or any(not s.grid or any(len(p.y)!=5 for p in s.grid) for s in staffs):
            raise RuntimeError('rest-only page lacks complete five-line staves')
        title=main.detect_title(debug,staffs[0])
        brace_img=main.prepare_brace_dot_image(predictions.symbols,predictions.staff)
        braces=main.create_rotated_bounding_boxes(brace_img,skip_merging=True,max_size=(100,-1))
        main.add_notes_to_staffs(staffs,[],predictions.symbols,predictions.notehead)
        multi=main.find_braces_brackets_and_grand_staff_lines(debug,staffs,braces)
        return multi,predictions.preprocessed,debug,title
    except Exception:
        debug.clean_debug_files_from_previous_runs()
        raise
