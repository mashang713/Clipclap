"""Feature folder names and helpers for temporal vs baseline UCF features."""

TEMPORAL_CLIP_FEATURE_METHOD = "cls_features_temporal_16"
STATIC_TEMPORAL_CLIP_FEATURE_METHOD = "cls_features_static_temporal_16"
BASELINE_AUDIO_FEATURE_METHOD = "cls_features_non_averaged"
TEMPORAL_CLIP_NUM_FRAMES = 16

TEMPORAL_FEATURE_METHODS = (
    TEMPORAL_CLIP_FEATURE_METHOD,
    STATIC_TEMPORAL_CLIP_FEATURE_METHOD,
)


def is_temporal_clip_feature_method(feature_extraction_method) -> bool:
    return str(feature_extraction_method) in TEMPORAL_FEATURE_METHODS


def is_static_temporal_clip_feature_method(feature_extraction_method) -> bool:
    return str(feature_extraction_method) == STATIC_TEMPORAL_CLIP_FEATURE_METHOD


def audio_feature_method_for_dataset(feature_extraction_method) -> str:
    """Temporal / static+temporal runs use legacy CLAP audio from the non-averaged tree."""
    if is_temporal_clip_feature_method(feature_extraction_method):
        return BASELINE_AUDIO_FEATURE_METHOD
    return str(feature_extraction_method)
