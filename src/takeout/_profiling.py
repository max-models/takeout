"""The scope-profiler tag of takeout's profiling regions.

The regions are opt-in: :mod:`takeout` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["takeout"])``.
"""

#: Tags of every takeout region; the first is the opt-in tag.
PROFILE_TAGS = ("takeout",)
