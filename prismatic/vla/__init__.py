"""VLA helpers.

Keep the RLDS/TF-backed materializer lazy.  Constants and native model/action
head imports are useful for manifest-driven tasks that deliberately do not
install TensorFlow; importing ``prismatic.vla.constants`` should not pull the
RLDS dataset package into the process.
"""


def __getattr__(name: str):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_vla_dataset_and_collator"]
