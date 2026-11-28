"""QLoRA training and vLLM serving of Inkling-2-Mini on one 24 GB Ampere GPU."""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# Runtime type checks on every module of this package. Hot code opts out with @typing.no_type_check;
# is_pep484_tower accepts an int where a float is annotated. The Triton module is skipped: beartype cannot wrap a
# JITFunction (Triton compiles the function's source).
beartype_this_package(
    conf=BeartypeConf(
        is_pep484_tower=True,
        claw_skip_package_names=("inkling_qlora.inkling_triton",),
    )
)
