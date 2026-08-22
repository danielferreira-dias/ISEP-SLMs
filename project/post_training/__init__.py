"""Student post-training stages for the ISEP dermatology thesis.

Dataset generation stays under :mod:`project.pipeline`.  This namespace owns
the model updates that consume frozen releases: E3 SFT is runnable, while E4
OPD and E5 GRPO currently expose fail-closed availability contracts.
"""

from project.post_training.grpo import GRPO_STAGE
from project.post_training.opd import OPD_STAGE

__all__ = ["GRPO_STAGE", "OPD_STAGE"]
