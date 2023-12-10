from enum import Enum


class Dimension(Enum):
    CAPABILITY = "Capability"
    CALIBRATION = "Calibration"
    ROBUSTNESS = "Robustness"
    FAIRNESS = "Fairness"
    EFFICIENCY = "Efficiency"
    TOXICITY = "Toxicity"
