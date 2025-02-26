from typing import Protocol

from lerobot.common.robots.aloha.configuration_aloha import AlohaRobotConfig
from lerobot.common.robots.config_abc import (
    ManipulatorRobotConfig,
    RobotConfig,
)
from lerobot.common.robots.koch.configuration_koch import KochBimanualRobotConfig, KochRobotConfig
from lerobot.common.robots.lekiwi.configuration_lekiwi import LeKiwiRobotConfig
from lerobot.common.robots.moss.configuration_moss import MossRobotConfig
from lerobot.common.robots.so_100.configuration_so_100 import So100RobotConfig
from lerobot.common.robots.stretch3.configuration_stretch3 import StretchRobotConfig


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "aloha":
        return AlohaRobotConfig(**kwargs)
    elif robot_type == "koch":
        return KochRobotConfig(**kwargs)
    elif robot_type == "koch_bimanual":
        return KochBimanualRobotConfig(**kwargs)
    elif robot_type == "moss":
        return MossRobotConfig(**kwargs)
    elif robot_type == "so100":
        return So100RobotConfig(**kwargs)
    elif robot_type == "stretch":
        return StretchRobotConfig(**kwargs)
    elif robot_type == "lekiwi":
        return LeKiwiRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robots.manipulator import ManipulatorRobot

        return ManipulatorRobot(config)
    elif isinstance(config, LeKiwiRobotConfig):
        from lerobot.common.robots.mobile_manipulator import MobileManipulator

        return MobileManipulator(config)
    else:
        from lerobot.common.robots.stretch3.robot_stretch3 import StretchRobot

        return StretchRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)
