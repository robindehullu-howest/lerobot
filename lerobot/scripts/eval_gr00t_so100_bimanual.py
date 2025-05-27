# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SO100 Real Robot
import time
import logging
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
import sys
import os
#sys.path.append(os.path.expanduser("C:/Users/robin/Documents/MCT/Sem 6/Stage/Isaac-GR00T/gr00t/eval/"))
#from service import ExternalRobotInferenceClient
from lerobot.common.gr00t_eval.service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f"  # Include milliseconds
)

#################################################################################


class SO100Robot:
    def __init__(self, calibrate=False, enable_camera=False):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        print("SO100CameraConfig: ", self.config.cameras)

        if not enable_camera:
            self.config.cameras = {}
            
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms

    def for_each_arm(self, fn, *args, **kwargs):
        """Apply a function to each arm in the robot."""
        for arm_name, arm in self.motor_bus.items():
            fn(arm_name, arm, *args, **kwargs)

    @contextmanager
    def activate(self):
        try:
            self.connect()
            #self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.for_each_arm(lambda arm_name, arm: arm.connect())
        self.disable()

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        self.enable()
        self.for_each_arm(lambda arm_name, arm: print(f"Connected to {arm_name} --- robot present position: {arm.read('Present_Position')}"))
        self.robot.is_connected = True

        if self.enable_camera and hasattr(self.robot, "cameras"):
            self.cameras = sorted(
                    self.robot.cameras.items(),
                    key=lambda x: x[1].camera_index
                    )
            for name, camera in self.cameras:
                camera.connect()
                print(f"Camera {name} (index {camera.camera_index}) connected")
        else:
            self.cameras = None

        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.for_each_arm(lambda arm_name, arm: arm.write("Mode", 0))
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.for_each_arm(lambda arm_name, arm: arm.write("P_Coefficient", 10))
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.for_each_arm(lambda arm_name, arm: arm.write("I_Coefficient", 0))
        self.for_each_arm(lambda arm_name, arm: arm.write("D_Coefficient", 32))
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.for_each_arm(lambda arm_name, arm: arm.write("Lock", 0))
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.for_each_arm(lambda arm_name, arm: arm.write("Maximum_Acceleration", 254))
        self.for_each_arm(lambda arm_name, arm: arm.write("Acceleration", 254))

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        # print("current_state", current_state)
        # print all keys of the observation
        # print("observation keys:", self.robot.capture_observation().keys())
        current_state = torch.tensor([90, 90, 90, 90, -70, 30])
        self.robot.send_action(current_state)
        time.sleep(2)
        print("-------------------------------- moving to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("-------------------------------- moving to home pose")
        home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_imgs(self):
        imgs = {
            name: self.get_observation()[f"observation.images.{name}"].data.numpy()
            for name, _ in self.cameras
        }

        # Convert to RGB format
        # for img in imgs.values():
        #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

        return imgs

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.for_each_arm(lambda arm_name, arm: arm.write("Torque_Enable", TorqueMode.ENABLED.value))

    def disable(self):
        self.for_each_arm(lambda arm_name, arm: arm.write("Torque_Enable", TorqueMode.DISABLED.value))

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, imgs, state):
        obs_dict = {
            f"video.{name}": img[np.newaxis, :, :, :]
            for name, img in imgs.items()
        }
        obs_dict["state.main_arm"] = state[:5][np.newaxis, :].astype(np.float64)
        obs_dict["state.main_gripper"] = state[5:6][np.newaxis, :].astype(np.float64)
        obs_dict["state.secondary_arm"] = state[6:11][np.newaxis, :].astype(np.float64)
        obs_dict["state.secondary_gripper"] = state[11:12][np.newaxis, :].astype(np.float64)
        obs_dict["annotation.human.task_description"] = [self.language_instruction]

        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("\nInference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {}

        obs_dict = {
            "video.top": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.main_gripper": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.secondary_gripper": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.main_arm": np.zeros((1, 5)),
            "state.main_gripper": np.zeros((1, 1)),
            "state.secondary_arm": np.zeros((1, 5)),
            "state.secondary_gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    #plt.figure(title)
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame

#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="10.110.17.183")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument(
        "--lang_instruction", type=str, default="Unfold a towel."
    )
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["main_arm", "main_gripper", "secondary_arm", "secondary_gripper"]
    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        if args.record_imgs:
            # create a folder to save the images and delete all the images in the folder
            output_dir = "outputs/eval_images"
            os.makedirs(output_dir, exist_ok=True)
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        robot = SO100Robot(calibrate=False, enable_camera=True)
        frame_count = 0
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                imgs = robot.get_current_imgs()
                state = robot.get_current_state()
                action = client.get_action(imgs, state)
                start_time = time.time()
                for i in range(ACTION_HORIZON):
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (12,), concat_action.shape
                    robot.set_target_state(torch.from_numpy(concat_action))
                    time.sleep(0.02)

                    # get the realtime image
                    imgs = robot.get_current_imgs()
                    view_img(imgs["top"])

                    if args.record_imgs:
                        # resize images to 320x240
                        for name, img in imgs.items():
                            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 240))
                            cv2.imwrite(f"outputs/eval_images/img_{name}_{frame_count}.jpg", img)
                        frame_count += 1

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
        )

        robot = SO100Robot(calibrate=False, enable_camera=True)

        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                imgs = [
                    dataset[i]["observation.images.top"].data.numpy(),
                    dataset[i]["observation.images.frontal"].data.numpy(),
                    dataset[i]["observation.images.gripper"].data.numpy()
                ]
                
                # original shape (3, 480, 640) for image data
                realtime_imgs = robot.get_current_imgs()

                img = imgs[0].transpose(1, 2, 0)
                view_img(img, realtime_imgs[0])
                actions.append(action)
                robot.set_target_state(action)
                time.sleep(0.05)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done all actions")
            #robot.go_home()
            print("Done home")
