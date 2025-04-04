# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate kangaroo.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/custom/spawn_kangaroo.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate bipedal robots."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import KANGAROO_CFG  # isort: skip


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


KANGAROO_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PAL/Kangaroo/kangaroo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "leg_left_.*": 0.0,
            "leg_right_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_left_1_motor",
                "leg_right_1_motor",
                "leg_left_2_motor",
                "leg_left_3_motor",
                "leg_right_2_motor",
                "leg_right_3_motor",
                "leg_left_4_motor",
                "leg_left_5_motor",
                "leg_left_length_motor",
                "leg_right_length_motor",
                "leg_right_4_motor",
                "leg_right_5_motor",
            ],
            effort_limit={
                "leg_left_1_motor": 3000,  # 2000,
                "leg_right_1_motor": 3000,  # 2000,
                "leg_left_2_motor": 3000,
                "leg_right_2_motor": 3000,
                "leg_left_3_motor": 3000,
                "leg_right_3_motor": 3000,
                "leg_left_4_motor": 3000,
                "leg_right_4_motor": 3000,
                "leg_left_5_motor": 3000,
                "leg_right_5_motor": 3000,
                "leg_left_length_motor": 5000,
                "leg_right_length_motor": 5000,
            },
            velocity_limit={
                "leg_left_1_motor": 0.4,
                "leg_right_1_motor": 0.4,
                "leg_left_2_motor": 0.4,
                "leg_right_2_motor": 0.4,
                "leg_left_3_motor": 0.4,
                "leg_right_3_motor": 0.4,
                "leg_left_4_motor": 0.4,
                "leg_right_4_motor": 0.4,
                "leg_left_5_motor": 0.4,
                "leg_right_5_motor": 0.4,
                "leg_left_length_motor": 0.625,
                "leg_right_length_motor": 0.625,
            },
            stiffness=100000.0,
            damping={
                "leg_left_1_motor": 632.5,
                "leg_right_1_motor": 632.5,
                "leg_left_2_motor": 632.5,
                "leg_right_2_motor": 632.5,
                "leg_left_3_motor": 632.5,
                "leg_right_3_motor": 632.5,
                "leg_left_4_motor": 632.5,
                "leg_right_4_motor": 632.5,
                "leg_left_5_motor": 632.5,
                "leg_right_5_motor": 632.5,
                "leg_left_length_motor": 632.5,
                "leg_right_length_motor": 632.5,
            },
            # armature={
            #     "leg_left_1_motor": 0.01,
            #     "leg_right_1_motor": 0.01,
            #     "leg_left_2_motor": 0.01,
            #     "leg_right_2_motor": 0.01,
            #     "leg_left_3_motor": 0.01,
            #     "leg_right_3_motor": 0.01,
            #     "leg_left_4_motor": 0.01,
            #     "leg_right_4_motor": 0.01,
            #     "leg_left_5_motor": 0.01,
            #     "leg_right_5_motor": 0.01,
            #     "leg_left_length_motor": 0.01,
            #     "leg_right_length_motor": 0.01,
            # },
        ),
    },
)

def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origins
    origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
        ]
    ).to(device=sim.device)

    # Robots
    kangaroo = Articulation(KANGAROO_FIXED_CFG.replace(prim_path="/World/kangaroo"))
    robots = [kangaroo]

    return robots, origins


def run_simulator(
    sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                # reset dof state
                # print("joint_names", robot.data.joint_names)
                # print("num_joints", len(robot.data.joint_names))
                # print("actuators\n", robot.actuators["motor"])
                # print(
                #     "default_joint_pos_limits\n",
                #     robot.data.default_joint_pos_limits[
                #         :, robot.actuators["motor"].joint_indices, :
                #     ],
                # )
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel,
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        # for robot in robots:
        #     robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        #     robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

    # design scene
    robots, origins = design_scene(sim)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
