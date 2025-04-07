"""Configuration for a kangaroo robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


KANGAROO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PAL/Kangaroo/kangaroo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
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
            stiffness={
                "leg_left_1_motor":         1000000.0,
                "leg_right_1_motor":        1000000.0,
                "leg_left_2_motor":         1000000.0,
                "leg_right_2_motor":        1000000.0,
                "leg_left_3_motor":         1000000.0,
                "leg_right_3_motor":        1000000.0,
                "leg_left_4_motor":         5000000.0,
                "leg_right_4_motor":        5000000.0,
                "leg_left_5_motor":         5000000.0,
                "leg_right_5_motor":        5000000.0,
                "leg_left_length_motor":    2000000.0,
                "leg_right_length_motor":   2000000.0,
            },
            damping={
                "leg_left_1_motor":         1000.0,
                "leg_right_1_motor":        1000.0,
                "leg_left_2_motor":         1000.0,
                "leg_right_2_motor":        1000.0,
                "leg_left_3_motor":         1000.0,
                "leg_right_3_motor":        1000.0,
                "leg_left_4_motor":         5000.0,
                "leg_right_4_motor":        5000.0,
                "leg_left_5_motor":         5000.0,
                "leg_right_5_motor":        5000.0,
                "leg_left_length_motor":    10000.0,
                "leg_right_length_motor":   10000.0,
            },
            # stiffness=500000.0,
            # damping={
            #     "leg_left_1_motor":         10000.0,
            #     "leg_right_1_motor":        10000.0,
            #     "leg_left_2_motor":         10000.0,
            #     "leg_right_2_motor":        10000.0,
            #     "leg_left_3_motor":         10000.0,
            #     "leg_right_3_motor":        10000.0,
            #     "leg_left_4_motor":         10000.0,
            #     "leg_right_4_motor":        10000.0,
            #     "leg_left_5_motor":         10000.0,
            #     "leg_right_5_motor":        10000.0,
            #     "leg_left_length_motor":    10000.0,
            #     "leg_right_length_motor":   10000.0,
            # },
            armature={
                "leg_left_1_motor": 0.01,
                "leg_right_1_motor": 0.01,
                "leg_left_2_motor": 0.01,
                "leg_right_2_motor": 0.01,
                "leg_left_3_motor": 0.01,
                "leg_right_3_motor": 0.01,
                "leg_left_4_motor": 0.01,
                "leg_right_4_motor": 0.01,
                "leg_left_5_motor": 0.01,
                "leg_right_5_motor": 0.01,
                "leg_left_length_motor": 0.01,
                "leg_right_length_motor": 0.01,
            },
        ),
    },
)
