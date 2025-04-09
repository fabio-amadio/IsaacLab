import argparse
from isaaclab.app import AppLauncher

# 添加 CLI 参数
parser = argparse.ArgumentParser(description="Kangaroo debug: Load and inspect robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入其他依赖
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.kangaroo.flat_env_cfg import KangarooFlatEnvCfg_PLAY
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
import torch

# 创建 Kangaroo 固定配置
KANGAROO_DEBUG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PAL/Kangaroo/kangaroo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # 悬浮空中
        joint_pos={"leg_left_.*": 0.0, "leg_right_.*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=[".*_motor"]
        )
    },
)

def main():
    # 加载环境配置
    env_cfg = KangarooFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 2.0
    env_cfg.scene.robot = KANGAROO_DEBUG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene.articulations["robot"]

    # 打印信息
    print("\n✅ Kangaroo successfully loaded!\n")
    print("📌 Joint Names:")
    for name in robot.joint_names:
        print(" -", name)

    print("\n📌 Initial Joint Positions:")
    print(robot.data.joint_pos[0])

    print("\n📌 Joint Limits:")
    print(robot.data.default_joint_pos_limits[0])

    # 保持运行直到手动关闭
    while simulation_app.is_running():
        env.step(torch.zeros_like(env.action_manager.action))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
