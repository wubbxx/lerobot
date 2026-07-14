# ================= 配置区域 =================
REPO_ID="reach_grape/0707"
TASK_NAME="Reach Target"
EPISODE_TIME=15
NUM_EPISODES=5
# ============================================

# 1. 如果路径存在则安全删除旧数据
CACHE_DIR="/home/nvidia/.cache/huggingface/lerobot/${REPO_ID}"
if [ -d "$CACHE_DIR" ]; then 
    echo "发现旧数据，正在清理缓存: $CACHE_DIR"
    rm -r "$CACHE_DIR"
fi

# 2. 启动 LeRobot 录制程序
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id="$REPO_ID" \
    --dataset.single_task="$TASK_NAME" \
    --dataset.episode_time_s="$EPISODE_TIME" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.push_to_hub=false \
    --dataset.reset_time_s=10 \
    --resume=False