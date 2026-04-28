lerobot-find-cameras opencv 

lerobot-find-port

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=0427/sort_blocks \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the block" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=15 \
    --resume=True


lerobot-dataset-viz --repo-id 0427/sort_blocks
  

lerobot-train \
  --dataset.repo_id=0427/sort_blocks  \
  --policy.type=diffusion \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=1 --policy.push_to_hub=False


lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=0427/eval_sort_blocks \
  --dataset.single_task="Grab blue block" \
  --policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model