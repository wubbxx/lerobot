lerobot-find-cameras opencv 

lerobot-find-port

lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=my_awesome_follower_arm     --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM0     --teleop.id=my_awesome_leader_arm     --display_data=true     --dataset.repo_id=reach_yellow/0519     --dataset.num_episodes=5     --dataset.single_task="Reach Yellow"     --dataset.push_to_hub=false     --dataset.episode_time_s=15     --dataset.reset_time_s=10      --resume=True


lerobot-dataset-viz --repo-id 0427/sort_blocks
  

lerobot-train \
  --dataset.repo_id=reach_yellow/0519   \
  --policy.type=diffusion \
  --output_dir=outputs/train/reach_yellow  \
  --job_name=reach_yellow \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=100000 --policy.push_to_hub=False


scp -r /home/nvidia/.cache/huggingface/lerobot/reach_yellow/ wbx@192.168.10.16:/home/wbx/.cache/huggingface/lerobot/reach_yellow

scp -r wbx@192.168.10.16:~/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/020000 /home/nvidia/lerobot/outputs/train/reach_yellow/checkpoints/020000

scp -r wbx@192.168.10.16:~/embodied-ai/lerobot/outputs/train/reach_yellow/checkpoints/last /home/nvidia/lerobot/outputs/train/reach_yellow/checkpoints/last


rm -r /home/nvidia/.cache/huggingface/lerobot/reach_yellow/eval_0519 && \
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=reach_yellow/eval_0519 \
  --dataset.single_task="reach the yellow block" \
  --policy.path=outputs/train/reach_yellow/checkpoints/last/pretrained_model \
  --dataset.push_to_hub=false \
--policy.num_inference_steps=10

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true

