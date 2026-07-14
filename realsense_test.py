import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 1. 创建配置和管道对象
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. 配置你想开启的数据流 (Stream)
    # 参数含义: 流类型, 分辨率宽, 分辨率高, 数据格式, 帧率
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. 启动 Pipeline 硬件流
    profile = pipeline.start(config)

    # 4. 创建对齐对象 (将深度图对齐到彩色图)
    # D435i 的物理 RGB 镜头和红外镜头有基线距离，对齐后两张图的物体在像素层面上才能完全重合
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("D435i 数据流已成功开启，按 'q' 键退出...")

    try:
        while True:
            # 5. 阻塞等待下一组有效的帧数据
            frames = pipeline.wait_for_frames()
            
            # 执行对齐
            aligned_frames = align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # 6. 将原生的数据帧 (Frame) 转换为标准的 NumPy 数组 (ndarray)
            # color_image: 形状为 (480, 640, 3) 的 HWC 矩阵
            # depth_image: 形状为 (480, 640) 的单通道矩阵，每个像素的值代表距离（单位：毫米）
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 7. 数据流可视化处理
            # 原始深度数据是 16 位的 (0~65535)，直接渲染是一片黑。
            # 我们通过 convertScaleAbs 将其缩放到 0~255，并挂上伪彩色（COLORMAP_JET）方便肉眼观察
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            # 水平拼接两张图，方便在同一个窗口显示
            display_window = np.hstack((color_image, depth_colormap))

            # 8. OpenCV 渲染
            cv2.imshow('D435i Real-time Streams', display_window)
            
            # 监听键盘，按 q 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 9. 释放硬件连接（如果不释放，下次运行会报设备占用的错误）
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()