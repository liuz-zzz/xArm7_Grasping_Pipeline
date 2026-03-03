# GraspPipeline代码剖析

## 1.Sorting_YOLO_SAM

用sorting脚本运行抓取单个物体的视觉处理

### 1.scene_cap_sort.py

**针对 Intel RealSense 系列相机（如 D435, D435i, D455 等），这段代码是非常标准且通用的。**

开机到拍摄到保存图片到关机的过程

### 2.yolo_detect.py

识别相应的类。

### 3.publish_grasp_sort.py

读取npy文件，发布抓取的TF。

### 4.demo_sorting.py

负责从“看到一张图”到“算出哪里能抓”的全过程。里面会进行加载模型，读图，算抓取，剔除碰撞，排序，保存结果。（graspnet demo的二开主要是为了保存图片）

### 5.filter_orientation_sort.py

筛选掉那些**姿态不合格**的抓取点。比如：手爪朝天的、太水平的、可能会撞桌子的。

### 6.append_uv_sorting.py

为剩下的抓取点计算 (U, V) 像素坐标（需要做矩阵乘法运算）。

### 7.yolo_detect.py

分割出特定物体

### 8.object_mask.py

负责“精修图”的，也就是把 YOLO 给的粗糙框框变成精确的轮廓。

### 9.filter_mask_sorting.py

可以理解为“匹配”或者“验证”

### 10.visualizegrasp_sort.py

数据可视化

### 11.move_to_grasp_sort.py

发布tf给机械臂

## 2.main_deterministic

1.它大部分时间都在不断地，这时候它只做 YOLO 检测，不做 SAM，也不做 GraspNet（为了流畅）。

2.接收信号，它可以被两种方式激活：你在键盘上按Enter2.接收到 SIGUSR1信号（这是由外部 Shell 脚本发送的，意味着机械臂已经归位，可以说开始了）。

3.全力思考 ，一旦被激活，它就开始疯狂计算：SAM 介入；Planner 决策；选出那个 ID；计算抓取，根据不同的物体应用graspnet ；vertical。

4.关键结果：它把算出来的抓取点保存到了grasp_points.json和selected_grasp.json

5.任务结束：算完之后，它就打印一句 Success，然后进入下一次循环。

### 1.camera_ros.py

ros2订阅RS话题类

### 2.detection.py

YOLO_World类

### 3.segmentation.py

不仅仅是抠图，它是机器人的“空间推理大脑”。如果说 detection 是“认出名字”，那这个文件就是“看懂关系”。

它主要做了三件事，一件比一件高级：智能修图 ,它不是生硬地用 SAM 的结果，而是根据物体类型做了,对于碗 (Bowl): 它会把中间挖空，只保留一圈甜甜圈状的边缘 。对于杯子 (Cup): 它会把掩码变大一圈 。判断位置关系：它通过计算掩码的重叠率 来推理物体关系：如果 **A 的面积 < B 的面积** 且 **A 在 B 内部**。它就会得出一个结论：**A 在 B 里面 (INSIDE)**。它能自己看出来“苹果在碗里”，并把这个事实写列表里。判断堆叠顺序 ：它读取深度图的 Z 值，计算每个物体的**平均深度**。如果 A 和 B 在 2D 上重叠，但 A 离相机更近（深度值更小）。它就得出一个结论：**A 压在 B 上面 (ON TOP OF)**。*例子*：如果勺子压着盘子，它会告诉 Planner：“先把上面的勺子拿走，否则盘子拿不出来。”

### 4.deterministc_planner.py

比较复杂后续研究，本质上就是做一些处理决定先抓哪个。

### 5.visualization.py

把识别的结果可视化出来

### 6.vertical.py

传统的视觉抓取算法

### 7.graspnet.py

把上面的数据拿过来算出抓取点，也就是6/7二选一

## 3.ros阶段

### 1.scene_spawner_mov.py

movelt2添加障碍物

### 2.scene_rviz_pub.py

接收相机图片，转换成点云发给 RViz 看。

### 3.markers_pub

把所有抓取点画成箭头或手爪模型，发给 RViz。

### 4.moveit_pub

监听 JSON 文件，把所有抓取点发布为 ROS PoseArray。（它并不直接控制 MoveIt 运动）。

### 5.selected_grasp_pub

监听 selected_grasp.json，把最终决定的那个抓取点发给 TF 和 RViz。





