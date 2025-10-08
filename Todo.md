# TailsitterRL MuJoCo 可视化集成纲领

## 项目概述
将实验生成的CSV轨迹数据集成到MuJoCo中进行3D可视化显示，提供直观的tailsitter飞行轨迹观察工具。

## 当前数据结构分析

### 训练日志数据 (logs/bptt_hover/training_log.csv)
- **内容**: 训练指标数据
- **字段**: epoch, loss, mean_position_error, final_position_error, mean_velocity_norm, mean_action_norm
- **用途**: 监控训练收敛性和性能

### 轨迹数据 (内存中)
- **来源**: HoverBPTTExperiment.evaluate() 方法 (hover_bptt.py:227-252)
- **字段**: time, positions, velocities, actions
- **格式**: PyTorch张量，包含完整的状态轨迹
- **当前状态**: 仅用于绘图，未持久化存储

## 集成方案设计

### 阶段1: 数据持久化
- [ ] 修改 hover_bptt.py 添加轨迹CSV保存功能
- [ ] 设计轨迹数据格式规范
- [ ] 实现轨迹数据导出机制
- [ ] 添加配置选项控制数据保存

### 阶段2: MuJoCo模型准备
- [ ] 获取或创建tailsitter的MuJoCo XML模型文件
- [ ] 验证模型参数与实验动力学一致性
- [ ] 确保状态空间映射正确性
- [ ] 测试模型加载和基础仿真

### 阶段3: 可视化工具选择与实现

#### 选项A: mujoco-python-viewer (实时播放)
- [ ] 安装和配置mujoco-python-viewer
- [ ] 实现CSV轨迹读取和解析
- [ ] 创建轨迹播放器脚本
- [ ] 添加播放控制功能 (暂停、速度调节、重置)

#### 选项B: mjc_viewer (HTML输出)
- [ ] 安装mjc_viewer依赖
- [ ] 实现轨迹数据转换到MuJoCo格式
- [ ] 生成可交互的HTML可视化文件
- [ ] 集成到项目工作流中

#### 选项C: 自定义可视化解决方案
- [ ] 基于现有MuJoCo Python API
- [ ] 实现自定义轨迹渲染器
- [ ] 添加多视角和相机控制
- [ ] 集成轨迹分析工具

### 阶段4: 功能增强
- [ ] 支持多条轨迹对比显示
- [ ] 添加轨迹误差可视化
- [ ] 实现实时性能指标显示
- [ ] 支持轨迹编辑和修改功能

### 阶段5: 工作流集成
- [ ] 将可视化集成到实验脚本
- [ ] 添加命令行参数控制可视化
- [ ] 实现批量轨迹处理
- [ ] 创建使用文档和示例

## 技术实现要点

### 数据格式设计
```
trajectory.csv:
time, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,
vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z,
action_thrust, action_wx, action_wy, action_wz
```

### 状态空间映射
- **位置**: data.qpos[0:3] ← trajectory[pos_x, pos_y, pos_z]
- **姿态**: data.qpos[3:7] ← trajectory[quat_w, quat_x, quat_y, quat_z]
- **速度**: data.qvel[0:3] ← trajectory[vel_x, vel_y, vel_z]
- **角速度**: data.qvel[3:6] ← trajectory[ang_vel_x, ang_vel_y, ang_vel_z]
- **控制**: data.ctrl ← trajectory[action_*]

### 文件结构规划
```
vtol_rl/
├── experiments/
│   ├── hover_bptt.py (修改)
│   └── visualization/
│       ├── trajectory_player.py
│       ├── csv_to_mujoco.py
│       └── compare_trajectories.py
├── config/
│   └── mujoco_models/
│       └── tailsitter.xml
└── scripts/
    └── visualize_trajectory.py
```

## 预期成果
1. **完整的轨迹可视化工具链**
2. **标准化的数据格式和接口**
3. **多种可视化选项支持**
4. **易于使用的命令行工具**
5. **详细的使用文档和示例**

## 风险与考虑
- **模型一致性**: 确保MuJoCo模型与实验动力学匹配
- **性能优化**: 大规模轨迹数据的流畅播放
- **兼容性**: 不同MuJoCo版本的API差异
- **用户体验**: 直观的界面和控制方式