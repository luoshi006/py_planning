# PyPlanning

A python motion planning demo

## BSplinePath

#### 拟合样条曲线
- 参考 CMU-LocalPlanner 生成轨迹算法，使用 Clamped Uniform B-Spline 表示轨迹
  - https://github.com/jizhang-cmu/ground_based_autonomy_basic
- 拟合时，给定等间距采样点，两端点的速度矢量
  - 《计算机辅助几何设计与非均匀有理 B 样条》

|                    生成默认轨迹              |         调整扫掠角度 FOV=180 °               |
|----------------------------------------------|---------------------------------------------|
| ![](fig/path_generator_bspline_default.png)  | ![](fig/path_generator_bspline_fov_180.png) |

#### BSplinePath Impl
- **MINVO 凸包**
  - 参考 MIT-MINVO / Mader 中的 MINVO，代替默认凸包进行避障检测
    - https://github.com/mit-acl/minvo
    - https://github.com/mit-acl/mader
  - 根据机器人形状膨胀凸包
    - 避障凸包示例中，机器人形状为边长 0.2 的正方形
    - 可以根据实际形状配置机器人轮廓点
- **弧长参数化**
  - 拟合默认参数 `u` 与弧长 `s` 之间的关系，实现对弧长的近似等间距采样 `u(s_eq)`
  - 弧长的近似计算采用 romberg 积分
- **轨迹动态更新** [WIP]
  - 动态移除已经行驶过的轨迹
  - 动态添加最新任务轨迹

|                 MINVO 凸包对比        |                    避障检测             |                  弧长参数化 - 等间隔采样                  |
|:-------------------------------------:|:---------------------------------------:|:-----------------------------------------------:|
| ![](fig/bspline_path_convex_hull.png) | ![](fig/path_generator_bspline_obs.png) | ![](fig/bspline_path_convex_hull_param_arc.png) |

## Local Planner
### TODO
- [ ] 轨迹生成失败后的处理
- [ ] 快到 Goal 时候的特殊处理
- [ ] 避障检测的算力优化
  - 目前算力耗时主要在轨迹的凸包计算，将其移到循环外
  - 考虑先使用 **折线膨胀** 预判断是否避障
  - 由于折线本身有不小的误差，所以碰撞系数暂定 0.67

|              MINVO 避障凸包              | ongoing |
|:----------------------------------------:|:-------:|
| ![](fig/bspline_path_collision_hull.png) |   ...   |

## Path Follower
- PurePursuit

## Simulator [WIP]
- [ ] `collision_cir_poly` 对于锐角多边形可能有问题，待测试
  - [ ] 需要把 碰撞点 可视化出来
  - [ ] 有可能穿墙？
- [ ] `motion_diff` 运动模型不精确
- [ ] `arrive` 到点判断升级
