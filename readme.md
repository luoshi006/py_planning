# PyPlanning

A python motion planning demo

## TODO - sim
- [ ] `collision_cir_poly` 对于锐角多边形可能有问题，待测试
  - [ ] 需要把 碰撞点 可视化出来
  - [ ] 有可能穿墙？
- [ ] `motion_diff` 运动模型不精确
- [ ] `arrive` 到点判断升级

## TODO - BSplinePath
- [x] 拟合样条曲线（给定 wpt, vel）
  - [x] 给定 wpt 的位置、端点方向
  - ![](fig/path_generator_bspline_default.png)
- [ ] BSpline 管理
  - [ ] MINVO 避障检测
  - [ ] 弧长参数化
- [ ] Keeprun 动态更新

