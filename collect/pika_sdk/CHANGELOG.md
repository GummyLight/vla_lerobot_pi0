## Version 0.1.1
### Features
- sense：增加了获取夹爪开合距离的接口。
- gripper：根据电流值设置夹爪夹持力度，确保其不会发生过流保护。
- examples：增加2种通过sense的夹爪值（angle、distance）映射到gripper中控制夹爪示例（sense_control_gripper.py）。
- 给串口加上了正则表达式。


### Bug Fixes
- 修复了 `xyzQuaternion2matrix` 循环导入的问题。
- 给串口增加了缓冲区长度，修复了因数据不断堆积导致处理速度过慢的问题。



### Other Changes
None
