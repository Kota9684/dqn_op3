# DQN for OP3 control

### Environment
- Ubuntu 16.04 LTS

### Requirement
- [Pytorch](https://pytorch.org/)  
- [OpenAI Gym](https://gym.openai.com/)  
- [ROS kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)  
- [ROBOTIS-OP3](https://github.com/ROBOTIS-GIT/ROBOTIS-OP3 )  

### Usage
OP3の起動後に別のterminalでlearning.pyを走らせる  
~~~
$ roslaunch op3_gazebo robotis_world.launch  
$ python learning.py
~~~
### References
 

DQN, ROS kinetic, OP3
