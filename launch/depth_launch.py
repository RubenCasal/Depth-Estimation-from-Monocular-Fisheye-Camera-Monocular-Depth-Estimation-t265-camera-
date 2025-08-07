import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package="depth_estimation_dac",
            executable = "t265_node",
            name= "t265_node",
            output="screen"
        ),

    launch_ros.actions.Node(
                package='depth_estimation_dac',
                executable='depth_node.py',
                name='depth_node',
                output='screen',
                parameters=[],
                arguments=[]
            )

    ])
        