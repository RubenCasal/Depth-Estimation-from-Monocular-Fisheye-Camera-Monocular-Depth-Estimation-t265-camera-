#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import open3d as o3d
import os
import sys

# Tu clase depth_estimator ya implementada
sys.path.append(os.path.dirname(__file__))
from depth_estimator import DepthEstimator


class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.bridge = CvBridge()
        self.depth_estimator = DepthEstimator()

        self.subscription = self.create_subscription(
            Image, "/rs_t265/fisheye_left", self.image_callback, qos_profile_sensor_data
        )

        self.pose_subscription = self.create_subscription(
            Odometry, "/rs_t265/odom", self.pose_callback, qos_profile_sensor_data
        )

        self.image_publisher = self.create_publisher(Image, "/rs_t265/depth_estimator_node", 1)
        self.pcd_publisher = self.create_publisher(PointCloud2, "/rs_t265/pointcloud", 1)

        self.get_logger().info("✅ Depth node initialized")

        self.global_map_xyz = []
        self.global_map_rgb = []
        self.current_pose = None
        self.prev_pose = None

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.current_pose = {
            "translation": np.array([pos.x, pos.y, pos.z]),
            "rotation_quat": np.array([ori.x, ori.y, ori.z, ori.w]),
        }

    def quaternion_to_rotation_matrix(self, q):
        return R.from_quat(q).as_matrix()

    def image_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("No pose received yet.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth, mask, erp_image = self.depth_estimator.estimation_pipeline(cv_image)
            depth_image = self.depth_estimator.get_depth_image(depth)

            # Publicar imagen de profundidad
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="bgr8")
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = "t265_frame"
            self.image_publisher.publish(depth_msg)

            # Nube local
            pcd = self.depth_estimator.reconstruct_pcd_erp(depth, mask)
            xyz_local = pcd.reshape(-1, 3)

            rgb_img = erp_image.squeeze(0).cpu().numpy()
            rgb_local = self.depth_estimator.process_rgb_point_cloud(rgb_img).reshape(-1, 3)

            valid_mask = xyz_local[:, 2] > 0
            xyz_local = xyz_local[valid_mask]
            rgb_local = rgb_local[valid_mask]

            # Pose actual
            R_cam = self.quaternion_to_rotation_matrix(self.current_pose["rotation_quat"])
            t_cam = self.current_pose["translation"].reshape(3, 1)

            # Filtro de movimiento
            if self.prev_pose is not None:
                delta_t = np.linalg.norm(t_cam - self.prev_pose["translation"])
                delta_r = np.linalg.norm(R.from_matrix(R_cam.T @ self.prev_pose["rotation"]).as_rotvec())
                if delta_t < 0.05 and delta_r < 0.02:
                    self.get_logger().info("🔁 Pose change too small — skipping point cloud update.")
                    return

            # Transformación global
            xyz_global = (R_cam @ xyz_local.T + t_cam).T
            self.global_map_xyz.append(xyz_global)
            self.global_map_rgb.append(rgb_local)

            # Actualiza pose anterior
            self.prev_pose = {"translation": t_cam, "rotation": R_cam}

            # Fusionar y downsample
            merged_xyz = np.vstack(self.global_map_xyz)
            merged_rgb = np.vstack(self.global_map_rgb)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_xyz)
            pcd.colors = o3d.utility.Vector3dVector(merged_rgb / 255.0)

            voxel_size = 0.05  # metros
            pcd_down = pcd.voxel_down_sample(voxel_size)

            xyz = np.asarray(pcd_down.points)
            rgb = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)

            # Publicar nube PointCloud2
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "odom"
            pc_msg = self.create_pointcloud2_msg(xyz, rgb, header)
            self.pcd_publisher.publish(pc_msg)

            self.get_logger().info(f"📦 Published filtered point cloud with {xyz.shape[0]} points")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to process image: {e}")

    def create_pointcloud2_msg(self, points, colors, header):
        rgb_packed = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        ).astype(np.uint32).view(np.float32)

        cloud_data = np.column_stack((points, rgb_packed)).astype(np.float32)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, cloud_data)


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
