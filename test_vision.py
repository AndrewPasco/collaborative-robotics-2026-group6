import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.pub = self.create_publisher(String, '/vision/target', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)
        self.count = 0

    def timer_cb(self):
        if self.count < 3:
            msg = String()
            msg.data = 'banana'
            self.pub.publish(msg)
            self.get_logger().info('Published target: banana')
            self.count += 1
        elif self.count == 3:
            self.get_logger().info('Done publishing')
            self.count += 1
            rclpy.shutdown()

rclpy.init()
node = TestNode()
rclpy.spin(node)
