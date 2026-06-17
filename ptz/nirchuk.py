from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
from isaac_core_dev_kit.udp.udp_bot import UdpBot
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture
import isaac_core_dev_kit.dev_utils as core_utils
import time

# CORE_PATH = "/home/ofer/clones/drones_gen"
# VID_LENGTH_SEC = 10

# core_utils.safe_rclpy_init()

# with HostIsaacManager(core_path=CORE_PATH, com_udp=True, show_isaac_logs=False, bbox_publisher=False):

#     video_capture_node = VideoCapture()
#     camera_point_sender = OnePointSender(lat=32.19965, lon=35.30593, alt=1000.0,
#                                roll=0.0, pitch=0.0, yaw=0.0)
#     # drone_point_sender = OnePointSender(lat=32.1997, lon=35.30593, alt=1000.0,
#     #                             roll=0.0, pitch=0.0, yaw=0.0, udp_port=33335)
#     drone_bot = UdpBot(start_lat=32.2037, start_lon=35.30593, start_alt=1000.0,
#                        start_roll_d=0.0, start_pitch_d=0.0, start_yaw_d=0.0,
#                        udp_port=33335)

#     video_capture_node.spin()
#     camera_point_sender.run(blocking=False)
#     drone_bot.run(blocking=False)
#     # drone_point_sender.run(blocking=False)

#     time.sleep(15)      # Wait for scene to render
    
#     video_capture_node.start_capture()

#     time.sleep(VID_LENGTH_SEC)

#     video_capture_node.stop_capture()
#     video_capture_node.save_data_to("./data/gg.mp4", 45)

#     camera_point_sender.close()
#     # drone_point_sender.close()
#     drone_bot.close()
#     video_capture_node.shutdown()




time_for_fifty = 2.0


# camera_point_sender = OnePointSender(lat=32.19965, lon=35.30593, alt=1000.0,
#                             roll=0.0, pitch=0.0, yaw=0.0)

drone_bot = UdpBot(start_lat=32.20647, start_lon=35.29034, start_alt=540.0,
                    start_roll_d=0.0, start_pitch_d=0.0, start_yaw_d=0.0,
                    udp_port=33335, send_rate_hz=30.0)

# camera_point_sender.run(blocking=False)
drone_bot.run(blocking=False)

def drone_to_start():

    drone_bot.move_right_left(distance_m=200.0, duration_s=time_for_fifty)
    drone_bot.move_forward_backward(distance_m=50.0, duration_s=time_for_fifty)
    drone_bot.move_up_down(distance_m=50.0, duration_s=time_for_fifty)

def scene_1():

    drone_to_start()
    drone_bot.move_forward_backward(distance_m=-400.0, duration_s=(time_for_fifty * 4))

def scene_2():
    
    drone_to_start()
    drone_bot.move_forward_backward(distance_m=-150.0, duration_s=time_for_fifty)
    drone_bot.move_to_point(target_lat=32.20647, target_lon=35.29034, target_alt=530.0,
                            target_yaw_d=0.0, target_roll_d=0.0, target_pitch_d=0.0,
                            duration_s=(time_for_fifty * 4))

scene_2()
    

# for i in range(3):

#     drone_bot.move_forward_backward(distance_m=100, duration_s=time_for_fifty)

#     drone_bot.move_right_left(distance_m=50.0, duration_s=time_for_fifty)
#     drone_bot.move_up_down(distance_m=50.0, duration_s=time_for_fifty)

#     drone_bot.move_right_left(distance_m=-100.0, duration_s=(time_for_fifty * 2))
#     drone_bot.move_up_down(distance_m=-100.0, duration_s=(time_for_fifty * 2))

#     drone_bot.move_right_left(distance_m=100.0, duration_s=(time_for_fifty * 2))
#     drone_bot.move_up_down(distance_m=50.0, duration_s=time_for_fifty)

#     drone_bot.move_right_left(distance_m=-50.0, duration_s=time_for_fifty)



# camera_point_sender.close()
# drone_bot.close()


# drone_bot.move_forward_backward(distance_m=30.0, duration_s=3.0)
