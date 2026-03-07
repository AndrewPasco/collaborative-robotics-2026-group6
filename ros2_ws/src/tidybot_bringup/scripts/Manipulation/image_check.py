import cv2


def main():
    img = cv2.imread("/home/locobot/collaborative-robotics-2026-group6/ros2_ws/src/tidybot_bringup/scripts/Navigation/debuggging_saves/img_1.jpg")
    if img is None:
        print("Failed to load image.")
        return

    # live display for finding pixel coordinates of interest'
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()