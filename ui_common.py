"""cv2 窗口的通用小工具。

存在的理由：Windows 上 highgui 的标题栏 X 按钮只是把窗口标成"不可见"，
waitKey 循环不会因此退出，界面就"关不掉"了——用户点 X 十几下，窗口还在，
必须回头按 q 或者杀进程。所有应用统一用 wait_key() 取键，它顺手查一次
窗口可见性，窗口没了就返回 ESC，循环自然退出。
"""
import cv2

ESC = 27
NO_KEY = 255            # waitKey 无按键时 cv2 返回 -1，&0xFF 之后是 255


def window_closed(win):
    """窗口是否已经被关掉（X 按钮 / destroyWindow / 后端失效）"""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:                      # 窗口对象已不存在
        return True


def wait_key(win, delay=1):
    """等同 cv2.waitKey，但窗口被关掉时返回 ESC"""
    k = cv2.waitKey(delay) & 0xFF
    if k != NO_KEY:
        return k
    return ESC if window_closed(win) else NO_KEY
