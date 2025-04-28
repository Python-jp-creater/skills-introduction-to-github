import cv2
import numpy as np

# 画面サイズ
width, height = 640, 480

# LED設定
led_radius = 8
led_color_on = (0, 255, 255)    # 黄色
led_color_off = (50, 50, 50)    # 消灯時はグレー
led_positions = [(100 + i * 100, 240) for i in range(5)]  # 5つ並べる
current_on_indices = []  # 点灯中のLEDのインデックスを管理するリスト

def draw_leds():
    global image

    # 背景クリア
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, pos in enumerate(led_positions):
        color = led_color_on if idx in current_on_indices else led_color_off
        cv2.circle(image, pos, led_radius, color, -1)

    # 光る効果を作成
    glow = np.zeros_like(image)
    for idx in current_on_indices:
        cv2.circle(glow, led_positions[idx], led_radius*2, led_color_on, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=15, sigmaY=15)

    # 合成
    final_image = cv2.addWeighted(image, 1.0, glow, 0.7, 0)
    return final_image

def mouse_callback(event, x, y, flags, param):
    global current_on_indices

    if event == cv2.EVENT_LBUTTONDOWN:
        # クリックした位置に近いLEDを探す
        for idx, pos in enumerate(led_positions):
            dist = np.linalg.norm(np.array(pos) - np.array((x, y)))
            if dist <= led_radius*2:  # LED周囲をクリックしたら有効
                if idx not in current_on_indices:
                    current_on_indices.append(idx)  # 点灯中のLEDに追加
                else:
                    current_on_indices.remove(idx)  # 既に点灯中なら消灯
                break

# ウィンドウ設定
cv2.namedWindow("LEDs")
cv2.setMouseCallback("LEDs", mouse_callback)

# メインループ
while True:
    frame = draw_leds()
    cv2.imshow("LEDs", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
