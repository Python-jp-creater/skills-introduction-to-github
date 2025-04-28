import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが検出できません")
        return

    MIN_AREA = 100
    MAX_AREA = 5000  # 追加：LEDより大きすぎたら無視
    THRESHOLD_BRIGHTNESS = 200  # 追加：もっと高めに設定
    ROUNDNESS_THRESHOLD = 0.7  # 追加：丸っぽい形だけ

    # カメラからフレーム取得
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # しきい値処理で明るい部分を強調
        _, thresh = cv2.threshold(blurred, THRESHOLD_BRIGHTNESS, 255, cv2.THRESH_BINARY)

        # モルフォロジー処理でノイズを除去
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # 輪郭検出
        contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        led_positions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue  # 小さすぎる・大きすぎるものは無視

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # 丸さの判定
            roundness = 4 * np.pi * (area / (perimeter * perimeter))
            if roundness < ROUNDNESS_THRESHOLD:
                continue  # 丸っぽくない形は無視

            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            # LED領域の平均輝度を確認
            mean_value = cv2.mean(gray, mask=mask)[0]

            if mean_value > THRESHOLD_BRIGHTNESS:
                x, y, w, h = cv2.boundingRect(contour)
                led_positions.append((x, y, w, h))

        # LEDの位置をX座標でソート（順番通りに並べる）
        led_positions.sort(key=lambda pos: pos[0])

        # LEDの矩形を描画
        for idx, (x, y, w, h) in enumerate(led_positions):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = str(idx + 1)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Threshold', morph)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
