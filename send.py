import os
import json
import cv2
import time
import datetime
from _detector import detect

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output') # 設定一個名為 output 的子目錄，用來存放程式輸出的檔
#                                         獲取當前腳本檔案所在的目錄路徑
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'suspects_current.json') # 最終輸出的 JSON 檔案路徑
DISPLAY_DURATION = 3.0  # 紅框顯示秒數
SEND_INTERVAL = 1.0     # 最短發送間隔（秒）

previous_suspects_ids = set() # 用來儲存suspects 列表中的所有 face_id
last_send_time = 0 # 記錄上一次發送/寫入 suspects 到 JSON 檔案的時間點
recent_suspects = {} # 用來記錄「最近」被偵測為可疑對象的 face_id 和相關資訊
# 結構：{ face_id: (bbox_元組, 上次更新為可疑的時間點) }
frame_count = 0 # 影格計數器
fps_start_time = time.time() # FPS (每秒影格數) 起始時間點
fps = 0 # 計算得到的 FPS 值

""" 這部分你來，目前輸出 id 很簡單。我要改輸出 id 像 " 睡覺人 i ", i = 1, 2, 3...
前面第 394 行就已經開始記錄 id 了
face_id = f"face_{len(face_tracking) + len(used_ids)}_{detected_face['center'][0]}_{detected_face['center'][1]}"
"""

# 用來記錄每個人目前的 display_id
display_id = {}
next_display_id = {} # 未來的 display_id

def send(suspects):
    """將 suspects 寫入 JSON，僅在列表有變化時觸發"""
    global previous_suspects_ids, last_send_time # 因為要修改所以要宣告 global

    current_time = time.time()
    current_suspects_ids = {s['face_id'] for s in suspects} 
    # 集合推導式 (set comprehension) 快速創建一個只包含 face_id 的集合
    now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] # 取到 3 是因為我發現很多之前的錯誤都是在毫秒部份
    #                                                      去掉最後3位數
    
    # 判斷是否需要執行警示
    # 條件一：有新增的可疑 ID 或有舊的可疑 ID 消失了
    # 條件二：或者suspects 列表為 True(確實有可疑對象需要發送) 並且，距離上次成功發送的時間已經超過了 SEND_INTERVAL
    if (current_suspects_ids != previous_suspects_ids or
        (suspects and current_time - last_send_time >= SEND_INTERVAL)):

        os.makedirs(OUTPUT_DIR, exist_ok=True)  
        # os.makedirs() 可以創建多層目錄，exist_ok=True 表示如果目錄已存也不會報錯
        with open(OUTPUT_FILE, 'w') as temp:  
        # 寫入模式('w')打開指定的 JSON 檔
        # 使用 with 可以確保檔案被安全關閉
            json.dump(suspects, temp, indent=2) # json.dump() 用於序列化，2 使得輸出的 JSON 檔案有縮排，更易讀
            # 2 個空格一個縮排

        # 比較現在和舊的 ID ，找出哪些是新的，哪些是被移除的
        new_suspects = current_suspects_ids - previous_suspects_ids
        # A - B，在集合 A 中，但不在集合 B 中

        removed_suspects = previous_suspects_ids - current_suspects_ids
        # B - A，在集合 B 中，但不在集合 A 中

        # 如果有新增的可疑對象
        if new_suspects: 
            print(f"[{now_str}] [send] 新增可疑對象: {', '.join(new_suspects)}")
        
        # 如果有被移除的對象
        if removed_suspects:
            print(f"[{now_str}] [send] 移除可疑對象: {', '.join(removed_suspects)}")

        # 如果沒有新增也沒有移除，但 suspects 仍然 True
        if not new_suspects and not removed_suspects and suspects:
            print(f"[send] 已更新 {len(suspects)} 個可疑對象的狀態")

        previous_suspects_ids = current_suspects_ids 
        #  將當前處理的 ID 集合 current_suspects_ids 儲存為下一次比較用的 previous_suspects_ids

        last_send_time = current_time 
        # 更新最後成功發送的時間為當前時間


def update_fps():
    """更新程式處理影像的幀率"""
    global frame_count, fps_start_time, fps # 一樣，因為會改，所以更新
    passed_time = time.time() - fps_start_time # 計算從上次 FPS 更新開始到現在所經過的時間
    if passed_time > 1: # 單位 : " 秒 "
        fps = frame_count / passed_time # 計算 FPS： 這段時間內處理的總影格數 (frame_count) / 經過的總時間 (passed_time)
        frame_count = 0 # 重置影格計數器 frame_count 為 0，為下一個計算週期做準備
        fps_start_time = time.time() # 重置 fps_start_time 為當前時間

def GaussianAndEq(image, ksize=(5, 5), sigmaX=0):
    """對影像進行高斯模糊處理"""
    G=cv2.GaussianBlur(image, ksize, sigmaX)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0]) # 對 Y 通道進行直方圖均衡化
    G=cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) # 將均衡化後的影像轉回 BGR 色彩空間
    return G # 返回處理後的影像

def main():
    global frame_count, recent_suspects
    cap = cv2.VideoCapture("1.mp4")
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        got_damm, frame = cap.read() 
        # got_damm : 有 True，搖頭 False
        # frame : 影像
        if frame is not None:
            frame = GaussianAndEq(frame)
        if not got_damm:
            print("無法讀取影像，請確認攝影機")
            break

        frame_count += 1 # 幀數往上加，等等會呼叫 update_fps
        update_fps()

        # 每3幀處理一次，不然會一直跳針
        if frame_count % 3 == 0:
            # 偵測可疑對象
            suspects = detect(frame) # detect 函數會執行所有臉部偵測、追蹤、特徵計算和睡眠判斷邏輯，在 _detector.py
            if suspects !=[]:
                print("[DEBUG] detect() 偵測結果:", suspects)

            # 傳送 JSON 檔案
            if suspects or previous_suspects_ids:
    # 如果幀偵測到了新的可疑情況 or revious_suspects_ids 不為 NULL，
    # 代表上一輪還有可疑對象，但這一輪沒有了，我們仍然需要呼叫 send(suspects) 去更新列表或移除消失的對象
    # 之所以要多呼叫 previous_suspects_ids 是因為我們需要 send() 去 [SEND] 移除可疑對象 這個訊息
                send(suspects)

            # 更新 recent_suspects 時間戳
            now = time.time()
            for s in suspects: # face_id 是鍵
                recent_suspects[s['face_id']] = (s['bbox'], now) # 持續刷新同一張臉(或同 id 每次的紀錄時間)
                

        # 清除過期紅框
        now = time.time()

        # recent_suspects 結構為 { face_id: (bbox_元組, 上次紀錄的時間) }
        # 條件：如果「當前時間 now」減去「上次確認為可疑的時間點 t」大於 DISPLAY_DURATION
        # 就表示這個 face_id 對應的紅框已經顯示超過了預設時長，且未被刷新(未再被列為suspect)，視為過期
        expired = [face_id for face_id, (_, t) in recent_suspects.items() if now - t > DISPLAY_DURATION]
        #                                   t 代表上次確認為可疑的時間點

        # 從 recent_suspects 刪除所有被標記為已過期的 face_id
        for face_id in expired:
            del recent_suspects[face_id] 
            # 我們先收集所有要刪除的鍵 (expired)，然後再單獨遍歷這個鍵列表進行刪除，直接修改怕有問題

        # 畫紅框
        for face_id, (bbox, _) in recent_suspects.items():
            x, y, w, h = bbox
            
            #畫紅框框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
          # cv2.rectangle(影像, 左上角座標, 右下角座標, 顏色BGR, 線條粗細)

            display_name = display_id.get(face_id, f"face_{face_id}")
            # 寫字                     split = 切開(這裡用_)
            cv2.putText(frame, display_name.split('_')[0], (x, y - 10), # [0] 是指前面的 lost or face (check_lost_faces)
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #cv2.putText(影像, 要繪製的文字, 文字的左下角起始座標, 字型, 字型大小的比例, 顏色, 線條粗細)
            # 我們目前賦予給他的 id 是根據 xy 座標

        # 畫 FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Suspicious_people', frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": # 只有當檔案被直接執行時才用，反正比較安全就對了，好習慣
    main()
