# _detector.py

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import os
import collections

# 載入設定
config_path = os.path.join(os.path.dirname(__file__), 'conditions.json')
with open(config_path, 'r') as temp:
    config=json.load(temp)
# 以上，從現在開始我的json叫做config，以後我們都用這個名字做讀取

# 設定闊值(以防萬一我們json忘了設定)
# threshold = 臨界點 = thresh(縮寫)

EAR_THRESHOLD       = config.get('ear_threshold', 0.20) # 眼睛縱橫
PITCH_THRESHOLD     = config.get('pitch_threshold', 4) # 仰角
MIN_FACE_WIDTH      = config.get('min_face_width', 50) # 臉寬
MAX_FACES           = config.get('max_faces', 5) # 最大臉部(以免抓到太遠的目標，導致系統混亂判斷)
BUFFER_SIZE         = config.get('buffer_size', 12) # 緩衝值
SLEEP_FRAME_THRESH  = config.get('sleep_frame_thresh', 20) # 睡眠幀數
IOU_THRESHOLD       = config.get('iou_threshold', 0.2) 
# 當前的臉部與上一幀的臉部的邊界框的 IOU 大於時，才是同一個臉部。(小於代表臉交比小)

HEAD_MOVE_THRESH    = config.get('head_move_threshold', 0.25) # 頭部移動
HAND_MOVE_THRESH    = config.get('hand_move_threshold', 0.30) # 手部移動
SLEEP_GRACE_PERIOD = config.get('sleep_grace_period', 2.0) 
# 忘了為甚麼我加這行，後續我檢查也沒用到，可能是漏了或我睡著了

SLEEP_STABILITY_THRESHOLD = config.get('sleep_stability_threshold', 3.0)  # 睡眠穩定
AWAKE_STABILITY_THRESHOLD = config.get('awake_stability_threshold', 5.0)  # 清醒穩定
MIN_SLEEP_DURATION = config.get('min_sleep_duration', 3.0)  # 最小睡眠報告時間（秒）


# 狀態追蹤
face_detected_buffer = collections.deque(maxlen=BUFFER_SIZE)
"""
設定雙端的queue，因為我們需要利用前一幀去判斷
用來記錄最近的 BUFFER_SIZE=12 幀中是否有偵測到任何臉部
每一幀的結果（1 代表有臉部，0 代表沒有）會被加到這個佇列中
"""

# 以下空字串的鍵都是 face_id
sleep_counters = {} 
# 計算一個人連續多少幀在睡覺，當計數達到 SLEEP_FRAME_THRESH 時，可能會觸發初步的睡眠判斷
prev_sleep_status = {} # 布林值 (上幀睡覺 / 清醒)
face_tracking = {} 
# 當偵測到多個臉部時需要根據 face_id 識別誰是誰，而不是每次都當成新的臉
pose_tracking = {}  # 追蹤身體關鍵點 (頭與手)
sleep_entry_time = {}  # 紀錄睡覺時間
eye_open_counters = {}         # 睜眼幀數
eye_closed_start_time = {}     # 閉眼開始時間點
sleep_transition_counter = {}  
# 追蹤睡眠狀態和清醒狀態的轉換穩定性，以供 confirmed_sleep_status 使用
confirmed_sleep_status = {}    
# 布林值，當我們透過sleep_transition_counter確定已經睡著了
sleep_suspects_buffer = {}     # 暫存等待確認的可疑對象
sleepy_ID = 1 # 睡覺的id

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh( #臉
    static_image_mode=False, # 影片
    max_num_faces=MAX_FACES, # json的臉最多設定
    refine_landmarks=True, # 為了模型更準確就開
    min_detection_confidence=0.3, # 偵測可信度
    min_tracking_confidence=0.3 # 追蹤可信度
)

# segmentation = 分割
pose = mp_pose.Pose( #姿勢
    static_image_mode=False,
    model_complexity=2, #選最高的就對了，跑不動是我的問題不是黃仁勳的問題
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 索引
POSE_IDX = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye_outer': 263,
    'right_eye_outer': 33,
    'mouth_left': 287,
    'mouth_right': 57
}
# 這上下兩個座標順序是一一對應的，不能亂改
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), #代表我們用鼻尖當成 3D 臉的中心點
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype='double')

LEFT_EYE_IDX = [33, 159, 145, 133]
RIGHT_EYE_IDX = [362, 386, 374, 263]
HEAD_KP = [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR]
HAND_KP = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, 
           mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

# bbox = bounding box 邊界框
# iou = Intersection over Union 交集比
def compute_iou(bbox1, bbox2):
    """ 
    x: 邊界框左上角的 x
    y: 邊界框左上角的 y
    w: 寬。
    h: 高。
    """
    x1, y1, w1, h1 = bbox1 
    x2, y2, w2, h2 = bbox2
    
    # 計算交集區域的寬和高
    # 如果 right < left 或 bottom < top，表示沒有交集，寬或高會是負的，取 max(0, ?) 確保面積不為負的
    x_intersect = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    """
    取 min 代表現在是右邊， max 代表現在是左邊，你就想我們取最保守值就好
    bbox=
            (x, y)-------
            |           |
            |           |
            |___________|(x + w, y + h)
    """
    y_intersect = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    
    # 計算交集面積
    area_intersect = x_intersect * y_intersect

    # 計算兩個邊界框各自的面積
    area1 = w1 * h1
    area2 = w2 * h2

    # 交集面積
    area_union = area1 + area2 - area_intersect
    
    return area_intersect / (area_union + 1e-6) # IoU 公式，+1e-6 只是為了防止除 0 無意義
    # 所以會回傳一個 float，表示比例
"""
主要用於臉部追蹤。
當系統在目前影格偵測到一個臉部時，它會跟上一影格已經追蹤到的臉部比較 IoU。
如果 IoU 大於某個閾值 (IOU_THRESHOLD)，系統就認為這是同一個臉，從而可以持續追蹤。
"""

# ear = Eye Aspect Ratio (眼睛縱橫比)
def compute_ear(landmarks, w, h):
    # 計算單隻眼睛的 ear
    # indices = 索引
    def _ear(indices):
        # 檢查所有需要的關鍵點是否都在影像範圍內 (0 到 1 之間)
        for i in indices: 
            if not (0 <= landmarks[i].x <= 1 and 0 <= landmarks[i].y <= 1):
                return 1.0  # 若眼睛 landmark 不完整，就假裝他睜眼吧哈哈
        #理論來說 landmark 的關鍵點應該是會 >=0 <=1 但是還是以防萬一影像不清楚，mp 卻硬抓導致系統嚴重錯判

        # p1, p2, p3, p4 代表眼睛的特定關鍵點
        # landmarks 輸出的一張臉的所有關鍵點列表
        # p1: landmarks[indices[0]] (水平點1, 眼角外側)
        # p4: landmarks[indices[3]] (水平點2, 眼角內側)
        # p2: landmarks[indices[1]] (垂直點1, 上眼瞼中間)
        # p3: landmarks[indices[2]] (垂直點2, 下眼瞼中間)
        p1 = np.array([landmarks[indices[0]].x * w, landmarks[indices[0]].y * h])
        p2 = np.array([landmarks[indices[1]].x * w, landmarks[indices[1]].y * h])
        p3 = np.array([landmarks[indices[2]].x * w, landmarks[indices[2]].y * h])
        p4 = np.array([landmarks[indices[3]].x * w, landmarks[indices[3]].y * h])
        # 轉換為影像中的實際座標： p_x = landmark.x * w，p_y = landmark.y * h

        #         線性代數，用向量算
        return np.linalg.norm(p2 - p3) / max(np.linalg.norm(p1 - p4), 1e-6)
        #   毆基里得距離公式    垂直距離差     水平距離差(雖然其實根本沒什麼變但是就當分母)
        """
        當眼睛睜得大時，垂直距離相對較大，EAR 值就比較大。
        當眼睛閉上時，垂直距離變得非常小（趨近於零），EAR 值也就跟著變小。
        而眼睛的水平距離（從眼角到眼角）在睜眼和閉眼時變化不大。

        我知道你可能會想阿我們不就直接取「垂直距離」就好?
        我也想過，但是我們必須先給他 正規化，以免說今天我只單純設定一個闊值，誰大就誰清醒
        誰小就誰睡覺，若是 人離的遠 或 眼睛小 就可能導致明明他睜眼但因為眼睛像素值小而被判定錯誤
        所以我們就統一除以水平距離，這樣就可以確保不管其他因素，我的判斷都是按照比例處理。
        """
    
    left_ear = _ear(LEFT_EYE_IDX)
    right_ear = _ear(RIGHT_EYE_IDX)
    return min(left_ear, right_ear)  # 任一眼閉起來就觸發

# pitch = 俯仰角
# yaw = 左右搖頭角度
# roll = 左右扭頭角度
# 根據偵測到的臉部關鍵點，估計出頭部在三維空間中的姿態。
def estimate_head_pose(landmarks, w, h):
    try:
        # 先把臉部關鍵點算出來。邏輯和上面那個眼睛的一樣
        image_points = np.array([
            (landmarks[POSE_IDX[k]].x * w, landmarks[POSE_IDX[k]].y * h)
            for k in POSE_IDX
        ], dtype='double') # 資料型態是 double (dtype = data type)
        """
        先跟你解釋為何我們這裡不像剛剛多檢查關鍵點是否 >=0 <=1。
        因為我們現在檢查的是鼻子嘴巴等等，相對眼睛會更大更好辨識。
        再來，我們現在做的事情有點像是構築出一個 3D 臉部模型，如果只是因為一兩個點錯誤其實還好
        但如果很多點位置都是亂的然後 mp 還硬抓，就會觸發 except。所以我們才用try... except...
        """
        focal_length = w # 假設焦距等於影像寬度，我知道不準確
                         # 目前先這樣，你可以考慮改用「相機校準」，不過那部份我不懂
        center = (w / 2, h / 2) # 光中心點假設正中央
        camera_matrix = np.array([
            [focal_length, 0, center[0]], # [0] 光心 x 座標
            [0, focal_length, center[1]], # [1] 光心 y 座標
            [0, 0, 1]
        ], dtype='double') 
        # 假設焦距約等於影像寬度，光心在影像中心，無扭曲，描述相機基本成像特性的矩陣。

        dist_coeffs = np.zeros((4, 1)) # 4行1列的全0陣列
        # 相機鏡頭是由多個曲面鏡片組成的，這些鏡片在折射光線時並不完美，會導致光學畸變。
        # 這裡設為 0 ，因為我沒有進行相機校準，所以我假設沒有畸變，你可以去改
        _, rot_vec, __ = cv2.solvePnP(MODEL_POINTS, image_points, 
                                                    camera_matrix, dist_coeffs)
    #   _ : 計算成功與否 ; rot_vec : 旋轉向量 ; __ : 物體對攝影機的相對位置
        """
        solvePnP :
        OpenCV 的 SolvePnP 可以求解一物點 3D 物空間坐標與其對應像點 2D 像坐標的轉換關係
        在某些條件已知時，這個轉換關係可以延伸為相機的 pose，使其成為一種定位定向方法，
        與攝影測量的後方交會有異曲同工之妙。https://mapostech.com/ros-opencv-solvepnp/
        (其實網路上都有公式可以直接抄)
        """
        rmat, ___ = cv2.Rodrigues(rot_vec) 
        # 將 旋轉向量 改成 3X3 矩陣，後續只要乘上 旋轉矩陣 就可以做幾何變換和角度提取
        # ___ : 雅可比矩陣，因為看起來很難而且也們用到，所以 skip
        # rmat = rotation matrix 旋轉矩陣
        pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
        yaw = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        roll = math.degrees(math.atan2(rmat[2,0], rmat[2,2]))
        # atan2 = 反正切值，得到一個角度。利用各自的正負號來計算角度和決定象限
        #         計算點 (x, y) 相對於原點 (0,0) 和 X 軸正方向所形成的角度。
        #         atan2 的返回值通常是一個介於 −π 到 π（−180 到 180）之間的弧度值

        # 公式 : 角度 = 弧度(atan2) * (180/pi)
        # 俯仰(pitch)、偏航(yaw)、翻轉(roll) --- 歐拉角
        return pitch, yaw, roll
    
    except Exception as e:
        return 0, 0, 0  # 在出錯時返回預設值
    
def check_lost_faces(_, suspects):
    """
    檢查之前追蹤但現在丟失的臉部。
    如果一個先前被標記為睡覺的臉部丟失超過特定時間，我們就懷疑他可能臉部朝向其他面睡覺。
    同時，此函數也會清理與這些丟失臉部相關的追蹤資料。
    """
    current_time = time.time() # 獲取當前時間點
    
    # 更新或清除臉部追蹤記錄
    lost_keys = [] # 初始化一個空列表，用來收集那些確定已經丟失的 face_id

    # 遍歷目前所有正在追蹤的臉部 (face_tracking 是一個全域字典)
    # 鍵(face_id) : {值(所有詳細資訊)}
    for face_id, data in face_tracking.items():
        # data 字典中儲存了每個 face_id 的資訊，例如：
        #       data['last_seen']: 上一次偵測到這個臉部的時間戳
        #       data['bbox']: 這個臉部最後出現時的邊界框 (x, y, 寬, 高)

        if current_time - data['last_seen'] > 10:  # 10秒內未見，視為丟失

            # prev_sleep_status 是一個全域字典，儲存了每個 face_id 先前的睡眠狀態 (True/False)
            if prev_sleep_status.get(face_id, False): 
            #這句話是在 prev_sleep 找 face_id，沒找到就用 False，就不會進 if 
                x, y, fw, fh = data['bbox']
                suspects.append({
                    'face_id': f"lost_{face_id}", # 給丟失的臉部一個新的ID標識
                    'bbox': (x, y, fw, fh), # 記錄其最後的邊界框
                    'timestamp': current_time, # 記錄它被標記為可疑事件的時間
                    'reason': '臉部丟失但先前被檢測為睡覺' # 記錄原因
                })
                print(f"[DEBUG] 臉部丟失但先前睡覺 -> {face_id}")

            #只要它丟失了，就將其 face_id 加入到 lost_keys 列表中
            lost_keys.append(face_id)
    
    # 清除過期記錄
    for key in lost_keys:
        # .pop(key, None) 中的 None 表示如果 key 不存在，不會引發錯誤，而是返回 None
        face_tracking.pop(key, None) 
        sleep_counters.pop(key, None) 
        prev_sleep_status.pop(key, None) 
        pose_tracking.pop(key, None)  
        sleep_entry_time.pop(key, None) 
        eye_open_counters.pop(key, None)
        eye_closed_start_time.pop(key, None)
        sleep_transition_counter.pop(key, None)
        confirmed_sleep_status.pop(key, None)
        sleep_suspects_buffer.pop(key, None)

def detect(frame): #每幀每幀處理
    global sleepy_ID
    suspects = [] # 初始化一個空列表，用來收集本幀發現的可疑事件
    h, w = frame.shape[:2] # 獲取影像幀的高度(h)和寬度(w)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe 模型需要 RGB 格式
    face_results = face_mesh.process(rgb) # 前面初始化的 mp_face_mesh.FaceMesh
    pose_results = pose.process(rgb) # 前面初始化的 mp_pose.Pose 模型實例
    
    # 檢查丟失的臉部
    check_lost_faces(None, suspects)
    
    face_present = bool(face_results.multi_face_landmarks)
    # face_results.multi_face_landmarks 是一個列表，如果沒有偵測到臉，它會是 None 或空列表
    # bool : None 或空列表轉換為 False，有內容的列表轉換為 True
    face_detected_buffer.append(1 if face_present else 0)
    #如果當前幀有臉部，則向緩衝區添加 1；否則添加 0(後來檢查發現這部份根本沒用到，因為系統已經修正的更穩定了)

    # 如果當前幀完全沒有偵測到臉部，提前結束
    if not face_present:
        return suspects
    
    # 處理每一個檢測到的臉部
    current_detected_faces = [] # 存放當前幀所有偵測到的臉部的處理後資訊
    for face_idx, lm in enumerate(face_results.multi_face_landmarks): #用 for 去處理所有臉部關鍵點
    #                   enumerate 就是會給()裡的東西變成 「(編號, 資訊)」，在這裡就是 (face_idx, lm)
    # face_idx 是指在這個 for 迴圈我們給他的臨時編號，跟真正的 face_id 不一樣
        # lm = landmark 臉部關鍵點
        # lm.landmark 是一個臉部所有關鍵點的列表
        # 每個關鍵點 i 都有 .x, .y, .z (正規化) (0.0 ~ 1.0)
        xs = [i.x for i in lm.landmark]
        # 這上下兩個意思就是我各自取出每個關鍵點套進陣列後準備進行運算
        ys = [i.y for i in lm.landmark]

        x_min, x_max = int(min(xs)*w), int(max(xs)*w) #算出真正 x 座標，取最大最小
        y_min, y_max = int(min(ys)*h), int(max(ys)*h) #算出真正ｙ座標，取最大最小
        # 提醒一下取最大最小是因為我現在要畫邊界框，所以我們要知道左上(xy最小)、右下(xy最大)
        face_w = x_max - x_min # 寬 weight
        face_h = y_max - y_min # 高 height
        bbox = (x_min, y_min, face_w, face_h) # 組成邊界框元組 (x_左上, y_左上, 寬度, 高度)

        #中心點座標
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2 # 使用 // 進行整數除法
        
        # 將這張臉的相關資訊打包，並添加到 current_detected_faces 列表中
        current_detected_faces.append({
            'idx': face_idx, 
            'bbox': bbox, 
            'center': (center_x, center_y), 
            'landmarks': lm.landmark
            # 保存原始的關鍵點，這樣後續需要用這些關鍵點進行 EAR 或頭部姿態估計時，可以直接取用
        })
        
    """
    接下來最主要是在做臉部追蹤 (Face Tracking)。
    它的目的是為每一張在畫面中出現的臉分配一個持續的、唯一的id (face_id)，即使這張臉在連續的影像幀中位置發生了變化，
    系統也能認出「啊，這還是上一幀的那張臉」。不會每次都刷新新的id給同個人
    """
    # 更新臉部追蹤
    new_face_tracking = {} # 用來儲存當前幀更新後的臉部追蹤資訊
    used_ids = set() # 用來記錄在本幀中已經成功匹配到的舊 face_id，避免被重複分配(set)
    
    # 與已追蹤臉部進行匹配，遍歷「當前幀偵測到的每一張臉」
    for detected_face in current_detected_faces: 
        """detected_face 包含 idx, bbox, center, landmarks""" 

        best_match_id = None # 初始化最佳匹配的舊 face_id 為 None
        max_iou = 0 # 初始化最大交並比 (IoU) 為 0
        detected_bbox = detected_face['bbox'] # 獲取當前偵測到的臉的邊界框
        
        # 嘗試將「當前偵測到的臉」與「上一幀已經追蹤到的臉」進行匹配
        # 用 for 遍歷全域變數 face_tracking 中的每一筆舊的追蹤記錄
        # face_tracking 此時儲存的是上一幀的追蹤結果
        for tracked_id, tracked_data in face_tracking.items():
            # tracked_id: 上一幀某張臉的 face_id
            # tracked_data: 包含該臉上一幀的資訊

            # 計算當前偵測到的臉的 bbox 與某個已追蹤臉的舊 bbox 之間的 IoU
            iou = compute_iou(detected_bbox, tracked_data['bbox'])
            if iou > max_iou and iou > IOU_THRESHOLD and tracked_id not in used_ids:
                # iou > max_iou : 確保重疊度最高
                # iou > IOU_THRESHOLD : IoU 必須大於我們設定的閾值才算有效匹配
                # tracked_id not in assigned_ids: 這個舊的 tracked_id 還沒有在本幀被用掉
                max_iou = iou
                best_match_id = tracked_id
        
        if best_match_id:  # 如果不為 None (表示追蹤到之前的一張臉)
            new_face_tracking[best_match_id] = {'bbox': detected_bbox, 'last_seen': time.time()} #同一 id 但更新時間和框框
            face_id = best_match_id
            used_ids.add(best_match_id) # 將這個 best_match_id 加入 used_ids，表示這 id 在本幀已被使用
        else:
            face_id = f"睡覺人_{sleepy_ID}"
            new_face_tracking[face_id] = {'bbox': detected_bbox, 'last_seen': time.time()}
            sleepy_ID += 1
    
    # 更新追蹤資料
    face_tracking.clear() # 先將儲存著上一幀的舊數據清空
    face_tracking.update(new_face_tracking) # new_face_tracking 包含了當前幀所有被成功追蹤或新分配ID的臉的最新資訊
    
    # 處理每個檢測到的臉部
    for face_idx, lm in enumerate(face_results.multi_face_landmarks):
        # 這次的目的是為每一張臉找到其在 face_tracking 中的持久 ID，並進行詳細分析

        # 重新計算基本資訊
        xs = [i.x for i in lm.landmark]
        ys = [i.y for i in lm.landmark]
        x_min, x_max = int(min(xs)*w), int(max(xs)*w)
        y_min, y_max = int(min(ys)*h), int(max(ys)*h)
        face_w = x_max - x_min
        face_h = y_max - y_min
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 找出對應的 face_id
        for face_id, tracked_data in face_tracking.items():
            tracked_bbox = tracked_data['bbox'] # 已追蹤臉的 bbox
            tx, ty, tw, th = tracked_bbox # bbox 的(x, y, w, h)
            tcx = tx + tw // 2 # 計算中心點 x 座標
            tcy = ty + th // 2 # 計算中心點 y 座標
            
            # 邏輯：如果當前處理的臉 (lm) 的中心點 (center_x, center_y)
            #       與某個已追蹤臉 (face_id) 的中心點 (tcx, tcy) 非常接近，那就當同一個
            if abs(center_x - tcx) < 20 and abs(center_y - tcy) < 20:
                break
        else:
            # 應該不會發生，但為了程式安全性
            face_id = f"face_{face_idx}_{center_x}_{center_y}"
        
        # 計算超出邊界的關鍵點比例
        # 遍歷這張臉的所有關鍵點 (lm.landmark)
        outside_ratio = sum(1 for i in lm.landmark if not (0 <= i.x <= 1 and 0 <= i.y <= 1)) / len(lm.landmark)
        #                                           會計算出「不在」有效範圍內的關鍵點數量   除以總關鍵點數量 (len(lm.landmark))，得到超出邊界的「比例」
        """
        這裡之所以要檢查[0, 1]，是因為如果我們在計算標準差時，把這些極端的、無效的座標值也包含進去
        那麼計算出來的標準差就會被這些離群值嚴重扭曲，變得非常大。
        """
        too_small = face_w < MIN_FACE_WIDTH #臉太小代表太遠，就不要了
        too_outside = outside_ratio >= 0.5 #代表臉有一半以上都在畫面外或找不到關鍵點，也不要
        
        # landmark 集中度過濾 - 判斷臉部形狀是否異常分散
        # 只收集「正常可靠的座標 (0 | 1)」
        valid_xs = [i.x for i in lm.landmark if 0 <= i.x <= 1]
        valid_ys = [i.y for i in lm.landmark if 0 <= i.y <= 1]

        # 計算這些有效座標在 x 和 y 方向上的標準差
        # 如果臉部關鍵點很集中，標準差就小；分散，標準差就大。
        # std() = 算標準差
        x_std = np.std(valid_xs) if valid_xs else 0 
        y_std = np.std(valid_ys) if valid_ys else 0
        too_scattered = (x_std > 0.15) or (y_std > 0.15) # 這些關鍵點太分散，不像一張正常的臉，可能誤把衣服或背景當成臉
        
        # 計算眼睛縱橫比
        ear = compute_ear(lm.landmark, w, h)
        # 估計頭部姿態
        pitch, yaw, roll = estimate_head_pose(lm.landmark, w, h)
        
        # 動態調整 EAR 閾值（根據頭部是否側臉，側臉時適度放寬）
        current_ear_threshold = EAR_THRESHOLD # 首先，使用預設的 EAR_THRESHOLD
        if abs(yaw) > 30 or abs(roll) > 20:
            # abs(yaw) > 30: 偏航角 (左右搖頭) 的絕對值是否超過30度 (你可以再自己調)
            # abs(roll) > 20: 翻滾角 (左右歪頭) 的絕對值是否超過20度 (你可以再自己調)

            """ <<透視收縮原理>>
            如果頭部確實有明顯的側轉或歪斜，我們適度「降低」EAR的判斷閾值 (乘以0.8使其變小)。
            理由是：當人側臉或歪頭時，從攝影機角度觀察到的眼睛縫隙即使在睜開狀態下，其計算出的EAR值也可能會比正臉時自然偏小。
            如果不調整閾值，這種情況下正常的睜眼可能會因為EAR偏小而被誤判為閉眼。
            所以，降低閾值是為了在這種情況下，需要眼睛閉合得更徹底 (EAR值更小)，才將其判斷為 eye_closed，
            從而減少因姿態引起的誤判。這使得對「閉眼」的判斷在側臉時變得「更嚴格」。
            """
            current_ear_threshold *= 0.8 # (你可以再自己調，選你覺得最準的)
        
        eye_closed = ear < current_ear_threshold # 判斷眼睛是否閉合
        head_down = pitch > PITCH_THRESHOLD # 通常，pitch 為正表示低頭。
        head_turned = abs(yaw) > 30 or abs(roll) > 20 # 這個條件與上面用於動態調整 EAR 閾值的條件是相同的
        
        # 記錄閉眼時間 or 重置
        if eye_closed: # 如果當前幀判斷為「閉眼」
            if face_id not in eye_closed_start_time:
                # 如果 eye_closed_start_time 字典中還沒有記錄這張臉 (face_id) 的閉眼開始時間，那代表這是第一次，趕快做紀錄
                eye_closed_start_time[face_id] = time.time()
            eye_open_counters[face_id] = 0  # 既然眼睛是閉著的，那麼「睜眼」的幀數就中斷了。
        else: # 如果當前幀判斷為「睜眼」
            eye_closed_start_time.pop(face_id, None) # 把睡著的紀錄pop掉
            eye_open_counters[face_id] = eye_open_counters.get(face_id, 0) + 1 
            # 增加這張臉連續「睜眼」的幀數。(到時候會依靠這個判斷要不要取消紅框框)
            
        # 額外條件
        y_position_ratio = y_min / h # 判斷臉部是否在畫面中相對位置，y_min是代表最上面，值為0表示在最頂部，1表示在最底部。

        extreme_low_head = y_position_ratio > 0.6 
        # 如果臉部頂端的位置超過了畫面高度的60%，則認為頭部處於極度低垂的狀態 (一樣你也可以調整)
        
        # 用手撐頭，透過下巴與鼻尖距離判斷
        # POSE_IDX['chin'] 和 POSE_IDX['nose_tip'] 是我們預先定義的下巴和鼻尖關鍵點
        chin_y = lm.landmark[POSE_IDX['chin']].y * h
        nose_y = lm.landmark[POSE_IDX['nose_tip']].y * h
        #    .y 是正規化y座標，乘以影像高度 h 轉換為像素y座標。

        chin_nose_distance = abs(chin_y - nose_y) # 計算下巴和鼻尖在距離
        # 當人非常困倦時，可能會無意識地用手撐住下巴，同時由於頸部肌肉放鬆，頭部可能會略微向後仰，以鏡頭角度來說距離會增大
        # 睡著時嘴巴可能會張開，距離增大
        # 用手撐的話可能 mp 會笨笨的去抓手的某個部位當成下巴點，可能增大
        head_propped = chin_nose_distance > 0.4 * face_h  
                                    # 你這部份可以改為我原本寫的 (0.15*h)，改掉只是因為網路說這樣寫會更準，但參數部份要重新調
        
    # 合併睡眠判斷邏輯

        # 主要判斷條件 (primary)，最常見的睡眠特徵：閉眼，並且伴有低頭或頭部顯著偏轉。
        primary   = eye_closed and (head_down or head_turned)

        # 次要判斷條件 (secondary) - 針對臉部過小的情況並且同時有低頭
        secondary = too_small and head_down

        # 第三判斷條件 (tertiary) - 針對臉部位置極低的情況閉眼，並且臉部在畫面中的位置非常靠下
        tertiary  = eye_closed and extreme_low_head

        # 第四判斷條件 (quaternary) - 針對撐頭的情況閉眼，並且我們前面基於「下巴-鼻尖距離」判斷的撐頭為 True
        quaternary = eye_closed and head_propped
        
        # 最終的睡眠候選人判斷 (sleep_candidate)
        """
        只要上述四種主要情況任何一種成立，就初步認為是睡眠候選。
        但是，有一個重要的否決條件：and not too_scattered。
        即使滿足了前面的某個睡眠組合條件，如果這張臉的關鍵點本身被判斷為
        「過於分散」(too_scattered)，意味著這個臉部偵測結果本身就不可靠，
        那麼最終的 sleep_candidate 仍然會是 False。
        這是為了避免基於一個本身就有問題的、不像正常臉型的偵測結果來做出睡眠判斷。
        """
        sleep_candidate = (primary or secondary or tertiary or quaternary) and not too_scattered
        
        
        current_pose = {} # 初始化一個空字典，用來存放當前幀我們感興趣的身體關鍵點的座標
        if pose_results.pose_landmarks: # 如果確實偵測到了姿勢，否則為 None
            for i in HEAD_KP + HAND_KP:
                # i 這個東西成為 HEAD_KP 和 HAND_KP 的枚舉，也就是說我們這裡會遍歷所有像是LEFT_EAR / RIGHT....
                landmark = pose_results.pose_landmarks.landmark[i]
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    current_pose[i] = (landmark.x, landmark.y)
        
        # 取得上一次睡眠狀態
        prev_flag = prev_sleep_status.get(face_id, False) # 有找到目前的臉 or 沒有
        # prev_sleep_status : {face_id: 布林值}

        prev_count = sleep_counters.get(face_id, 0) # 截至上一幀，此人已連續多少幀被標記為 sleep_candidate
        # sleep_counters : {face_id: 整數}
        
        # 更新計數器
        curr_count = prev_count + 1 if sleep_candidate else 0 # 如果當前幀的 sleep_candidate 為 True 就 +1，反之
        sleep_counters[face_id] = curr_count # 把這個 id 睡的時間繼續記錄下來
        
        # 確定當前狀態 加入閉眼超過 5 秒的判斷
        eye_closed_duration = time.time() - eye_closed_start_time.get(face_id, time.time())
        #                                   eye_closed_start_time 是個全域字典 {face_id: 閉眼開始的時間點}

        # 根據「先前是否已處於睡眠狀態 (prev_flag)」來決定如何判斷 curr_flag
        if not prev_flag:
            """
            如果「上一幀的狀態還不是確定的睡眠狀態」(prev_flag 為 False):
            要在此幀被判定為進入睡眠狀態 (curr_flag = True)，需要同時滿足兩個條件：
                1. 連續成為 sleep_candidate 的幀數 (curr_count) 必須達到 SLEEP_FRAME_THRESH。

                2. 並且，眼睛實際閉合的總時長 (eye_closed_duration) 必須達到至少5秒
                  (這個 5.0 你可以改)。
                  這個額外的5秒閉眼時長要求，是為了過濾掉那些雖然持續了一段時間
                  保持了像睡覺的姿態 (滿足了curr_count)，但可能只是長時間眨眼、
                  揉眼睛或其他短暫閉眼的情況，而不是真正的開始進入睡眠。
            """
            curr_flag = curr_count >= SLEEP_FRAME_THRESH and eye_closed_duration >= 5
        else:
            # 若已經進入睡眠，不再檢查閉眼秒數，只看是否仍符合幀數條件
            curr_flag = curr_count >= SLEEP_FRAME_THRESH  # 已經睡著，就不再檢查眼睛秒數

        
        # 進入睡覺狀態
        now = time.time()
        # sleep_transition_counter : 記錄了curr_flag 定義的任何一個狀態的起點(不管睡不睡，雖然我們還是有定義 True / False)
        last_record = sleep_transition_counter.get(face_id, {'last_state': curr_flag, 'start_time': now})
                    #  sleep_transition_counter : { face_id: {'last_state': 布林值, 'start_time': 時間戳} }

        # 檢查當前狀態 curr_flag 是否與上一次記錄的狀態 last_record['last_state'] 發生了變化
        if curr_flag != last_record['last_state']: # 如果從清醒變睡眠，或從睡眠變清醒
            sleep_transition_counter[face_id] = {'last_state': curr_flag, 'start_time': now}
        else:
            duration = now - last_record['start_time'] # 計算當前這個相同的狀態已經持續了多久
            required = SLEEP_STABILITY_THRESHOLD if curr_flag else AWAKE_STABILITY_THRESHOLD

        # 真正的狀態轉換判斷與觸發
        if curr_flag: # 如果當前幀判斷是睡眠 (curr_flag is True)
            sleep_entry_time[face_id] = now # 更新時間
            sleep_duration = now - eye_closed_start_time.get(face_id, now) # # 計算總閉眼時長

            if sleep_duration >= MIN_SLEEP_DURATION: # 如果總閉眼時長也達標
                suspects.append({
                    'face_id': face_id,
                    'bbox': (x_min, y_min, face_w, face_h),
                    'timestamp': now,
                    'reason': '穩定睡眠確認'
                })
                #print(f"[DEBUG] 穩定睡眠確認 -> {face_id} (持續 {sleep_duration:.1f} 秒)")
        
        # 更新狀態
        prev_sleep_status[face_id] = curr_flag # 將當前幀的 curr_flag 存為下一幀的 prev_flag
        pose_tracking[face_id] = current_pose # 更新該 face_id 的最新姿態數據

        # 只在狀態變化時輸出 Debug 信息
        if curr_flag != prev_flag:
            print(f"[DEBUG] {face_id} 狀態變化: {'睡覺' if curr_flag else '清醒'}")
            print(f"  EAR={ear:.2f} (閉眼={eye_closed}) "
                  f"pitch={pitch:.1f} (低頭={head_down}) "
                  f"y_pos={y_position_ratio:.2f} (極低頭={extreme_low_head}) "
                  f"臉部大小={face_w}x{face_h} (太小={too_small}) "
                  f"outside={outside_ratio:.2f} (太外={too_outside}) "
                  f"primary={primary} secondary={secondary} tertiary={tertiary} "
                  f"quaternary={quaternary} 計數={curr_count}/{SLEEP_FRAME_THRESH}")
    
    return suspects
