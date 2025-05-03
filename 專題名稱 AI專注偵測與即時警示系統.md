# 專題名稱 AI專注偵測與即時警示系統：遠距與實體課堂中的注意力守門員

---

## 📘 Git 團隊使用手冊（完整版）

> 適用於三人小組在 VS Code + GitHub 協作專案

---

### 🔧 基本工具與安裝

1. **Git 安裝**（如果還沒裝）：

   * Windows: [https://git-scm.com/downloads](https://git-scm.com/downloads)
   * 裝好後打開 cmd 輸入 `git --version` 確認安裝成功

2. **GitHub 註冊帳號**： [https://github.com](https://github.com)

   * 建議三人都註冊，並使用常用 Email 綁定

3. **VS Code 安裝**（開發環境）： [https://code.visualstudio.com/](https://code.visualstudio.com/)

   * 建議安裝 GitLens 擴充工具（更好看 Git 版本紀錄）

---

### 🚀 初次使用流程（由隊長/主導者操作）

1. 登入 GitHub，點右上 `+` → `New Repository`

   * Repository name: `ai-attention-project`
   * ✅ 勾選 `Add a README`
   * 建立後複製網址，如：`https://github.com/yourname/ai-attention-project.git`

2. 邀請隊友加入專案：

   * 點選 Repo → Settings → Collaborators → 輸入隊友帳號邀請

---

### 💻 每人 clone 專案到電腦（只做一次）

#### 方法一：用 VS Code GUI 操作

1. 開啟 VS Code → 點左側 Source Control（🔃）
2. 點 `Clone Repository` → 貼上 Repo 連結 → 選擇儲存路徑
3. 點「Open Project」打開整個專案

#### 方法二：用指令

```bash
git clone https://github.com/yourname/ai-attention-project.git
cd ai-attention-project
code .  # 用 VS Code 開啟此專案
```

---

### 🔁 每天開工流程（每次都要做）

1. **先拉下最新程式碼**（避免蓋掉別人修改）

```bash
git pull origin main
```

2. **開始編輯你的區塊**（例如 A 改 detector/、B 改 web\_ui/）

3. **寫完後提交**（保存更動 + 上傳給其他人）

```bash
git add .
git commit -m "新增眼睛偵測功能"
git push origin main
```

---

### 🚨 避免衝突建議

| 規則            | 說明                      |
| ------------- | ----------------------- |
| 每人負責不同檔案/資料夾  | 不要同時改一個檔案（尤其是 main.py）  |
| 改前先 pull      | 確保是最新版本再改               |
| 改完就 push      | 讓其他人看得到你的更新             |
| 出現 conflict 時 | VS Code 會標紅，請協調解決誰的程式保留 |

---

### 📁 專案建議架構（範例）

```bash
ai-attention-project/
├── detector/       # A 負責 - Mediapipe 模組
├── logic/          # C 負責 - 睡覺判斷邏輯
├── web_ui/         # B 負責 - Flask / Streamlit 畫面
├── shared/         # 共用設定、參數、資料結構
├── test_video/     # 測試影片素材（不上傳 GitHub）
├── README.md       # 專案說明文件
└── requirements.txt # 所有套件列表
```

> 🔺 如果 test\_video 太大，記得加到 `.gitignore` 避免推上 GitHub

---

### 🧪 補充：Git 常用指令整理

| 功能          | 指令                       |
| ----------- | ------------------------ |
| 檢查 Git 是否正常 | `git --version`          |
| 初始化 Git 倉庫  | `git init`（通常 clone 就不用） |
| 查看目前狀態      | `git status`             |
| 查看修改差異      | `git diff`               |
| 查看提交紀錄      | `git log`                |
| 取消本地更動      | `git checkout -- 檔名`     |

---

### 📌 附註

* 團隊可在 `README.md` 加上更新日誌區塊（例如誰完成什麼）
* 若遇到無法解決的 Git 問題，請找隊長處理或私訊老師詢問
* 團隊使用 Google Drive 管理非程式資料（簡報、影片、草稿等）

---

