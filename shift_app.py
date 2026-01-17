import os
import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta

# =========================
# Mode & page config
# =========================
try:
    mode = st.query_params.get("mode", "staff")
    if isinstance(mode, list):
        mode = mode[0]
except Exception:
    mode = "staff"

layout = "wide" if mode == "admin" else "centered"
st.set_page_config(page_title="Shift App", layout=layout)

# =========================
# Admin password
# =========================
ADMIN_PASSWORD = ""
try:
    if "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    pass

# =========================
# Storage
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "shift_data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "shifts.csv")

# =========================
# Helpers
# =========================
TIMES_15 = [(datetime.min + timedelta(minutes=15*i)).time() for i in range(96)]

def hm(t: time) -> str:
    return t.strftime("%H:%M")

def parse_hm(s):
    if s is None:
        return None
    try:
        if pd.isna(s):
            return None
    except Exception:
        pass

    s = str(s).strip()
    if not s or s.lower() in ("nan", "none"):
        return None

    # "HH:MM:SS" -> "HH:MM"
    if len(s) >= 5 and s[2] == ":":
        s = s[:5]

    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception:
        return None

def dt_of(d, t):
    return datetime.combine(d, t)

def minutes_between(d, start_str, end_str):
    stt = parse_hm(start_str)
    ett = parse_hm(end_str)
    if stt is None or ett is None:
        return 0
    sdt = dt_of(d, stt)
    edt = dt_of(d, ett)
    if edt <= sdt:
        return 0
    return int((edt - sdt).total_seconds() // 60)

# =========================
# CSV IO
# =========================
COLUMNS = ["id", "submitted_at", "name", "date", "start", "end", "note"]

def read_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(CSV_PATH)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].notna()].copy()
    return df

def save_data(df):
    df2 = df.copy()
    df2["date"] = df2["date"].astype(str)
    df2.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# =========================
# UI
# =========================
st.title("🗓 シフト管理")

# =====================================================
# STAFF PAGE
# =====================================================
if mode != "admin":
    st.subheader("✍️ スタッフ：シフト提出")

    name = st.text_input("名前（必須）")
    d = st.date_input("日付", value=date.today())

    start = st.selectbox("開始（15分単位）", TIMES_15, index=36, format_func=lambda x: x.strftime("%H:%M"))
    end   = st.selectbox("終了（15分単位）", TIMES_15, index=72, format_func=lambda x: x.strftime("%H:%M"))

    note = st.text_input("メモ（任意）")

    if st.button("提出", type="primary"):
        if not name.strip():
            st.error("名前を入力してください")
            st.stop()
        if end <= start:
            st.error("終了は開始より後にしてください")
            st.stop()

        df = read_data()

        # ★ 同日・同名は削除して上書き
        df = df[~((df["date"] == d) & (df["name"] == name.strip()))]

        new_row = {
            "id": str(uuid.uuid4()),
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name.strip(),
            "date": d,
            "start": hm(start),
            "end": hm(end),
            "note": note.strip()
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)

        st.success("提出しました！")

    st.info("スタッフ用URL例： `https://<your-app>.streamlit.app/?mode=staff`")
    st.stop()

# =====================================================
# ADMIN PAGE
# =====================================================
st.subheader("🔒 管理者")

if not ADMIN_PASSWORD:
    st.error("ADMIN_PASSWORD が secrets に設定されていません")
    st.stop()

if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

if not st.session_state.admin_ok:
    pw = st.text_input("管理者パスワード", type="password")
    if st.button("ログイン"):
        if pw == ADMIN_PASSWORD:
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    st.stop()

# =========================
# Admin main
# =========================
df = read_data()
if df.empty:
    st.info("提出がありません")
    st.stop()

# 日付選択（提出がある日のみ）
avail_dates = sorted(df["date"].unique())
target_date = st.selectbox("日付を選択", avail_dates, index=len(avail_dates)-1)

day_df = df[df["date"] == target_date].copy()
if day_df.empty:
    st.info("この日の提出はありません")
    st.stop()

# 表示順（開始順）
day_df["start_t"] = day_df["start"].apply(parse_hm)
day_df = day_df.sort_values(["start_t", "name"]).drop(columns=["start_t"])

# 合計時間（分）を計算して列追加
day_df["minutes"] = day_df.apply(lambda r: minutes_between(target_date, r["start"], r["end"]), axis=1)
day_df["total_h"] = (day_df["minutes"] / 60.0).round(2)

# =========================
# Table
# =========================
st.subheader("📋 シフト一覧")
st.dataframe(
    day_df[["name", "start", "end", "note", "total_h"]],
    use_container_width=True
)

# =====================================================
# 人数グラフ（時間帯ごとの人数）
# =====================================================
st.subheader("👥 時間帯ごとの人数")

c1, c2, c3 = st.columns(3)
with c1:
    open_t = st.time_input("集計開始", value=time(7,0))
with c2:
    close_t = st.time_input("集計終了", value=time(22,0))
with c3:
    step_min = st.selectbox("刻み", [15, 30, 60], index=1)

open_dt = dt_of(target_date, open_t)
close_dt = dt_of(target_date, close_t)
if close_dt <= open_dt:
    st.error("集計終了は集計開始より後にしてください")
    st.stop()

# 各人の勤務区間（datetime）
shifts = []
for r in day_df.itertuples():
    stt = parse_hm(r.start)
    ett = parse_hm(r.end)
    if stt is None or ett is None:
        continue
    sdt = dt_of(target_date, stt)
    edt = dt_of(target_date, ett)
    if edt <= sdt:
        continue
    shifts.append((r.name, sdt, edt))

slots = []
t = open_dt
step = timedelta(minutes=step_min)
while t < close_dt:
    slots.append(t)
    t += step

labels = [x.strftime("%H:%M") for x in slots]
counts = []
names_in_slot = []

for s0 in slots:
    s1 = s0 + step
    names = [nm for (nm, a, b) in shifts if (a < s1 and b > s0)]
    counts.append(len(names))
    names_in_slot.append(" / ".join(names))

head_df = pd.DataFrame({"時間": labels, "人数": counts, "名前": names_in_slot})

colL, colR = st.columns([1, 1])
with colL:
    st.dataframe(head_df, use_container_width=True)
with colR:
    figc, axc = plt.subplots()
    axc.plot(labels, counts, marker="o")
    axc.set_xlabel("Time")
    axc.set_ylabel("Headcount")
    axc.set_title(f"Headcount ({step_min}-min slots)")
    axc.grid(True, alpha=0.3)

    keep_every = max(1, (60 // step_min))
    for i, tick in enumerate(axc.get_xticklabels()):
        if i % keep_every != 0:
            tick.set_visible(False)

    st.pyplot(figc)

# =====================================================
# シフト図（ガント）＋右側に合計時間
# =====================================================
st.subheader("📊 シフト図（横：時間 / 縦：名前）＋ 合計時間")

# 日本語フォント：環境によって存在しない可能性があるので安全に
# （存在しない場合も文字化けが起きにくいように fallback ）
plt.rcParams["font.family"] = [
    "IPAexGothic", "IPAPGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo", "DejaVu Sans"
]

# 名前ごとにまとめ（同名が複数行になってもガントは名前一つにしたいならここで統合）
# 今回は「同日・同名は上書き」なので基本1行だが、安全のため groupby で最終行のみ
day_latest = day_df.sort_values("submitted_at").groupby(["date","name"], as_index=False).tail(1)
day_latest = day_latest.sort_values(["start", "name"]).reset_index(drop=True)

# 各人の合計時間（分→時間）
minutes_map = {
    r["name"]: minutes_between(target_date, r["start"], r["end"])
    for _, r in day_latest.iterrows()
}

fig, ax = plt.subplots(figsize=(12, max(3, 0.7 * len(day_latest) + 1)))

base = dt_of(target_date, time(0, 0))

yticks = []
ylabels = []

for i, r in enumerate(day_latest.itertuples(index=False)):
    stt = parse_hm(r.start)
    ett = parse_hm(r.end)
    if stt is None or ett is None:
        continue

    sdt = dt_of(target_date, stt)
    edt = dt_of(target_date, ett)

    left_h = (sdt - base).total_seconds() / 3600.0
    width_h = (edt - sdt).total_seconds() / 3600.0
    if width_h <= 0:
        continue

    ax.barh(i, width_h, left=left_h, height=0.6, alpha=0.9)

    # メモ（空なら何も出さない）
    memo = r.note if isinstance(r.note, str) and r.note.strip() else ""
    if memo:
        ax.text(left_h + width_h/2, i, memo, ha="center", va="center", fontsize=9)

    # 右側に合計時間（時間）
    total_min = minutes_map.get(r.name, 0)
    total_h = total_min / 60.0
    ax.text(24.15, i, f"{total_h:.2f} h", va="center", ha="left", fontsize=10)

    yticks.append(i)
    ylabels.append(r.name)

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xlabel("時間")
ax.set_ylabel("名前")
ax.set_title(f"{target_date} のシフト（右側：合計時間）")

ax.set_xlim(0, 25.5)  # 右側の合計時間表示用に少し余白
ax.set_xticks(range(0, 25, 1))
ax.grid(axis="x", alpha=0.3)

# 右側のラベル
ax.text(24.15, len(ylabels) + 0.2, "合計", fontsize=11, ha="left", va="bottom")

st.pyplot(fig)

st.info("管理者URL例： `https://<your-app>.streamlit.app/?mode=admin`（共有しない）")

