import os
import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, date, time, timedelta

# ============================================================
# Mode + page config（最初に呼ぶ）
# ============================================================
try:
    mode = st.query_params.get("mode", "staff")
    if isinstance(mode, list):
        mode = mode[0]
except Exception:
    mode = "staff"

layout = "wide" if mode == "admin" else "centered"
st.set_page_config(page_title="Shift Planner", layout=layout)

# ============================================================
# Admin password（Secrets or env）
# ============================================================
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
try:
    if not ADMIN_PASSWORD and "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    ADMIN_PASSWORD = ""

# ============================================================
# Fonts（日本語）
# ============================================================
JP_FONT = None
FONT_PATH = os.path.join(os.path.dirname(__file__), "ipaexg.ttf")
if os.path.exists(FONT_PATH):
    try:
        JP_FONT = fm.FontProperties(fname=FONT_PATH)
    except Exception:
        JP_FONT = None

# ============================================================
# Storage
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "shift_data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "shift_requests.csv")

# ============================================================
# Helpers
# ============================================================
def hm(t: time) -> str:
    return t.strftime("%H:%M")

def parse_hm(s):
    # nan / None / 空を安全に None にする
    if s is None:
        return None
    if isinstance(s, float) and pd.isna(s):
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception:
        return None

def dt_of(d: date, t: time) -> datetime:
    return datetime.combine(d, t)

def minutes_from(base: datetime, dt: datetime) -> float:
    return (dt - base).total_seconds() / 60.0

def build_slots(open_dt, close_dt, step_min):
    t = open_dt
    step = timedelta(minutes=step_min)
    slots = []
    while t < close_dt:
        slots.append(t)
        t += step
    return slots

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

# ============================================================
# CSV read/write（古いCSVでも落ちない）
# ============================================================
BASE_COLS = [
    "id","submitted_at",
    "name","date",
    "start","end",
    "note",
    "place",  # S / H / SH
]

def read_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=BASE_COLS)

    df = pd.read_csv(CSV_PATH)

    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = ""

    # 型の整理
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["name"] = df["name"].astype(str)
    df["start"] = df["start"].astype(str)
    df["end"] = df["end"].astype(str)
    df["note"] = df["note"].astype(str)
    df["place"] = df["place"].astype(str)
    return df

def save_data(df: pd.DataFrame):
    df2 = df.copy()
    df2["date"] = df2["date"].astype(str)
    df2.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def upsert_rows(rows: list[dict]):
    """
    同じ (date, name) があれば上書き（最新を残す）
    ※「スタッフが同じ日に複数提出したら最後の1件だけ残る」仕様
    """
    df = read_data()
    if df.empty:
        df = pd.DataFrame(columns=BASE_COLS)

    new_df = pd.DataFrame(rows)

    # 既存から同じ(date,name)を消して、新しいのを追加
    if not df.empty:
        key_existing = set(zip(df["date"].astype(str), df["name"].astype(str)))
        key_new = set(zip(new_df["date"].astype(str), new_df["name"].astype(str)))
        # df側で new のキーと一致する行を落とす
        df = df[~df.apply(lambda r: (str(r["date"]), str(r["name"])) in key_new, axis=1)]

    df = pd.concat([df, new_df], ignore_index=True)
    save_data(df)

# ============================================================
# 15分刻み
# ============================================================
TIMES_15 = [(datetime.min + timedelta(minutes=m)).time() for m in range(0, 24*60, 15)]
def pick_15(label, key, default=time(9, 0)):
    idx = TIMES_15.index(default) if default in TIMES_15 else 0
    return st.selectbox(label, TIMES_15, index=idx, key=key, format_func=lambda x: x.strftime("%H:%M"))

# ============================================================
# Place display helpers
# ============================================================
PLACE_LABEL = {"S": "(S)", "H": "(H)", "SH": "(SH)"}
PLACE_COLOR = {"S": "#2ecc71", "H": "#ff66b3", "SH": "#222222"}  # green / pink / black

def place_tag(place):
    p = (place or "").strip().upper()
    if p not in ["S", "H", "SH"]:
        p = "SH"
    return p

# ============================================================
# UI
# ============================================================
st.title("🗓 シフト管理")

# ============================================================
# STAFF PAGE（管理者ページに行く導線は出さない）
# ============================================================
if mode != "admin":
    st.subheader("✍️ スタッフ：シフト提出")
    st.caption("※このページ（staff URL）だけ共有する想定です。")

    if "shift_rows" not in st.session_state:
        st.session_state.shift_rows = [0]
        st.session_state.next_row_id = 1

    staff_name = st.text_input("名前（必須）", key="staff_name")

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ シフトを追加"):
            st.session_state.shift_rows.append(st.session_state.next_row_id)
            st.session_state.next_row_id += 1
    with colB:
        if st.button("🧹 全部クリア"):
            st.session_state.shift_rows = [0]
            st.session_state.next_row_id = 1

    st.divider()

    rows_to_remove = []
    for rid in list(st.session_state.shift_rows):
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"### シフト {rid+1}")
            with top[1]:
                if st.button("🗑 削除", key=f"del_row_{rid}"):
                    rows_to_remove.append(rid)

            d = st.date_input("日付", value=date.today(), key=f"d_{rid}")
            start_t = pick_15("開始（15分単位）", key=f"start_{rid}", default=time(9,0))
            end_t   = pick_15("終了（15分単位）", key=f"end_{rid}", default=time(18,0))

            # 店舗選択（排他）
            place = st.radio(
                "店舗",
                options=["S", "H", "SH"],
                index=2,
                key=f"place_{rid}",
                format_func=lambda x: {"S":"サブウェイ","H":"ハーゲンダッツ","SH":"どちらでも"}[x],
                horizontal=True
            )

            note_each = st.text_input("メモ（任意）", key=f"note_{rid}", placeholder="例：15時から用事")

    if rows_to_remove:
        st.session_state.shift_rows = [r for r in st.session_state.shift_rows if r not in rows_to_remove]
        if not st.session_state.shift_rows:
            st.session_state.shift_rows = [0]
            st.session_state.next_row_id = 1
        st.rerun()

    st.divider()

    if st.button("✅ まとめて送信", type="primary"):
        if not staff_name.strip():
            st.error("名前を入力してね")
            st.stop()

        errors = []
        rows_to_save = []

        for rid in st.session_state.shift_rows:
            d = st.session_state.get(f"d_{rid}")
            start_t = st.session_state.get(f"start_{rid}")
            end_t   = st.session_state.get(f"end_{rid}")
            place = place_tag(st.session_state.get(f"place_{rid}", "SH"))
            note = safe_str(st.session_state.get(f"note_{rid}", "")).strip()

            sdt = dt_of(d, start_t)
            edt = dt_of(d, end_t)
            if edt <= sdt:
                errors.append(f"シフト {rid+1}: 終了が開始より前/同じ")
                continue

            rows_to_save.append({
                "id": str(uuid.uuid4()),
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": staff_name.strip(),
                "date": d.isoformat(),
                "start": hm(start_t),
                "end": hm(end_t),
                "note": note,          # 空なら空のまま
                "place": place,
            })

        if errors:
            st.error("入力エラーがあります：\n- " + "\n- ".join(errors))
            st.stop()

        # ★ 同じ(date,name)は上書き
        upsert_rows(rows_to_save)
        st.success(f"{len(rows_to_save)} 件 送信しました！（同じ日付＋同じ名前は上書き）")

        st.session_state.shift_rows = [0]
        st.session_state.next_row_id = 1

    st.info("スタッフ用URL例： `https://<your-app>.streamlit.app/?mode=staff`")
    st.stop()

# ============================================================
# ADMIN PAGE（ログイン）
# ============================================================
st.subheader("🔒 管理者：集計")

if not ADMIN_PASSWORD:
    st.error("管理者パスワードが未設定です。Secrets に `ADMIN_PASSWORD = \"...\"` を設定してください。")
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
    st.info("管理者用URL例： `https://<your-app>.streamlit.app/?mode=admin`（共有しない）")
    st.stop()

# ============================================================
# Data load
# ============================================================
df = read_data()
if df.empty:
    st.info("まだ提出がありません。")
    st.stop()

# 日付選択（その日だけ集計＆グラフ）
dates = sorted(df["date"].unique())
target_day = st.selectbox("日付を選択", dates, index=len(dates)-1)

day_df = df[df["date"] == target_day].copy()
day_df["place"] = day_df["place"].apply(place_tag)

# 表示用：名前の横に (S)(H)(SH)
day_df["name_tag"] = day_df.apply(lambda r: f"{r['name']}{PLACE_LABEL.get(r['place'],'(SH)')}", axis=1)

# ============================================================
# Sidebar settings
# ============================================================
with st.sidebar:
    st.subheader("表示範囲")
    open_time = st.time_input("表示開始", value=time(7,0))
    close_time = st.time_input("表示終了", value=time(22,0))
    step_min = st.selectbox("人数集計の刻み", [15, 30, 60], index=1)

open_dt = dt_of(target_day, open_time)
close_dt = dt_of(target_day, close_time)
if close_dt <= open_dt:
    st.error("表示終了は表示開始より後にしてください")
    st.stop()

# ============================================================
# Build shift objects（その日の人一覧）
# ============================================================
people = []
for _, r in day_df.iterrows():
    st_t = parse_hm(r["start"])
    en_t = parse_hm(r["end"])
    if st_t is None or en_t is None:
        continue
    sdt = dt_of(target_day, st_t)
    edt = dt_of(target_day, en_t)
    if edt <= sdt:
        continue

    minutes = (edt - sdt).total_seconds() / 60.0
    people.append({
        "name": r["name"],
        "name_tag": r["name_tag"],
        "place": r["place"],
        "start_dt": sdt,
        "end_dt": edt,
        "minutes": minutes,
        "note": safe_str(r.get("note", "")).strip(),
    })

if not people:
    st.warning("この日付の有効な提出がありません（時間が不正 or データ欠損）")
    st.stop()

# 名前順に（見やすい）
people = sorted(people, key=lambda x: (x["name"], x["start_dt"]))

# ============================================================
# ① 人数グラフ（時間帯ごとの人数）
# ============================================================
st.write("## 👥 時間帯ごとの人数")

slots = build_slots(open_dt, close_dt, step_min)
labels = [t.strftime("%H:%M") for t in slots]
step = timedelta(minutes=step_min)

counts = []
names_each = []

for s0 in slots:
    s1 = s0 + step
    active = [p["name_tag"] for p in people if (p["start_dt"] < s1 and p["end_dt"] > s0)]
    counts.append(len(active))
    names_each.append(" / ".join(active))

head_df = pd.DataFrame({"時間": labels, "人数": counts, "名前": names_each})
st.dataframe(head_df, use_container_width=True)

fig1, ax1 = plt.subplots()
ax1.plot(labels, counts, marker="o")
ax1.set_xlabel("Time")
ax1.set_ylabel("Headcount")
ax1.set_title(f"Headcount by time ({step_min}-min slots)")
ax1.grid(True, alpha=0.3)
keep_every = max(1, (60 // step_min))
for i, tick in enumerate(ax1.get_xticklabels()):
    if i % keep_every != 0:
        tick.set_visible(False)
st.pyplot(fig1)

# ============================================================
# ② シフト図（ガント）＋右側に合計時間（日本語対応）
# ============================================================
st.write("## 📊 シフト図（横：時間 / 縦：名前）＋合計時間")

fig2, ax2 = plt.subplots(figsize=(12, max(3, 0.7 * len(people))))
y_height, y_gap = 8, 4
yticks, ylabels = [], []

for i, p in enumerate(people):
    y = i * (y_height + y_gap)
    yticks.append(y + y_height/2)
    ylabels.append(p["name_tag"])

    x0 = minutes_from(open_dt, p["start_dt"])
    w = minutes_from(open_dt, p["end_dt"]) - x0
    if w <= 0:
        continue

    # 色（S/H/SH）
    color = PLACE_COLOR.get(p["place"], "#222222")
    ax2.broken_barh([(x0, w)], (y, y_height), facecolors=color, edgecolors="none", alpha=0.90)

    # メモ（空なら出さない）
    if p["note"]:
        ax2.text(x0, y + y_height + 1, p["note"], fontsize=9, va="bottom", ha="left")

    # 右側に合計時間
    hours = p["minutes"] / 60.0
    ax2.text(minutes_from(open_dt, close_dt) + 10, y + y_height/2,
             f"{hours:.2f} h", va="center", ha="left", fontsize=10)

total_min = (close_dt - open_dt).total_seconds() / 60
ax2.set_xlim(0, total_min + 80)  # 右側に時間表示スペース

# hour ticks
hour_ticks, hour_labels = [], []
t = open_dt.replace(minute=0, second=0, microsecond=0)
if t < open_dt:
    t += timedelta(hours=1)
while t <= close_dt:
    hour_ticks.append(minutes_from(open_dt, t))
    hour_labels.append(t.strftime("%H"))
    t += timedelta(hours=1)

ax2.set_xticks(hour_ticks)
ax2.set_xticklabels(hour_labels)
ax2.set_xlabel("Hour")
ax2.set_yticks(yticks)

# 日本語フォント適用（名前）
if JP_FONT is not None:
    ax2.set_yticklabels(ylabels, fontproperties=JP_FONT)
else:
    ax2.set_yticklabels(ylabels)  # フォント無い場合は仕方ない

ax2.grid(True, axis="x", alpha=0.25)
ax2.set_title(f"Gantt ({target_day.isoformat()})")

# 凡例（色説明）
ax2.text(0.98, 1.02, "S=サブウェイ(緑) / H=ハーゲンダッツ(ピンク) / SH=どちらでも(黒)",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=10)

st.pyplot(fig2)

# ============================================================
# URL案内
# ============================================================
st.info("スタッフ用URL： `https://<your-app>.streamlit.app/?mode=staff`（共有OK）")
st.warning("管理者用URL： `https://<your-app>.streamlit.app/?mode=admin`（共有しない）")

