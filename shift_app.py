import os
import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, date, time, timedelta

# ============================================================
# Mode + page config (must be first)
# ============================================================
try:
    mode = st.query_params.get("mode", "staff")
    if isinstance(mode, list):
        mode = mode[0]
except Exception:
    mode = "staff"

st.set_page_config(
    page_title="Shift Planner",
    layout="wide" if mode == "admin" else "centered",
)

# ============================================================
# Storage
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "shift_data")
os.makedirs(DATA_DIR, exist_ok=True)
SHIFT_CSV = os.path.join(DATA_DIR, "shifts.csv")
ALLOWED_CSV = os.path.join(DATA_DIR, "allowed_dates.csv")

# ============================================================
# Admin password (Secrets or env)
# ============================================================
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
try:
    if not ADMIN_PASSWORD and "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = str(st.secrets["ADMIN_PASSWORD"])
except Exception:
    ADMIN_PASSWORD = ""

# ============================================================
# Optional Japanese font (ipaexg.ttf in repo root)
# ============================================================
JP_FONT = None
FONT_PATH = os.path.join(os.path.dirname(__file__), "ipaexg.ttf")
if os.path.exists(FONT_PATH):
    try:
        JP_FONT = fm.FontProperties(fname=FONT_PATH)
    except Exception:
        JP_FONT = None

# ============================================================
# Constants (store)
# ============================================================
STORE_OPTIONS = ["サブウェイ", "ハーゲンダッツ", "どちらでも"]
STORE_LABEL = {"サブウェイ": "(S)", "ハーゲンダッツ": "(H)", "どちらでも": "(SH)"}
STORE_COLOR = {"サブウェイ": "#2ecc71", "ハーゲンダッツ": "#ff66b3", "どちらでも": "#222222"}

# ============================================================
# Helpers
# ============================================================
def read_csv_safe(path: str, cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def hm(t: time) -> str:
    return t.strftime("%H:%M")

def parse_hm(s):
    """'HH:MM' 以外は None にする（nan/空/壊れ値は確実に落とす）"""
    if s is None:
        return None
    if isinstance(s, float) and pd.isna(s):
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    # 余計な秒が来たら切る（"09:00:00" -> "09:00"）
    if len(s) >= 5 and s[2] == ":":
        s = s[:5]
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
    out = []
    while t < close_dt:
        out.append(t)
        t += step
    return out

def normalize_store(x) -> str:
    """古いCSV/空/NAN/揺れを救済して必ず3種に寄せる"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "どちらでも"
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return "どちらでも"
    if s in STORE_OPTIONS:
        return s
    low = s.lower()
    if "sub" in low:
        return "サブウェイ"
    if "haag" in low or "hagen" in low:
        return "ハーゲンダッツ"
    return "どちらでも"

def display_name(name: str, store: str) -> str:
    return f"{name}{STORE_LABEL.get(store, '(SH)')}"

def normalize_date_str(x) -> str:
    """
    date が '2026-01-18', '2026/01/18', '2026-01-18 00:00:00' みたいに来ても
    'YYYY-MM-DD' に統一。無理なら ''。
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    # まず pandas に投げる
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.date().isoformat()

# ============================================================
# Load allowed dates (as date objects)
# ============================================================
allowed_df = read_csv_safe(ALLOWED_CSV, ["date"])
allowed_dates = []
for x in allowed_df["date"].tolist():
    ds = normalize_date_str(x)
    if ds:
        allowed_dates.append(datetime.strptime(ds, "%Y-%m-%d").date())
allowed_dates = sorted(set(allowed_dates))

# ============================================================
# Title
# ============================================================
st.title("🗓 シフト管理")

# ============================================================
# STAFF PAGE
# ============================================================
if mode != "admin":
    st.subheader("✍️ スタッフ：シフト提出")
    st.caption("※スタッフにはこのURLだけ共有： `...?mode=staff`")

    if not allowed_dates:
        st.warning("提出可能日（試合日）が未設定です。管理者に連絡してください。")
        st.stop()

    if "rows" not in st.session_state:
        st.session_state.rows = [0]
        st.session_state.next_id = 1

    name = st.text_input("名前（必須）", key="staff_name")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ シフトを追加"):
            st.session_state.rows.append(st.session_state.next_id)
            st.session_state.next_id += 1
    with c2:
        if st.button("🧹 全部クリア"):
            st.session_state.rows = [0]
            st.session_state.next_id = 1

    st.divider()

    remove = []
    for rid in list(st.session_state.rows):
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"### シフト {rid+1}")
            with top[1]:
                if st.button("🗑 削除", key=f"del_{rid}"):
                    remove.append(rid)

            d = st.selectbox(
                "日付（提出可能日のみ）",
                allowed_dates,
                format_func=lambda x: x.strftime("%Y-%m-%d"),
                key=f"d_{rid}"
            )
            s = st.time_input("開始", value=time(9, 0), key=f"s_{rid}")
            e = st.time_input("終了", value=time(18, 0), key=f"e_{rid}")

            store = st.radio(
                "店舗",
                STORE_OPTIONS,
                horizontal=True,
                key=f"store_{rid}"
            )

            note = st.text_input("メモ（任意）", key=f"note_{rid}")

    if remove:
        st.session_state.rows = [r for r in st.session_state.rows if r not in remove]
        if not st.session_state.rows:
            st.session_state.rows = [0]
            st.session_state.next_id = 1
        st.rerun()

    st.divider()

    if st.button("✅ まとめて送信", type="primary"):
        if not name.strip():
            st.error("名前を入力してね")
            st.stop()

        df = read_csv_safe(SHIFT_CSV, ["id","submitted_at","date","name","start","end","store","note"])

        # 行ごとに (date,name) を上書き
        for rid in st.session_state.rows:
            d = st.session_state.get(f"d_{rid}")
            s = st.session_state.get(f"s_{rid}")
            e = st.session_state.get(f"e_{rid}")
            store = normalize_store(st.session_state.get(f"store_{rid}", "どちらでも"))
            note = (st.session_state.get(f"note_{rid}", "") or "").strip()

            if d is None or s is None or e is None:
                st.error("日付/時間が未入力の行があります")
                st.stop()
            if e <= s:
                st.error("終了が開始より前/同じの行があります")
                st.stop()

            date_str = d.isoformat()

            # 既存を削除して上書き
            df["date_norm"] = df["date"].apply(normalize_date_str)
            df["name_norm"] = df["name"].astype(str).str.strip()
            df = df[~((df["date_norm"] == date_str) & (df["name_norm"] == name.strip()))]
            df = df.drop(columns=["date_norm","name_norm"], errors="ignore")

            df.loc[len(df)] = [
                str(uuid.uuid4()),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                date_str,
                name.strip(),
                hm(s),
                hm(e),
                store,
                note
            ]

        save_csv(df, SHIFT_CSV)
        st.success("提出しました！（同じ日付＋同じ名前は上書き）")

        st.session_state.rows = [0]
        st.session_state.next_id = 1

    st.info("スタッフ用URL： `https://<あなたのアプリ>.streamlit.app/?mode=staff`")
    st.stop()

# ============================================================
# ADMIN PAGE (login)
# ============================================================
st.subheader("🔒 管理者：試合日設定・集計")

if not ADMIN_PASSWORD:
    st.error("管理者パスワードが未設定です。Secrets に ADMIN_PASSWORD を設定してください。")
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
    st.info("管理者用URL： `https://<あなたのアプリ>.streamlit.app/?mode=admin`（共有しない）")
    st.stop()

# ============================================================
# Admin: Allowed dates editor
# ============================================================
st.write("## 📅 提出可能日（試合日）")

colA, colB = st.columns([1, 2])
with colA:
    new_day = st.date_input("追加する日付", value=date.today(), key="new_allowed_day")
    if st.button("➕ 追加"):
        allowed_dates.append(new_day)
        allowed_dates = sorted(set(allowed_dates))
        save_csv(pd.DataFrame({"date": [d.isoformat() for d in allowed_dates]}), ALLOWED_CSV)
        st.success("追加しました")
        st.rerun()

with colB:
    if allowed_dates:
        st.write("### 登録済み（押すと削除）")
        for d in allowed_dates:
            if st.button(f"❌ {d.isoformat()}", key=f"rm_{d.isoformat()}"):
                allowed_dates = [x for x in allowed_dates if x != d]
                save_csv(pd.DataFrame({"date": [x.isoformat() for x in allowed_dates]}), ALLOWED_CSV)
                st.success("削除しました")
                st.rerun()
    else:
        st.info("まだ登録がありません。試合日を追加してください。")

st.divider()

# ============================================================
# Admin: Load shifts (robust)
# ============================================================
st.write("## 📥 提出データ（読み込み状況）")

shift_df = read_csv_safe(SHIFT_CSV, ["id","submitted_at","date","name","start","end","store","note"])
st.write("SHIFT_CSV:", SHIFT_CSV)
st.write("exists:", os.path.exists(SHIFT_CSV))
st.write("rows:", len(shift_df))
if shift_df.empty:
    st.info("まだ提出がありません。")
    st.stop()

# 正規化
shift_df["date_norm"] = shift_df["date"].apply(normalize_date_str)
shift_df["name_norm"] = shift_df["name"].astype(str).str.strip()
shift_df["start_norm"] = shift_df["start"].apply(lambda x: hm(parse_hm(x)) if parse_hm(x) else "")
shift_df["end_norm"] = shift_df["end"].apply(lambda x: hm(parse_hm(x)) if parse_hm(x) else "")
shift_df["store_norm"] = shift_df["store"].apply(normalize_store)
shift_df["note_norm"] = shift_df["note"].apply(lambda x: "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x).strip())
shift_df["submitted_at_dt"] = pd.to_datetime(shift_df["submitted_at"], errors="coerce")

# 有効行だけ抽出
valid = shift_df[
    (shift_df["date_norm"] != "") &
    (shift_df["name_norm"] != "") &
    (shift_df["start_norm"] != "") &
    (shift_df["end_norm"] != "")
].copy()

st.caption(f"全行: {len(shift_df)} / 有効行(集計対象): {len(valid)}")
if len(valid) == 0:
    st.error("有効な提出が0件です。date/start/end が壊れている可能性があります。")
    st.dataframe(shift_df[["date","name","start","end","store","note"]].head(50), use_container_width=True)
    st.stop()

# 日付候補
dates_have = sorted(valid["date_norm"].unique())
target_date_str = st.selectbox("集計する日付", dates_have, index=len(dates_have)-1)
target_day = datetime.strptime(target_date_str, "%Y-%m-%d").date()

day_df = valid[valid["date_norm"] == target_date_str].copy()
if day_df.empty:
    st.info("この日付の提出はありません。")
    st.stop()

# 同日・同名は最新（submitted_atがNaTでも最後の行を採用）
day_df = day_df.sort_values(["submitted_at_dt"], na_position="first")
day_df = day_df.drop_duplicates(subset=["date_norm","name_norm"], keep="last")

# 表示範囲
with st.sidebar:
    st.subheader("表示範囲")
    open_time = st.time_input("表示開始", value=time(7, 0))
    close_time = st.time_input("表示終了", value=time(22, 0))
    step_min = st.selectbox("人数集計の刻み", [15, 30, 60], index=1)

open_dt = dt_of(target_day, open_time)
close_dt = dt_of(target_day, close_time)
if close_dt <= open_dt:
    st.error("表示終了は表示開始より後にしてください")
    st.stop()

# people 作成
people = []
dropped = 0
for _, r in day_df.iterrows():
    st_t = parse_hm(r["start_norm"])
    en_t = parse_hm(r["end_norm"])
    if st_t is None or en_t is None:
        dropped += 1
        continue
    sdt = dt_of(target_day, st_t)
    edt = dt_of(target_day, en_t)
    if edt <= sdt:
        dropped += 1
        continue

    minutes = (edt - sdt).total_seconds() / 60.0
    store = r["store_norm"]
    name = r["name_norm"]
    note = r["note_norm"]

    people.append({
        "name": name,
        "store": store,
        "name_tag": display_name(name, store),
        "start_dt": sdt,
        "end_dt": edt,
        "minutes": minutes,
        "note": note,
    })

people = sorted(people, key=lambda x: (x["start_dt"], x["name"]))

st.caption(f"この日の人数: {len(people)}（不正で除外: {dropped}）")
st.dataframe(day_df[["name_norm","date_norm","start_norm","end_norm","store_norm","note_norm","submitted_at"]], use_container_width=True)

st.divider()

# ============================================================
# Headcount (time slot) + table + graph
# ============================================================
st.write("## 👥 時間帯ごとの人数")

slots = build_slots(open_dt, close_dt, step_min)
labels = [t.strftime("%H:%M") for t in slots]
step = timedelta(minutes=step_min)

counts, names_each = [], []
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
ax1.set_title(f"Headcount ({step_min}-min)")
ax1.grid(True, alpha=0.3)

keep_every = max(1, (60 // step_min))
for i, tick in enumerate(ax1.get_xticklabels()):
    if i % keep_every != 0:
        tick.set_visible(False)

st.pyplot(fig1)

st.divider()

# ============================================================
# Gantt + total hours (right)
# ============================================================
st.write("## 📊 シフト図（ガント）＋合計時間（右）")

if not people:
    st.info("表示できるシフトがありません（start/end が不正の可能性）。")
    st.stop()

fig2, ax2 = plt.subplots(figsize=(12, max(3, 0.75 * len(people))))
y_height, y_gap = 8, 4
yticks, ylabels = [], []

total_min = (close_dt - open_dt).total_seconds() / 60.0
ax2.set_xlim(0, total_min + 120)

for i, p in enumerate(people):
    y = i * (y_height + y_gap)
    yticks.append(y + y_height / 2)
    ylabels.append(p["name_tag"])

    x0 = minutes_from(open_dt, p["start_dt"])
    w = minutes_from(open_dt, p["end_dt"]) - x0
    if w <= 0:
        continue

    color = STORE_COLOR.get(p["store"], "#222222")
    ax2.broken_barh([(x0, w)], (y, y_height), facecolors=color, edgecolors="none", alpha=0.90)

    if p["note"]:
        ax2.text(x0, y + y_height + 1, p["note"], fontsize=9, va="bottom", ha="left")

    ax2.text(total_min + 10, y + y_height / 2, f"{p['minutes']/60:.2f} h",
             va="center", ha="left", fontsize=10)

# x ticks by hour
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
if JP_FONT is not None:
    ax2.set_yticklabels(ylabels, fontproperties=JP_FONT)
else:
    ax2.set_yticklabels(ylabels)

ax2.grid(True, axis="x", alpha=0.25)
ax2.set_title(f"Gantt ({target_day.isoformat()})")

st.pyplot(fig2)

st.info("スタッフ用URL： `https://shift-app-nkyl4zuhzrjejz8zxxlh3a.streamlit.app/?mode=staff`（共有OK）")
st.warning("管理者用URL： `https://shift-app-nkyl4zuhzrjejz8zxxlh3a.streamlit.app/?mode=admin`（共有しない）")

