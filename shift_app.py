import os
import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, date, time, timedelta

# =====================================================
# 初期設定
# =====================================================
try:
    mode = st.query_params.get("mode", "staff")
except Exception:
    mode = "staff"

st.set_page_config(
    page_title="Shift Planner",
    layout="wide" if mode == "admin" else "centered"
)

DATA_DIR = "shift_data"
os.makedirs(DATA_DIR, exist_ok=True)

SHIFT_CSV = f"{DATA_DIR}/shifts.csv"
ALLOWED_CSV = f"{DATA_DIR}/allowed_dates.csv"

# 日本語フォント
jp_font = fm.FontProperties(fname="ipaexg.ttf")

# 管理者PW
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

# =====================================================
# 共通関数
# =====================================================
def read_csv(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def hm(t): return t.strftime("%H:%M")

def parse_hm(s):
    if pd.isna(s) or not str(s).strip():
        return None
    return datetime.strptime(str(s), "%H:%M").time()

# =====================================================
# 管理者：提出可能日設定
# =====================================================
allowed_df = read_csv(ALLOWED_CSV, ["date"])
allowed_dates = sorted(
    [datetime.strptime(d, "%Y-%m-%d").date() for d in allowed_df["date"] if str(d)]
)

# =====================================================
# スタッフ画面
# =====================================================
if mode != "admin":
    st.title("✍️ シフト提出")

    if not allowed_dates:
        st.warning("現在、提出可能日が設定されていません")
        st.stop()

    name = st.text_input("名前（必須）")
    rows = st.session_state.setdefault("rows", [0])

    if st.button("➕ シフト追加"):
        rows.append(len(rows))

    shifts = []

    for i in rows:
        st.divider()
        d = st.selectbox(
            f"日付（{i+1}）",
            allowed_dates,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            key=f"d{i}"
        )
        c1, c2 = st.columns(2)
        with c1:
            s = st.time_input("開始", value=time(9,0), key=f"s{i}")
        with c2:
            e = st.time_input("終了", value=time(18,0), key=f"e{i}")

        store = st.radio(
            "店舗",
            ["サブウェイ", "ハーゲンダッツ", "どちらでも"],
            horizontal=True,
            key=f"store{i}"
        )
        note = st.text_input("メモ（任意）", key=f"note{i}")

        shifts.append((d, s, e, store, note))

    if st.button("送信"):
        if not name:
            st.error("名前を入力してください")
            st.stop()

        df = read_csv(SHIFT_CSV, ["date","name","start","end","store","note"])

        for d, s, e, store, note in shifts:
            if e <= s:
                st.error("終了時刻が不正です")
                st.stop()

            # 上書き
            df = df[~((df["date"] == str(d)) & (df["name"] == name))]

            df.loc[len(df)] = [
                str(d), name, hm(s), hm(e), store, note
            ]

        save_csv(df, SHIFT_CSV)
        st.success("提出完了！")
        st.session_state.rows = [0]

    st.stop()

# =====================================================
# 管理者画面
# =====================================================
st.title("🔒 管理者画面")

pw = st.text_input("管理者パスワード", type="password")
if pw != ADMIN_PASSWORD:
    st.stop()

# 提出可能日管理
st.subheader("📅 提出可能日（試合日）")
new_day = st.date_input("追加する日付")
if st.button("追加"):
    allowed_dates.append(new_day)
    allowed_dates = sorted(set(allowed_dates))
    save_csv(pd.DataFrame({"date":[d.isoformat() for d in allowed_dates]}), ALLOWED_CSV)
    st.rerun()

for d in allowed_dates:
    if st.button(f"❌ {d}", key=str(d)):
        allowed_dates.remove(d)
        save_csv(pd.DataFrame({"date":[x.isoformat() for x in allowed_dates]}), ALLOWED_CSV)
        st.rerun()

st.divider()

# シフト集計
df = read_csv(SHIFT_CSV, ["date","name","start","end","store","note"])
if df.empty:
    st.info("提出なし")
    st.stop()

df["date"] = pd.to_datetime(df["date"]).dt.date
target = st.selectbox("表示する日付", sorted(df["date"].unique()))
day_df = df[df["date"] == target]

# 色
COLORS = {
    "サブウェイ": "green",
    "ハーゲンダッツ": "pink",
    "どちらでも": "black"
}
LABEL = {
    "サブウェイ": "S",
    "ハーゲンダッツ": "H",
    "どちらでも": "SH"
}

# ガント図
fig, ax = plt.subplots(figsize=(12, 0.7*len(day_df)))
yticks, labels = [], []

for i, r in enumerate(day_df.itertuples()):
    s = datetime.combine(target, parse_hm(r.start))
    e = datetime.combine(target, parse_hm(r.end))
    ax.barh(
        i,
        (e-s).seconds/3600,
        left=s.hour + s.minute/60,
        color=COLORS[r.store]
    )
    yticks.append(i)
    labels.append(f"{r.name} ({LABEL[r.store]})")

ax.set_yticks(yticks)
ax.set_yticklabels(labels, fontproperties=jp_font)
ax.set_xlabel("時間")
ax.set_title(f"{target} のシフト")
st.pyplot(fig)

# 人数グラフ
st.subheader("👥 時間帯ごとの人数")
times = range(7, 23)
counts = []

for h in times:
    c = 0
    for r in day_df.itertuples():
        if parse_hm(r.start).hour <= h < parse_hm(r.end).hour:
            c += 1
    counts.append(c)

fig2, ax2 = plt.subplots()
ax2.plot(times, counts, marker="o")
ax2.set_xlabel("時間")
ax2.set_ylabel("人数")
st.pyplot(fig2)
