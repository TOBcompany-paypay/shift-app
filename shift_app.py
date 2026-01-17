import os
import uuid
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta

# ===== 管理者認証 =====
query = st.experimental_get_query_params()
mode = query.get("mode", ["staff"])[0]

def require_admin():
    if "admin_authed" not in st.session_state:
        st.session_state.admin_authed = False

    if not st.session_state.admin_authed:
        pw = st.text_input("管理者パスワード", type="password")
        if pw == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.admin_authed = True
            st.experimental_rerun()
        else:
            st.stop()

if mode == "admin":
    require_admin()

# =============================
# Storage
# =============================
DATA_DIR = os.path.join(os.path.dirname(__file__), "shift_data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "shift_requests.csv")

# =============================
# Config (admin password)
# =============================
# どれかで設定できる：
# 1) 環境変数 ADMIN_PASSWORD
# 2) Streamlit secrets の ADMIN_PASSWORD
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
try:
    if not ADMIN_PASSWORD and hasattr(st, "secrets") and "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    pass

# =============================
# Helpers
# =============================
def hm(t: time) -> str:
    return t.strftime("%H:%M")

def parse_hm(s: str):
    s = (s or "").strip()
    if not s:
        return None
    return datetime.strptime(s, "%H:%M").time()

def dt_of(d: date, t: time) -> datetime:
    return datetime.combine(d, t)

def clamp_break(start_dt, end_dt, b_start, b_end):
    if b_start is None or b_end is None:
        return None, None
    if b_end <= b_start:
        return None, None
    if b_end <= start_dt or b_start >= end_dt:
        return None, None
    bs = max(start_dt, b_start)
    be = min(end_dt, b_end)
    if be <= bs:
        return None, None
    return bs, be

def segments_minus_breaks(start_dt, end_dt, breaks):
    segs = [(start_dt, end_dt)]
    for (bs, be) in sorted(breaks, key=lambda x: x[0]):
        new = []
        for (a, b) in segs:
            if be <= a or bs >= b:
                new.append((a, b))
                continue
            if a < bs:
                new.append((a, bs))
            if be < b:
                new.append((be, b))
        segs = [(a, b) for (a, b) in new if b > a]
    return segs

def working_at_time(segs, qdt):
    return any(a <= qdt < b for (a, b) in segs)

def working_in_slot(segs, s0, s1):
    return any((a < s1) and (b > s0) for (a, b) in segs)

def minutes_from(base, dt):
    return (dt - base).total_seconds() / 60.0

def build_slots(open_dt, close_dt, step_min):
    t = open_dt
    step = timedelta(minutes=step_min)
    slots = []
    while t < close_dt:
        slots.append(t)
        t += step
    return slots

# 15分刻みの選択肢
TIMES_15 = [(datetime.min + timedelta(minutes=m)).time() for m in range(0, 24*60, 15)]
def pick_15(label, key, default=time(9, 0)):
    idx = TIMES_15.index(default) if default in TIMES_15 else 0
    return st.selectbox(label, TIMES_15, index=idx, key=key, format_func=lambda x: x.strftime("%H:%M"))

def read_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=[
            "id","submitted_at",
            "name","date",
            "orig_start","orig_end","orig_note",
            "admin_start","admin_end",
            "admin_break1_start","admin_break1_end",
            "admin_break2_start","admin_break2_end",
            "admin_note",
            "admin_deleted","admin_updated_at"
        ])
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # bool整形
    if "admin_deleted" in df.columns:
        df["admin_deleted"] = df["admin_deleted"].fillna(False).astype(bool)
    else:
        df["admin_deleted"] = False
    return df

def save_data(df: pd.DataFrame):
    df2 = df.copy()
    # date をISO文字列に
    df2["date"] = df2["date"].astype(str)
    df2.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def append_rows(rows: list[dict]):
    df = read_data()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save_data(df)

def effective_time(row, col_admin, col_orig):
    v = (row.get(col_admin, "") or "").strip()
    if v:
        return v
    return (row.get(col_orig, "") or "").strip()

def is_overridden(row):
    # admin_start/admin_end どっちか入ってたら「変更あり」とみなす
    return bool((row.get("admin_start","") or "").strip() or (row.get("admin_end","") or "").strip()
                or (row.get("admin_break1_start","") or "").strip() or (row.get("admin_break2_start","") or "").strip()
                or (row.get("admin_note","") or "").strip())

# =============================
# App routing by URL
# =============================
# Streamlit 1.53: st.query_params が使えます
try:
    mode = st.query_params.get("mode", "staff")
    if isinstance(mode, list):
        mode = mode[0]
except Exception:
    mode = "staff"

# スマホ表示崩れ対策：スタッフは centered
layout = "centered" if mode != "admin" else "wide"
st.set_page_config(page_title="Shift Planner", layout=layout)

st.title("🗓 シフト管理")

# =============================
# STAFF PAGE (mode=staff)
# =============================
if mode != "admin":
    st.subheader("✍️ スタッフ：シフト提出")
    st.caption("※管理者ページは表示されません（スタッフ用URL）。")

    if "shift_rows" not in st.session_state:
        st.session_state.shift_rows = [0]
        st.session_state.next_row_id = 1

    staff_name = st.text_input("名前（必須）", key="staff_name")
    common_note = st.text_input("メモ（任意・共通）", key="staff_common_note", placeholder="例：授業のため17時まで")

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

    # 各行：削除ボタン付き
    rows_to_remove = []
    for rid in list(st.session_state.shift_rows):
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"### シフト {rid+1}")
            with top[1]:
                if st.button("🗑 この行を削除", key=f"del_row_{rid}"):
                    rows_to_remove.append(rid)

            d = st.date_input("日付", value=date.today(), key=f"d_{rid}")
            start_t = pick_15("開始（15分単位）", key=f"start_{rid}", default=time(9,0))
            end_t   = pick_15("終了（15分単位）", key=f"end_{rid}", default=time(18,0))
            note_each = st.text_input("この行のメモ（任意）", key=f"note_{rid}", placeholder="例：15時から用事")

    # 削除反映
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
            sdt = dt_of(d, start_t)
            edt = dt_of(d, end_t)
            if edt <= sdt:
                errors.append(f"シフト {rid+1}: 終了が開始より前/同じ")
                continue

            note_each = (st.session_state.get(f"note_{rid}", "") or "").strip()
            note_all = (common_note or "").strip()
            merged_note = " / ".join([x for x in [note_all, note_each] if x])

            rows_to_save.append({
                "id": str(uuid.uuid4()),
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": staff_name.strip(),
                "date": d.isoformat(),
                "orig_start": hm(start_t),
                "orig_end": hm(end_t),
                "orig_note": merged_note,

                # 管理者変更欄は空で保存
                "admin_start": "",
                "admin_end": "",
                "admin_break1_start": "",
                "admin_break1_end": "",
                "admin_break2_start": "",
                "admin_break2_end": "",
                "admin_note": "",
                "admin_deleted": False,
                "admin_updated_at": "",
            })

        if errors:
            st.error("入力エラーがあります：\n- " + "\n- ".join(errors))
            st.stop()

        append_rows(rows_to_save)
        st.success(f"{len(rows_to_save)} 件 提出しました！")

        # reset
        st.session_state.shift_rows = [0]
        st.session_state.next_row_id = 1

    # スタッフに案内（URL分離）
    st.info("スタッフ用URL例： `http://<host>:8501/?mode=staff`  （このURLだけ共有）")

    st.stop()

# =============================
# ADMIN PAGE (mode=admin) with password
# =============================
st.subheader("🔒 管理者：集計・編集")

if not ADMIN_PASSWORD:
    st.warning("管理者パスワードが未設定です。環境変数 ADMIN_PASSWORD か secrets に設定してください。")
else:
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
        st.info("管理者用URL例： `http://<host>:8501/?mode=admin`（このURLは共有しない）")
        st.stop()

df = read_data()
if df.empty:
    st.info("まだ提出がありません。")
    st.stop()

# 日付選択
dates = sorted(df["date"].unique())
target_day = st.selectbox("日付を選択", dates, index=len(dates)-1)

day_df = df[df["date"] == target_day].copy()

with st.sidebar:
    st.subheader("表示範囲")
    open_time = st.time_input("営業開始（表示）", value=time(7,0), key="admin_open")
    close_time = st.time_input("営業終了（表示）", value=time(22,0), key="admin_close")
    step_min = st.selectbox("人数集計の刻み", [15, 30, 60], index=1, key="admin_step")

open_dt = dt_of(target_day, open_time)
close_dt = dt_of(target_day, close_time)
if close_dt <= open_dt:
    st.error("表示の営業終了は営業開始より後にしてください")
    st.stop()

# =============================
# Admin edit controls (select one row to edit)
# =============================
st.write("## 🛠 シフトの変更・削除（管理者）")
st.caption("元の提出（orig_*）は残し、変更後（admin_*）を別欄に保存します。集計は admin_* が入っていればそれを優先します。")

# 表示用ラベル
def label_row(r):
    o = f"{r['orig_start']}-{r['orig_end']}"
    a = ""
    if is_overridden(r) or r.get("admin_deleted", False):
        ae = f"{(r.get('admin_start','') or '').strip() or r['orig_start']}-{(r.get('admin_end','') or '').strip() or r['orig_end']}"
        a = f" → {ae}"
    delmark = " [削除]" if bool(r.get("admin_deleted", False)) else ""
    return f"{r['name']} / {r['date']} / {o}{a}{delmark}"

# 選択
day_df = day_df.sort_values(["name","orig_start"])
options = day_df["id"].tolist()
labels = {rid: label_row(day_df[day_df["id"]==rid].iloc[0].to_dict()) for rid in options}

selected_id = st.selectbox("編集する提出を選択", options, format_func=lambda rid: labels.get(rid, rid))

row = df[df["id"] == selected_id].iloc[0].to_dict()

# 元（スタッフ入力）
st.write("### 元の提出（スタッフ入力）")
st.write(f"- 名前：**{row['name']}**")
st.write(f"- 日付：**{row['date']}**")
st.write(f"- 時間：**{row['orig_start']}–{row['orig_end']}**")
if (row.get("orig_note","") or "").strip():
    st.write(f"- メモ：{row['orig_note']}")

st.write("### 変更後（管理者が反映する内容）")

# 現在の admin 値（なければ orig を初期値に）
cur_start = parse_hm((row.get("admin_start","") or "").strip() or row["orig_start"])
cur_end   = parse_hm((row.get("admin_end","") or "").strip() or row["orig_end"])

# 休憩（最大2回・15分単位・自由）
cur_b1s = parse_hm((row.get("admin_break1_start","") or "").strip())
cur_b1e = parse_hm((row.get("admin_break1_end","") or "").strip())
cur_b2s = parse_hm((row.get("admin_break2_start","") or "").strip())
cur_b2e = parse_hm((row.get("admin_break2_end","") or "").strip())

c1, c2 = st.columns(2)
with c1:
    new_start = pick_15("開始（反映）", key="admin_new_start", default=cur_start or time(9,0))
with c2:
    new_end   = pick_15("終了（反映）", key="admin_new_end", default=cur_end or time(18,0))

st.write("#### 休憩（反映：最大2回）")
bcol1, bcol2 = st.columns(2)
with bcol1:
    use_b1 = st.checkbox("休憩1を使う", value=bool(cur_b1s and cur_b1e), key="admin_use_b1")
with bcol2:
    use_b2 = st.checkbox("休憩2を使う", value=bool(cur_b2s and cur_b2e), key="admin_use_b2")

if use_b1:
    bb1, bb2 = st.columns(2)
    with bb1:
        nb1s = pick_15("休憩1 開始", key="admin_b1s", default=cur_b1s or time(12,0))
    with bb2:
        nb1e = pick_15("休憩1 終了", key="admin_b1e", default=cur_b1e or time(13,0))
else:
    nb1s = nb1e = None

if use_b2:
    bb3, bb4 = st.columns(2)
    with bb3:
        nb2s = pick_15("休憩2 開始", key="admin_b2s", default=cur_b2s or time(15,0))
    with bb4:
        nb2e = pick_15("休憩2 終了", key="admin_b2e", default=cur_b2e or time(15,15))
else:
    nb2s = nb2e = None

admin_note = st.text_input("管理者メモ（任意）", value=(row.get("admin_note","") or ""), key="admin_note")

btn1, btn2, btn3, btn4 = st.columns([1,1,1,1])
with btn1:
    save_btn = st.button("💾 変更を保存", type="primary")
with btn2:
    clear_btn = st.button("↩ 変更をクリア（元に戻す）")
with btn3:
    del_btn = st.button("🗑 この提出を削除（非表示）")
with btn4:
    undel_btn = st.button("♻ 削除を取り消し")

def update_row_in_df(df, rid, updates: dict):
    idx = df.index[df["id"] == rid]
    if len(idx) == 0:
        return df
    i = idx[0]
    for k, v in updates.items():
        df.at[i, k] = v
    return df

if save_btn:
    sdt = dt_of(target_day, new_start)
    edt = dt_of(target_day, new_end)
    if edt <= sdt:
        st.error("終了（反映）が開始より前/同じです")
        st.stop()

    # break normalize + overlap
    breaks = []
    if use_b1:
        bs, be = clamp_break(sdt, edt, dt_of(target_day, nb1s), dt_of(target_day, nb1e))
        if not bs or not be:
            st.error("休憩1が不正（勤務外 or 終了<=開始）です")
            st.stop()
        breaks.append((bs, be))
    if use_b2:
        bs, be = clamp_break(sdt, edt, dt_of(target_day, nb2s), dt_of(target_day, nb2e))
        if not bs or not be:
            st.error("休憩2が不正（勤務外 or 終了<=開始）です")
            st.stop()
        breaks.append((bs, be))

    if len(breaks) == 2:
        (a1, a2), (b1, b2) = sorted(breaks, key=lambda x: x[0])
        if not (a2 <= b1):
            st.error("休憩1と休憩2が重なっています。ずらしてください。")
            st.stop()

    df = update_row_in_df(df, selected_id, {
        "admin_start": hm(new_start),
        "admin_end": hm(new_end),
        "admin_break1_start": hm(nb1s) if use_b1 else "",
        "admin_break1_end": hm(nb1e) if use_b1 else "",
        "admin_break2_start": hm(nb2s) if use_b2 else "",
        "admin_break2_end": hm(nb2e) if use_b2 else "",
        "admin_note": (admin_note or "").strip(),
        "admin_deleted": False,
        "admin_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_data(df)
    st.success("保存しました")
    st.rerun()

if clear_btn:
    df = update_row_in_df(df, selected_id, {
        "admin_start": "",
        "admin_end": "",
        "admin_break1_start": "",
        "admin_break1_end": "",
        "admin_break2_start": "",
        "admin_break2_end": "",
        "admin_note": "",
        "admin_deleted": False,
        "admin_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_data(df)
    st.success("元に戻しました")
    st.rerun()

if del_btn:
    df = update_row_in_df(df, selected_id, {
        "admin_deleted": True,
        "admin_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_data(df)
    st.success("削除（非表示）にしました")
    st.rerun()

if undel_btn:
    df = update_row_in_df(df, selected_id, {
        "admin_deleted": False,
        "admin_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_data(df)
    st.success("削除を取り消しました")
    st.rerun()

st.divider()

# =============================
# Output: Original vs Modified tables
# =============================
st.write("## 📋 出力：元の提出 と 変更後（反映）")

day_df = df[df["date"] == target_day].copy()

# 元の提出
orig_out = day_df[["id","name","date","orig_start","orig_end","orig_note","submitted_at"]].copy()
orig_out = orig_out.sort_values(["name","orig_start"])
st.write("### 元の提出（スタッフ入力）")
st.dataframe(orig_out, use_container_width=True)

# 変更後（反映）: effective 値 + 休憩 + 削除状態
def build_effective(df_day: pd.DataFrame):
    rows = []
    for _, r in df_day.iterrows():
        eff_start = effective_time(r, "admin_start", "orig_start")
        eff_end   = effective_time(r, "admin_end", "orig_end")
        b1 = ""
        if (r.get("admin_break1_start","") or "").strip() and (r.get("admin_break1_end","") or "").strip():
            b1 = f"{r['admin_break1_start']}-{r['admin_break1_end']}"
        b2 = ""
        if (r.get("admin_break2_start","") or "").strip() and (r.get("admin_break2_end","") or "").strip():
            b2 = f"{r['admin_break2_start']}-{r['admin_break2_end']}"
        btxt = " / ".join([x for x in [b1, b2] if x])

        rows.append({
            "id": r["id"],
            "name": r["name"],
            "date": r["date"],
            "effective_start": eff_start,
            "effective_end": eff_end,
            "breaks": btxt,
            "admin_note": (r.get("admin_note","") or "").strip(),
            "deleted": bool(r.get("admin_deleted", False)),
            "admin_updated_at": (r.get("admin_updated_at","") or "").strip(),
        })
    out = pd.DataFrame(rows).sort_values(["name","effective_start"])
    return out

eff_out = build_effective(day_df)
st.write("### 変更後（反映用：管理者が決めた時間・休憩）")
st.dataframe(eff_out, use_container_width=True)

st.divider()

# =============================
# Aggregation using effective schedule (skip deleted)
# =============================
st.write("## 📊 集計（反映データで計算）")

# build staff objects
staff = []
for _, r in day_df.iterrows():
    if bool(r.get("admin_deleted", False)):
        continue
    eff_start = effective_time(r, "admin_start", "orig_start")
    eff_end   = effective_time(r, "admin_end", "orig_end")
    sdt = dt_of(target_day, parse_hm(eff_start))
    edt = dt_of(target_day, parse_hm(eff_end))

    breaks = []
    b1s = parse_hm((r.get("admin_break1_start","") or "").strip())
    b1e = parse_hm((r.get("admin_break1_end","") or "").strip())
    if b1s and b1e:
        bs, be = clamp_break(sdt, edt, dt_of(target_day, b1s), dt_of(target_day, b1e))
        if bs and be:
            breaks.append((bs, be))

    b2s = parse_hm((r.get("admin_break2_start","") or "").strip())
    b2e = parse_hm((r.get("admin_break2_end","") or "").strip())
    if b2s and b2e:
        bs, be = clamp_break(sdt, edt, dt_of(target_day, b2s), dt_of(target_day, b2e))
        if bs and be:
            breaks.append((bs, be))

    segs = segments_minus_breaks(sdt, edt, breaks)

    staff.append({
        "name": r["name"],
        "start_dt": sdt,
        "end_dt": edt,
        "breaks": breaks,
        "segs": segs,
    })

# 指定時刻：人数＋名前
c1, c2 = st.columns([1, 2])
with c1:
    q_time = st.time_input("この時刻に働いている人", value=open_time, key="agg_qtime")
with c2:
    qdt = dt_of(target_day, q_time)
    active = [s["name"] for s in staff if working_at_time(s["segs"], qdt)]
    st.metric("人数", f"{len(active)} 人")
    st.write("**勤務中:** " + (" / ".join(active) if active else "なし"))

# 時間帯人数 + 名前（表+グラフ）
slots = build_slots(open_dt, close_dt, step_min)
labels = [t.strftime("%H:%M") for t in slots]
step = timedelta(minutes=step_min)

counts, name_list = [], []
for s0 in slots:
    s1 = s0 + step
    names = [p["name"] for p in staff if working_in_slot(p["segs"], s0, s1)]
    counts.append(len(names))
    name_list.append(" / ".join(names))

head_df = pd.DataFrame({"時間": labels, "人数": counts, "名前": name_list})

c3, c4 = st.columns([1, 1])
with c3:
    st.write(f"### 🧮 時間帯人数（{step_min}分刻み）＋名前")
    st.dataframe(head_df, use_container_width=True)
with c4:
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

# ガント（休憩は抜ける）
st.write("### 📊 ガント（反映データ + 休憩）")
fig2, ax2 = plt.subplots(figsize=(12, max(3, 0.6 * len(staff))))
y_height, y_gap = 8, 4
yticks, ylabels = [], []

for i, p in enumerate(staff):
    y = i * (y_height + y_gap)
    yticks.append(y + y_height/2)
    ylabels.append(p["name"])

    bars = []
    for (a, b) in p["segs"]:
        x0 = minutes_from(open_dt, a)
        w = minutes_from(open_dt, b) - x0
        if w > 0:
            bars.append((x0, w))
    ax2.broken_barh(bars, (y, y_height), facecolors="#7fbf00", edgecolors="none", alpha=0.85)

    ax2.text(minutes_from(open_dt, p["start_dt"]), y + y_height + 1,
             f"{p['start_dt'].strftime('%H:%M')}-{p['end_dt'].strftime('%H:%M')}",
             va="bottom", ha="left", fontsize=9)

    if p["breaks"]:
        btxt = " / ".join([f"{bs.strftime('%H:%M')}-{be.strftime('%H:%M')}" for (bs, be) in p["breaks"]])
        ax2.text(minutes_from(open_dt, p["end_dt"]), y + y_height + 1,
                 f" (休 {btxt})",
                 va="bottom", ha="left", fontsize=9)

total_min = (close_dt - open_dt).total_seconds() / 60
ax2.set_xlim(0, total_min)

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
ax2.set_yticklabels(ylabels)
ax2.grid(True, axis="x", alpha=0.25)
ax2.set_title(f"Gantt ({target_day.isoformat()})")
st.pyplot(fig2)

st.info("管理者URL例： `http://<host>:8501/?mode=admin`（パスワード必須）")
