"""
app/db.py  — 数据库操作模块
每次检测自动记录，无需手动管理。.
"""

import os
import sqlite3
from datetime import date, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "detection_stats.db")


def init_db():
    """读取 schema.sql 并执行建表（幂等，可重复执行）。."""
    schema_path = os.path.join(BASE_DIR, "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.sql not found at {schema_path}")
    conn = sqlite3.connect(DB_PATH)
    with open(schema_path, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"[DB] 数据库初始化完成: {DB_PATH}")


def get_conn():
    """返回带 row_factory 的连接（可用列名访问）。."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ 会话操作 ━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_session(
    session_type, source_filename, source_path, model_name, model_path, confidence, iou_threshold
) -> int:
    """新建一条会话记录，返回 session_id。."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO detection_sessions
        (session_type, source_filename, source_path, model_name, model_path,
         confidence, iou_threshold, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (session_type, source_filename, source_path, model_name, model_path, confidence, iou_threshold),
    )
    sid = cur.lastrowid
    conn.commit()
    conn.close()
    return sid


def update_session(session_id, **fields):
    """批量更新会话字段。."""
    if not fields:
        return
    conn = get_conn()
    cur = conn.cursor()
    for k, v in fields.items():
        cur.execute(f"UPDATE detection_sessions SET {k}=? WHERE id=?", (v, session_id))
    conn.commit()
    conn.close()


def complete_session(session_id, total_object_count, unique_class_count, processed_frames, output_path, duration_ms):
    """标记会话为 completed。."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """UPDATE detection_sessions
        SET status='completed',
            total_object_count=?,
            unique_class_count=?,
            processed_frames=?,
            output_path=?,
            duration_ms=?,
            completed_at=datetime('now', 'localtime')
        WHERE id=?""",
        (total_object_count, unique_class_count, processed_frames, output_path, duration_ms, session_id),
    )
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━ 类别记录 ━━━━━━━━━━━━━━━━━━━━━━━━━


def record_class_counts(session_id, class_counts: dict, frame_index=0):
    """写入各类别计数。.

    Args:
        session_id: 会话ID
        class_counts: { class_name: (class_id, count, avg_confidence) }
        frame_index: 帧序号（图片=0，视频为实际帧号）
    """
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    for cls_name, (cls_id, cnt, avg_conf) in class_counts.items():
        cur.execute(
            """INSERT INTO detection_classes
            (session_id, class_name, class_id, count, avg_confidence, frame_index, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, cls_name, cls_id, cnt, avg_conf, frame_index, now),
        )
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━ 日汇总 UPSERT ━━━━━━━━━━━━━━━━━━━━━━━━━


def upsert_daily_stats(session_type, class_counts: dict, duration_ms):
    """日期级 UPSERT： INSERT OR REPLACE + COALESCE 增量累加。.
    """
    today = date.today().isoformat()
    total_objs = sum(v[1] for v in class_counts.values())
    img_delta = 1 if session_type == "image" else 0
    vid_delta = 1 if session_type == "video" else 0

    # 映射 class_name → 列名
    col_map = {
        "pedestrian": "pedestrian_count",
        "people": "people_count",
        "bicycle": "bicycle_count",
        "car": "car_count",
        "van": "van_count",
        "truck": "truck_count",
        "tricycle": "tricycle_count",
        "awning-tricycle": "awning_tricycle_count",
        "bus": "bus_count",
        "motor": "motor_count",
    }

    # 构建 SET 子句（增量累加）
    set_parts = [
        "total_objects  = total_objects  + ?",
        "total_sessions = total_sessions + 1",
        "total_images   = total_images   + ?",
        "total_videos   = total_videos   + ?",
        'updated_at = datetime("now", "localtime")',
    ]
    bindings = [total_objs, img_delta, vid_delta]

    for cls_name, col in col_map.items():
        if cls_name in class_counts:
            set_parts.append(f"{col} = {col} + ?")
            bindings.append(class_counts[cls_name][1])

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""INSERT INTO daily_stats
        (stat_date, total_objects, total_sessions, total_images, total_videos,
         {",".join(col_map.values())})
        VALUES (?, ?, 1, ?, ?,
                {",".join(["COALESCE(?,0)"] * len(col_map))})
        ON CONFLICT(stat_date) DO UPDATE SET
            {", ".join(set_parts)}""",
        [today, total_objs, img_delta, vid_delta] + [class_counts.get(cls, (0, 0))[1] for cls in col_map] + bindings,
    )
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━ 查询 API ━━━━━━━━━━━━━━━━━━━━━━━━━


def query_overview():
    """返回全局概览数据。."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT "
        "COALESCE(SUM(total_objects),0) t, "
        "COALESCE(SUM(total_images),0) img, "
        "COALESCE(SUM(total_videos),0) vid "
        "FROM daily_stats"
    )
    row = cur.fetchone()

    today = date.today().isoformat()
    cur.execute("SELECT total_objects FROM daily_stats WHERE stat_date=?", (today,))
    today_row = cur.fetchone()

    cur.execute("""SELECT stat_date,total_objects,total_images,total_videos
                  FROM daily_stats
                  WHERE stat_date >= date('now','-7 days')
                  ORDER BY stat_date ASC""")
    trend = cur.fetchall()
    conn.close()

    return {
        "total_all": row["t"],
        "total_images": row["img"],
        "total_videos": row["vid"],
        "today_all": today_row["total_objects"] if today_row else 0,
        "trend": [dict(r) for r in trend],
    }


def query_class_stats():
    """返回各类别累计计数。."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""SELECT class_name,
                          SUM(count)             AS total_cnt,
                          COUNT(*)               AS frames_with_class,
                          AVG(avg_confidence)    AS avg_conf
                   FROM detection_classes
                   GROUP BY class_name
                   ORDER BY total_cnt DESC""")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_sessions(limit=20, offset=0, session_type=None, status=None):
    """返回分页会话列表。."""
    conn = get_conn()
    cur = conn.cursor()
    sql = "SELECT * FROM detection_sessions WHERE 1=1"
    params = []
    if session_type:
        sql += " AND session_type=?"
        params.append(session_type)
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cur.execute(sql, params)
    rows = cur.fetchall()

    # 查总数（忽略分页参数）
    count_sql = "SELECT COUNT(*) FROM detection_sessions WHERE 1=1"
    count_params = []
    if session_type:
        count_sql += " AND session_type=?"
        count_params.append(session_type)
    if status:
        count_sql += " AND status=?"
        count_params.append(status)
    cur.execute(count_sql, count_params)
    total = cur.fetchone()[0]

    conn.close()
    return [dict(r) for r in rows], total


def query_export(date_from, date_to, filter_class=None):
    """返回导出用的原始数据。."""
    conn = get_conn()
    cur = conn.cursor()
    params = [date_from, date_to]
    sql = """SELECT s.id session_id, s.session_type, s.source_filename,
                    s.model_name, s.total_object_count, s.duration_ms,
                    s.created_at, c.class_name, c.count, c.avg_confidence
             FROM detection_sessions s
             LEFT JOIN detection_classes c ON c.session_id = s.id
             WHERE DATE(s.created_at) BETWEEN ? AND ?"""
    if filter_class:
        sql += " AND c.class_name = ?"
        params.append(filter_class)
    sql += " ORDER BY s.created_at DESC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
