-- detection_stats.db 建库脚本  |  SQLite 3

-- ① 检测会话主表
CREATE TABLE IF NOT EXISTS detection_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_type        VARCHAR(10)  NOT NULL CHECK (session_type IN ('image','video')),
    source_filename     VARCHAR(255),
    source_path         VARCHAR(512),
    model_name          VARCHAR(100) NOT NULL,
    model_path          VARCHAR(512),
    confidence          FLOAT        DEFAULT 0.25,
    iou_threshold       FLOAT        DEFAULT 0.45,
    frame_count         INTEGER      DEFAULT 1,
    processed_frames    INTEGER      DEFAULT 0,
    total_object_count  INTEGER      DEFAULT 0,
    unique_class_count  INTEGER      DEFAULT 0,
    status              VARCHAR(20)  DEFAULT 'pending'
                        CHECK (status IN ('pending','running','completed','failed')),
    error_message       TEXT,
    output_path         VARCHAR(512),
    duration_ms         INTEGER,
    created_at          DATETIME     DEFAULT (datetime('now', 'localtime')),
    updated_at          DATETIME     DEFAULT (datetime('now', 'localtime')),
    completed_at        DATETIME
);

CREATE INDEX IF NOT EXISTS idx_sess_status   ON detection_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sess_created  ON detection_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sess_model    ON detection_sessions(model_name);

-- ② 各类别计数明细（核心）
CREATE TABLE IF NOT EXISTS detection_classes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER      NOT NULL,
    class_name      VARCHAR(50)  NOT NULL,
    class_id        INTEGER      NOT NULL,
    count           INTEGER      NOT NULL DEFAULT 0,
    avg_confidence  FLOAT        DEFAULT 0.0,
    frame_index     INTEGER      DEFAULT 0,
    recorded_at     DATETIME     DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES detection_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cls_session  ON detection_classes(session_id);
CREATE INDEX IF NOT EXISTS idx_cls_name     ON detection_classes(class_name);
CREATE INDEX IF NOT EXISTS idx_cls_recorded ON detection_classes(recorded_at);

-- ③ 日汇总表（预聚合）
CREATE TABLE IF NOT EXISTS daily_stats (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    stat_date                DATE UNIQUE NOT NULL,

    total_sessions           INTEGER DEFAULT 0,
    total_images             INTEGER DEFAULT 0,
    total_videos            INTEGER DEFAULT 0,
    total_frames            INTEGER DEFAULT 0,
    total_objects           INTEGER DEFAULT 0,

    pedestrian_count         INTEGER DEFAULT 0,
    people_count             INTEGER DEFAULT 0,
    bicycle_count            INTEGER DEFAULT 0,
    car_count                INTEGER DEFAULT 0,
    van_count                INTEGER DEFAULT 0,
    truck_count              INTEGER DEFAULT 0,
    tricycle_count           INTEGER DEFAULT 0,
    awning_tricycle_count    INTEGER DEFAULT 0,
    bus_count                INTEGER DEFAULT 0,
    motor_count              INTEGER DEFAULT 0,

    avg_processing_time_ms   FLOAT DEFAULT 0,
    updated_at               DATETIME DEFAULT (datetime('now', 'localtime'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dstat_date ON daily_stats(stat_date);

-- ④ 导出日志
CREATE TABLE IF NOT EXISTS export_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    export_type    VARCHAR(20)  NOT NULL,
    date_from       DATE,
    date_to         DATE,
    filter_class    VARCHAR(50),
    file_path       VARCHAR(512),
    record_count    INTEGER      DEFAULT 0,
    created_at      DATETIME     DEFAULT (datetime('now', 'localtime'))
);

-- ⑤ 触发器：自动更新 updated_at
CREATE TRIGGER IF NOT EXISTS trg_sess_updated
AFTER UPDATE ON detection_sessions
BEGIN
    UPDATE detection_sessions
    SET updated_at = datetime('now', 'localtime')
    WHERE id = NEW.id;
END;
