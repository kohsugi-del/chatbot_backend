-- ============================================================
-- 会話ログ用テーブル（Supabase / PostgreSQL 用）
-- ローカル SQLite は models_log.py から SQLAlchemy が自動生成
-- ============================================================

-- セッション（1回の会話）
CREATE TABLE IF NOT EXISTS sessions (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    municipality_id text        NOT NULL DEFAULT 'htrk-asahikawa',
    started_at      timestamptz NOT NULL DEFAULT now()
);

-- 発話（1ターンごとの記録）
CREATE TABLE IF NOT EXISTS turns (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid        REFERENCES sessions(id),
    turn_order  integer     NOT NULL,
    role        text        NOT NULL,           -- 'user' or 'assistant'
    content     text        NOT NULL,
    user_type   text,                           -- 'jobseeker' / 'company' / 'other'
    topic_type  text,                           -- 'job' / 'site_usage' / 'other'
    bias_type   text,                           -- 'loss_aversion' / 'status_quo' / 'choice_overload'
    created_at  timestamptz NOT NULL DEFAULT now()
);
