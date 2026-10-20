-- ===================================================================
-- Oracle 21c DDL: app_users テーブルおよびシーケンス定義
-- ===================================================================

-- 1. シーケンス作成 (自動採番用)
CREATE SEQUENCE app_users_seq
    START WITH 1
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;

-- 2. テーブル作成
CREATE TABLE app_users (
    id          NUMBER(19)          DEFAULT app_users_seq.NEXTVAL,
    name        VARCHAR2(100)       NOT NULL,
    email       VARCHAR2(100)       NOT NULL,
    create_at   TIMESTAMP           DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT pk_app_users PRIMARY KEY (id),
    CONSTRAINT uk_app_users_name UNIQUE (name)
);

-- 3. 論理コメント（定義情報の可視化）
COMMENT ON TABLE  app_users           IS 'アプリケーションユーザー管理テーブル';
COMMENT ON COLUMN app_users.id        IS 'ユーザーID (app_users_seq.NEXTVAL により自動採番)';
COMMENT ON COLUMN app_users.name      IS 'ユーザー名 (必須・一意)';
COMMENT ON COLUMN app_users.email     IS 'メールアドレス (必須)';
COMMENT ON COLUMN app_users.create_at IS '登録日時 (デフォルト: CURRENT_TIMESTAMP)';
