CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS repair_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id BIGINT NOT NULL,
    user_id BIGINT,
    mode VARCHAR NOT NULL CHECK (mode IN ('ml', 'manual')),
    status VARCHAR NOT NULL CHECK (status IN ('created', 'queued', 'processing', 'pricing', 'done', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    timeout_at TIMESTAMPTZ NOT NULL,
    original_image_key VARCHAR,
    composited_image_key VARCHAR,
    idempotency_key VARCHAR UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_repair_requests_status_timeout
    ON repair_requests (status, timeout_at);

CREATE TABLE IF NOT EXISTS detected_parts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL REFERENCES repair_requests (id) ON DELETE CASCADE,
    part_type VARCHAR NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_w FLOAT,
    bbox_h FLOAT,
    crop_image_key VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_detected_parts_request_id
    ON detected_parts (request_id);

CREATE TABLE IF NOT EXISTS detected_damages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL REFERENCES repair_requests (id) ON DELETE CASCADE,
    part_id UUID REFERENCES detected_parts (id) ON DELETE SET NULL,
    damage_type VARCHAR NOT NULL,
    part_type VARCHAR NOT NULL,
    source VARCHAR NOT NULL CHECK (source IN ('ml', 'manual')),
    confidence FLOAT,
    mask_image_key VARCHAR,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_detected_damages_request_id
    ON detected_damages (request_id);

CREATE TABLE IF NOT EXISTS pricing_rules (
    id SERIAL PRIMARY KEY,
    part_type VARCHAR NOT NULL,
    damage_type VARCHAR NOT NULL,
    labor_hours FLOAT NOT NULL,
    labor_cost_rub FLOAT NOT NULL,
    UNIQUE (part_type, damage_type)
);

CREATE TABLE IF NOT EXISTS outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id UUID NOT NULL,
    topic VARCHAR NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_events_unpublished
    ON outbox_events (created_at)
    WHERE published_at IS NULL;

INSERT INTO pricing_rules (part_type, damage_type, labor_hours, labor_cost_rub) VALUES
    ('bumper_front',      'scratch',    1.0,  800.0),
    ('bumper_front',      'dent',       2.0, 1500.0),
    ('bumper_front',      'crack',      3.0, 2500.0),
    ('bumper_front',      'rust',       2.5, 2000.0),
    ('bumper_front',      'paint_chip', 0.5,  400.0),
    ('bumper_rear',       'scratch',    1.0,  800.0),
    ('bumper_rear',       'dent',       2.0, 1500.0),
    ('bumper_rear',       'crack',      3.0, 2500.0),
    ('bumper_rear',       'rust',       2.5, 2000.0),
    ('bumper_rear',       'paint_chip', 0.5,  400.0),
    ('door_front_left',   'scratch',    1.5, 1200.0),
    ('door_front_left',   'dent',       3.0, 2500.0),
    ('door_front_left',   'crack',      4.0, 3500.0),
    ('door_front_left',   'rust',       3.5, 3000.0),
    ('door_front_left',   'paint_chip', 0.5,  400.0),
    ('door_front_right',  'scratch',    1.5, 1200.0),
    ('door_front_right',  'dent',       3.0, 2500.0),
    ('door_front_right',  'crack',      4.0, 3500.0),
    ('door_front_right',  'rust',       3.5, 3000.0),
    ('door_front_right',  'paint_chip', 0.5,  400.0),
    ('door_rear_left',    'scratch',    1.5, 1200.0),
    ('door_rear_left',    'dent',       3.0, 2500.0),
    ('door_rear_left',    'crack',      4.0, 3500.0),
    ('door_rear_left',    'rust',       3.5, 3000.0),
    ('door_rear_left',    'paint_chip', 0.5,  400.0),
    ('door_rear_right',   'scratch',    1.5, 1200.0),
    ('door_rear_right',   'dent',       3.0, 2500.0),
    ('door_rear_right',   'crack',      4.0, 3500.0),
    ('door_rear_right',   'rust',       3.5, 3000.0),
    ('door_rear_right',   'paint_chip', 0.5,  400.0),
    ('hood',              'scratch',    1.5, 1200.0),
    ('hood',              'dent',       3.5, 3000.0),
    ('hood',              'crack',      5.0, 4500.0),
    ('hood',              'rust',       4.0, 3500.0),
    ('hood',              'paint_chip', 0.5,  400.0),
    ('trunk',             'scratch',    1.5, 1200.0),
    ('trunk',             'dent',       3.0, 2500.0),
    ('trunk',             'crack',      4.5, 4000.0),
    ('trunk',             'rust',       3.5, 3000.0),
    ('trunk',             'paint_chip', 0.5,  400.0),
    ('fender_front_left', 'scratch',    1.0, 1000.0),
    ('fender_front_left', 'dent',       2.5, 2000.0),
    ('fender_front_left', 'crack',      3.5, 3000.0),
    ('fender_front_left', 'rust',       3.0, 2500.0),
    ('fender_front_left', 'paint_chip', 0.5,  400.0),
    ('fender_front_right','scratch',    1.0, 1000.0),
    ('fender_front_right','dent',       2.5, 2000.0),
    ('fender_front_right','crack',      3.5, 3000.0),
    ('fender_front_right','rust',       3.0, 2500.0),
    ('fender_front_right','paint_chip', 0.5,  400.0),
    ('fender_rear_left',  'scratch',    1.0, 1000.0),
    ('fender_rear_left',  'dent',       2.5, 2000.0),
    ('fender_rear_left',  'crack',      3.5, 3000.0),
    ('fender_rear_left',  'rust',       3.0, 2500.0),
    ('fender_rear_left',  'paint_chip', 0.5,  400.0),
    ('fender_rear_right', 'scratch',    1.0, 1000.0),
    ('fender_rear_right', 'dent',       2.5, 2000.0),
    ('fender_rear_right', 'crack',      3.5, 3000.0),
    ('fender_rear_right', 'rust',       3.0, 2500.0),
    ('fender_rear_right', 'paint_chip', 0.5,  400.0),
    ('headlight_left',    'scratch',    0.5,  600.0),
    ('headlight_left',    'dent',       1.0, 1200.0),
    ('headlight_left',    'crack',      2.0, 3500.0),
    ('headlight_left',    'rust',       1.5, 2000.0),
    ('headlight_left',    'paint_chip', 0.5,  600.0),
    ('headlight_right',   'scratch',    0.5,  600.0),
    ('headlight_right',   'dent',       1.0, 1200.0),
    ('headlight_right',   'crack',      2.0, 3500.0),
    ('headlight_right',   'rust',       1.5, 2000.0),
    ('headlight_right',   'paint_chip', 0.5,  600.0)
ON CONFLICT (part_type, damage_type) DO NOTHING;
