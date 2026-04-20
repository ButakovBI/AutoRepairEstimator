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
    ml_error_code VARCHAR,
    ml_error_message VARCHAR,
    idempotency_key VARCHAR UNIQUE
);

-- Defensive-ish migration for upgrades of existing deployments: the
-- initial schema did not carry these two columns, and without them the
-- AbandonRequestUseCase and ProcessInferenceResultUseCase silently drop
-- the user-visible reason for a failed/abandoned request on save.
-- IF NOT EXISTS keeps this idempotent on fresh installs where the
-- CREATE TABLE above already provisioned the columns.
ALTER TABLE repair_requests
    ADD COLUMN IF NOT EXISTS ml_error_code VARCHAR,
    ADD COLUMN IF NOT EXISTS ml_error_message VARCHAR;

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

-- pricing_rules: one row per (part, damage). The workshop rate card (see
-- thesis tables 5 and 6) is inherently a range rather than a single number
-- because real-world labour always has an upper and lower estimate. We store
-- four columns so the bot can render e.g. "23–30 тыс. руб., 2–3 дня" without
-- losing precision. For scratches we store the painting range here; the
-- cheaper polishing alternative is a separate note driven by the constants
-- in auto_repair_estimator.backend.domain.value_objects.pricing_constants.
CREATE TABLE IF NOT EXISTS pricing_rules (
    id SERIAL PRIMARY KEY,
    part_type VARCHAR NOT NULL,
    damage_type VARCHAR NOT NULL,
    labor_hours_min FLOAT NOT NULL,
    labor_hours_max FLOAT NOT NULL,
    labor_cost_min_rub FLOAT NOT NULL,
    labor_cost_max_rub FLOAT NOT NULL,
    UNIQUE (part_type, damage_type),
    CHECK (labor_hours_min <= labor_hours_max),
    CHECK (labor_cost_min_rub <= labor_cost_max_rub)
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

-- Seed data is a 1:1 transcription of thesis tables 5 (cost, thousands of
-- RUB) and 6 (duration, business days/hours). Conversions used:
--   * costs are stored in RUB: table value * 1000
--   * "1 day"  = 8 h, "1.5 days" = 12 h, "2 days"   = 16 h,
--     "2-3 days" = 16-24 h, "5 days" = 40 h, "0.5 days" = 4 h,
--     "1-2 days" = 8-16 h.
-- Damage-type mapping to table columns:
--   * scratch     -> "Царапина (покраска)" (painting; polishing is a soft
--                    alternative surfaced via the bot note, not a row here)
--   * rust        -> same as "Царапина (покраска)" per requirements spec
--   * dent        -> "Вмятина (рихтовка + покраска)"
--   * paint_chip  -> "Замена детали"
--   * crack       -> "Замена детали" for body parts only (never on glass:
--                    on glass/headlight only broken_glass / broken_headlight
--                    are priced; other damages there are filtered out)
--   * broken_glass / broken_headlight -> "Замена детали"
--   * flat_tire / anything on wheel   -> no row here (routed to the tyre
--                    shop via a user-facing note)
INSERT INTO pricing_rules (part_type, damage_type,
                           labor_hours_min, labor_hours_max,
                           labor_cost_min_rub, labor_cost_max_rub) VALUES
    -- Front fender: замена 20, вмятина 23-30, 2-3 дня
    ('front_fender',     'scratch',          8,  8,   10000,  18000),
    ('front_fender',     'rust',             8,  8,   10000,  18000),
    ('front_fender',     'dent',            16, 24,   23000,  30000),
    ('front_fender',     'paint_chip',       8,  8,   20000,  20000),
    ('front_fender',     'crack',            8,  8,   20000,  20000),
    -- Rear fender: замена 75-100, 5 дней
    ('rear_fender',      'scratch',          8,  8,   10000,  18000),
    ('rear_fender',      'rust',             8,  8,   10000,  18000),
    ('rear_fender',      'dent',            16, 24,   23000,  30000),
    ('rear_fender',      'paint_chip',      40, 40,   75000, 100000),
    ('rear_fender',      'crack',           40, 40,   75000, 100000),
    -- Door: замена 20, 1.5-2 дня
    ('door',             'scratch',          8,  8,   10000,  18000),
    ('door',             'rust',             8,  8,   10000,  18000),
    ('door',             'dent',            16, 24,   23000,  30000),
    ('door',             'paint_chip',      12, 16,   20000,  20000),
    ('door',             'crack',           12, 16,   20000,  20000),
    -- Trunk: замена 20, 1.5-2 дня
    ('trunk',            'scratch',          8,  8,   10000,  18000),
    ('trunk',            'rust',             8,  8,   10000,  18000),
    ('trunk',            'dent',            16, 24,   23000,  30000),
    ('trunk',            'paint_chip',      12, 16,   20000,  20000),
    ('trunk',            'crack',           12, 16,   20000,  20000),
    -- Roof: замена 75-100, 5 дней
    ('roof',             'scratch',          8,  8,   10000,  18000),
    ('roof',             'rust',             8,  8,   10000,  18000),
    ('roof',             'dent',            16, 24,   23000,  30000),
    ('roof',             'paint_chip',      40, 40,   75000, 100000),
    ('roof',             'crack',           40, 40,   75000, 100000),
    -- Hood: вмятина 30-35 (2 дня), замена 28-30 (2 дня)
    ('hood',             'scratch',          8,  8,   10000,  18000),
    ('hood',             'rust',             8,  8,   10000,  18000),
    ('hood',             'dent',            16, 16,   30000,  35000),
    ('hood',             'paint_chip',      16, 16,   28000,  30000),
    ('hood',             'crack',           16, 16,   28000,  30000),
    -- Bumper: вмятина 3-5 (1-2 дня), замена 18 (1.5 дня)
    ('bumper',           'scratch',          8,  8,   10000,  18000),
    ('bumper',           'rust',             8,  8,   10000,  18000),
    ('bumper',           'dent',             8, 16,    3000,   5000),
    ('bumper',           'paint_chip',      12, 12,   18000,  18000),
    ('bumper',           'crack',           12, 12,   18000,  18000),
    -- Front windshield: только замена 5-10, 1 день
    ('front_windshield', 'broken_glass',     8,  8,    5000,  10000),
    -- Rear windshield: маппится как лобовое стекло
    ('rear_windshield',  'broken_glass',     8,  8,    5000,  10000),
    -- Side window: только замена 3, 1 день
    ('side_window',      'broken_glass',     8,  8,    3000,   3000),
    -- Headlight: только замена 3, 0.5 дня
    ('headlight',        'broken_headlight', 4,  4,    3000,   3000)
ON CONFLICT (part_type, damage_type) DO NOTHING;
