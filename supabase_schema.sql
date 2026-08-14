create table if not exists public.options_snapshots (
    id bigint generated always as identity primary key,
    symbol text not null,
    snapshot_date date not null,
    spot double precision not null,
    net_gex double precision not null,
    gross_gex double precision not null,
    gamma_flip double precision,
    flip_distance_pct double precision,
    call_wall double precision not null,
    put_wall double precision not null,
    call_open_interest bigint not null,
    put_open_interest bigint not null,
    put_call_oi_ratio double precision,
    net_delta_exposure double precision not null,
    contract_count integer not null,
    min_dte integer not null,
    max_dte integer not null,
    assumption text not null,
    expiration_filter text not null default 'Custom 7–365 DTE',
    dealer_call_weight double precision not null default 1.0,
    dealer_put_weight double precision not null default -1.0,
    strike_profile jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

alter table public.options_snapshots
    add column if not exists expiration_filter text not null default 'Custom 7–365 DTE',
    add column if not exists dealer_call_weight double precision not null default 1.0,
    add column if not exists dealer_put_weight double precision not null default -1.0;

alter table public.options_snapshots
    drop constraint if exists options_snapshots_unique_model;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'options_snapshots_unique_model_v2'
    ) then
        alter table public.options_snapshots
            add constraint options_snapshots_unique_model_v2
            unique (
                symbol,
                snapshot_date,
                expiration_filter,
                assumption,
                dealer_call_weight,
                dealer_put_weight
            );
    end if;
end $$;

create index if not exists options_snapshots_symbol_date_idx
    on public.options_snapshots (symbol, snapshot_date desc);

alter table public.options_snapshots enable row level security;
revoke all on table public.options_snapshots from anon, authenticated;

comment on table public.options_snapshots is
    'Private daily summaries and strike profiles for the options positioning dashboard. Accessed server-side with a secret key.';

create table if not exists public.stock_candles (
    id bigint generated always as identity primary key,
    symbol text not null,
    session_date date not null,
    open double precision not null,
    high double precision not null,
    low double precision not null,
    close double precision not null,
    volume bigint,
    created_at timestamptz not null default now(),
    constraint stock_candles_symbol_date_unique unique (symbol, session_date)
);

create index if not exists stock_candles_symbol_date_idx
    on public.stock_candles (symbol, session_date desc);

alter table public.stock_candles enable row level security;
revoke all on table public.stock_candles from anon, authenticated;

comment on table public.stock_candles is
    'Private split-adjusted daily OHLCV candles cached for dashboard price overlays.';

create table if not exists public.volatility_snapshots (
    id bigint generated always as identity primary key,
    symbol text not null,
    snapshot_date date not null,
    tenor text not null,
    target_dte integer not null,
    actual_dte integer not null,
    expiration date not null,
    spot double precision not null,
    atm_iv double precision,
    call_25d_iv double precision,
    put_25d_iv double precision,
    skew_25d double precision,
    call_10d_iv double precision,
    put_10d_iv double precision,
    skew_10d double precision,
    archive_path text,
    chain_contract_count integer,
    calculation_version text,
    created_at timestamptz not null default now(),
    constraint volatility_snapshots_symbol_date_tenor_unique
        unique (symbol, snapshot_date, tenor)
);

alter table public.volatility_snapshots
    add column if not exists call_10d_iv double precision,
    add column if not exists put_10d_iv double precision,
    add column if not exists skew_10d double precision,
    add column if not exists archive_path text,
    add column if not exists chain_contract_count integer,
    add column if not exists calculation_version text;

create index if not exists volatility_snapshots_symbol_tenor_date_idx
    on public.volatility_snapshots (symbol, tenor, snapshot_date desc);

alter table public.volatility_snapshots enable row level security;
revoke all on table public.volatility_snapshots from anon, authenticated;

comment on table public.volatility_snapshots is
    'Private constant-tenor volatility summaries. surface_v3_gex rows retain ATM, 10D and 25D IV metrics plus a link to an archived bounded chain with locally reconstructed IV, delta and gamma; legacy rows remain untouched.';

create table if not exists public.collection_runs (
    id bigint generated always as identity primary key,
    collector text not null,
    github_run_id text,
    symbol text not null,
    requested_date date not null,
    snapshot_date date,
    status text not null,
    contract_count integer,
    archive_path text,
    archive_bytes bigint,
    summary_rows_saved integer not null default 0,
    unavailable_tenors text,
    api_credits_consumed integer,
    api_credits_remaining integer,
    min_dte integer not null default 0,
    max_dte integer not null default 45,
    range_filter text not null default 'otm',
    strike_limit integer not null default 30,
    calculation_version text,
    collection_tier text not null default 'legacy',
    error text,
    created_at timestamptz not null default now()
);

alter table public.collection_runs
    add column if not exists collection_tier text not null default 'legacy';

create index if not exists collection_runs_requested_date_symbol_idx
    on public.collection_runs (requested_date desc, symbol);
create index if not exists collection_runs_collector_created_idx
    on public.collection_runs (collector, created_at desc);
create index if not exists collection_runs_collector_date_tier_idx
    on public.collection_runs (collector, requested_date desc, collection_tier);

alter table public.collection_runs enable row level security;
revoke all on table public.collection_runs from anon, authenticated;

comment on table public.collection_runs is
    'Private audit log for MarketData option-chain collection, including actual credit usage, archive metadata, and whether a request saved a full GEX surface or the 25D-priority fallback.';

insert into storage.buckets (id, name, public)
values ('options-chain-archive', 'options-chain-archive', false)
on conflict (id) do nothing;

-- Make newly created tables and constraints visible to Supabase's REST API immediately.
notify pgrst, 'reload schema';
