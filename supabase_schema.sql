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
    strike_profile jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    constraint options_snapshots_unique_model
        unique (symbol, snapshot_date, min_dte, max_dte, assumption)
);

create index if not exists options_snapshots_symbol_date_idx
    on public.options_snapshots (symbol, snapshot_date desc);

alter table public.options_snapshots enable row level security;
revoke all on table public.options_snapshots from anon, authenticated;

comment on table public.options_snapshots is
    'Private daily summaries and strike profiles for the options positioning dashboard. Accessed server-side with a secret key.';
