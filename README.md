# Options Positioning Dashboard

A private, mobile-friendly Streamlit dashboard for slower-moving US equity and ETF options positioning. It uses MarketData.app end-of-day option chains, saves daily summaries to Supabase, and can collect a watchlist automatically with GitHub Actions.

## What this version includes

- Seven quick expiration filters plus a synchronized custom 0–1,095 DTE range slider
- Gamma exposure by strike and separate net/absolute-total gamma charts by expiration
- Modelled gamma flip across a 70%–130% underlying-price range
- Call wall and put wall
- Net and absolute-total gamma exposure per 1% underlying move
- Put/call open-interest ratio and net delta exposure
- Fixed assumptions plus adjustable call/put dealer weights from −1 to +1
- Split-adjusted daily price candles cached in Supabase, with a separate real-time stock-price overlay
- Price/history overlays for spot, flip, walls, net GEX, and put/call OI
- A large price-and-gamma map with candlestick/line modes and two right-side gamma-profile layouts
- TradingView Lightweight Charts interactions across all analytical charts: mouse/touch panning, wheel/pinch zoom, crosshairs, and tappable series controls
- CSV export of the filtered chain
- Responsive layout for desktop and phone browsers
- Optional scheduled snapshots after each US market session

The app intentionally uses historical end-of-day chains. This fits medium/long-term positioning analysis and avoids spending credits on data freshness that the dashboard does not need.

If the requested date is not yet available on an EOD-only MarketData.app plan, the app automatically retries the provider's latest fully closed session. This keeps manual and scheduled collection working before publication, on weekends, and around market holidays.

## 1. Create the Supabase table

1. Open your Supabase project. If you already deployed the earlier version, repeat these steps once; the script safely adds the new columns and price-candle table.
2. Go to **SQL Editor** and create a new query.
3. Paste all of [`supabase_schema.sql`](supabase_schema.sql) and run it. This creates or upgrades `options_snapshots` and creates `stock_candles`.
4. Open **Connect** (or **Integrations → Data API**) and copy the Project URL.
5. In **Project Settings → API Keys**, copy the `sb_secret_...` value shown under **Secret keys**.

The secret key is used only on the Streamlit/GitHub server. Never put it in source code, a screenshot, chat, or a public GitHub field. The publishable key is not needed for this dashboard.

## 2. Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
streamlit run app.py
```

Edit `.streamlit/secrets.toml` before the last command:

```toml
MARKETDATA_TOKEN = "your_marketdata_app_token"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SECRET_KEY = "your_sb_secret_key"
AUTO_SAVE_SNAPSHOTS = true
```

The real `secrets.toml` is excluded by `.gitignore` and must never be committed.

## 3. Put the project on GitHub

Create a new empty repository on GitHub—do not add a README or `.gitignore` there—then run these commands from the dashboard folder:

```bash
git branch -M main
git add .
git commit -m "Build options positioning dashboard"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

GitHub may ask you to sign in through the browser or use GitHub Desktop. Do not upload `.streamlit/secrets.toml`.

## 4. Deploy on Streamlit Community Cloud

1. In Streamlit Community Cloud, choose **Create app**.
2. Select the GitHub repository, branch `main`, and entrypoint `app.py`.
3. Open **Advanced settings → Secrets**.
4. Paste the same TOML values shown above.
5. Deploy, then use Streamlit's sharing controls to restrict access to the email addresses you invite.

On a phone, open the deployed URL in Safari or Chrome and add it to the home screen. No separate mobile app is needed.

## 5. Turn on automatic daily history

The included workflow runs at 01:00 UTC Tuesday through Saturday, after the prior US market session.

In the GitHub repository:

1. Go to **Settings → Secrets and variables → Actions**.
2. Add these repository secrets:
   - `MARKETDATA_TOKEN`
   - `SUPABASE_URL`
   - `SUPABASE_SECRET_KEY`
3. Add these repository variables:
   - `WATCHLIST` — comma-separated, for example `SPY,QQQ,NVDA,MSFT`
   - `EXPIRATION_FILTER` — use one exact value from the list below; if omitted, the workflow uses the credit-conscious `61–120 DTE`
   - `MIN_OPEN_INTEREST` — optional; defaults to `10` to exclude tiny positions and reduce returned contracts
   - `DEALER_ASSUMPTION` — optional; defaults to `Standard: calls + / puts -`
   - `DEALER_CALL_WEIGHT` — optional; used by `Custom dealer weights`, for example `-0.40`
   - `DEALER_PUT_WEIGHT` — optional; used by `Custom dealer weights`, for example `-0.70`
4. Go to **Actions → Save daily options snapshots → Run workflow** once to test it.

Valid `EXPIRATION_FILTER` values are:

```text
21–60 DTE
61–120 DTE
121–240 DTE
241–365 DTE
Over one year
Monthly expirations only
All expirations
```

Start with a short watchlist and check MarketData.app credit usage before expanding it. The Action log now prints the credits MarketData reports for each ticker and endpoint. Scheduled jobs upsert the same ticker/date/model combination, so rerunning a day does not create duplicate database rows, but it can still repeat API usage.

`All expirations` can be dramatically more expensive than a bounded DTE range for SPY, QQQ, and other large chains. Historical chains are billed according to the number of option symbols returned. The most effective controls are a bounded expiration filter and a sensible minimum OI. Use `All expirations` only when the broader coverage is worth the additional credits.

The collector also saves that session's daily price candle. When the interactive app first needs a longer price chart, it requests up to five years of daily candles once and caches them in `stock_candles`; later chart toggles reuse those rows.

## Model definitions

### Gamma exposure

For each contract:

```text
GEX = gamma × open interest × 100 × spot² × 0.01 × assumed dealer sign
```

This produces estimated dollar gamma exposure for a 1% underlying move. Historical chains can contain rounded zero Greeks and, in some responses, zero or missing IV. The dashboard therefore recalculates Greeks with Black–Scholes from the supplied IV and DTE when available. If IV is unusable, it derives implied volatility from the option quote before calculating the Greeks; vendor Greeks are only the final fallback. An all-zero curve is rejected instead of being shown as a false flip or $50 wall. The gamma-regime curve uses the same model inputs across its simulated spot range.

### Dealer assumptions

- **Standard: calls + / puts −** is the common public-dashboard convention and can produce a gamma flip.
- **Dealers short all options** assigns both calls and puts negative gamma. Because every contract has the same gamma sign, a zero crossing normally does not exist.
- **Dealers long all options** assigns both sides positive gamma and likewise normally has no flip.
- **Custom dealer weights** applies separate call and put weights from −1 to +1:

  ```text
  Scenario GEX = call weight × call gamma exposure + put weight × put gamma exposure
  ```

  `−1` means entirely short, `0` neutral, and `+1` entirely long for that category.

Open interest does not identify the owner or trade direction. These modes are sensitivity tests, not observed dealer books.

### Price data and chart engine

- Price charts use MarketData.app's split-adjusted daily stock-candle endpoint, not a TradingView Essential subscription or an IBKR session.
- The candle endpoint does not supply real-time completed OHLC bars. The dashboard therefore keeps its completed daily candles intact and overlays MarketData.app's real-time SmartMid stock price as a separately labelled line/marker. The option-derived levels remain tied to their end-of-day snapshot.
- TradingView Essential does not expose a private datafeed API for combining hosted TradingView data with this dashboard's Supabase gamma series.
- The interface uses the open-source TradingView Lightweight Charts engine. It is only the chart renderer; MarketData.app and Supabase remain the data sources.
- The pinned chart-engine file and its license are bundled under `static/`, so the deployed app does not depend on a public JavaScript CDN.
- Changing candlestick/line mode, gamma-profile layout, visible period, chart layers, or dealer weights does not call the options API again.
- Moving the DTE slider does require **Load positioning**, because a custom expiration range changes which contracts the provider returns.

### Put/call open interest

For the selected DTE range and minimum-OI filter:

```text
Put/call OI ratio = sum of open interest on included puts / sum of open interest on included calls
```

Open interest is the number of outstanding contracts that remain open. It is not the day's trading volume. A ratio of `1.66`, for example, means the included puts have 1.66 times the open interest of the included calls. It does not reveal whether those contracts were bought or sold, who owns them, or whether the positioning is a directional bet or a hedge.

### Walls and flip

- Call wall: strike with the greatest absolute call-side GEX inside the selected horizon.
- Put wall: strike with the greatest absolute put-side GEX inside the selected horizon.
- Gamma flip: the zero crossing nearest current spot on the repriced net-GEX curve.

These levels can help organize support/resistance and volatility-regime analysis, but they are not guaranteed barriers or trading signals.

## Project structure

```text
app.py                         Streamlit interface
src/analytics.py               GEX, walls, flip, profiles, summaries
src/marketdata_client.py       MarketData.app historical-chain client
src/storage.py                 Private Supabase snapshot storage
src/charts.py                  TradingView Lightweight Charts renderer
src/expiration_filters.py      Expiration presets and API parameters
static/                        Bundled chart engine and license
scripts/daily_snapshot.py      Scheduled watchlist collector
.github/workflows/             GitHub Actions schedule
supabase_schema.sql            Database setup
tests/                         Calculation tests
```

## Verify before deploying

```bash
pytest -q
python -m compileall app.py src scripts
```
