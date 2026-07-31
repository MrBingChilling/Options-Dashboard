# Options Positioning Dashboard

A private, mobile-friendly Streamlit dashboard for slower-moving US equity and ETF options positioning. It uses MarketData.app end-of-day option chains, saves daily summaries to Supabase, and can collect a watchlist automatically with GitHub Actions.

## What the first version includes

- Gamma exposure by strike and expiration
- Modelled gamma flip across a 70%–130% underlying-price range
- Call wall and put wall
- Net and gross gamma exposure per 1% underlying move
- Put/call open-interest ratio and net delta exposure
- Explicit dealer-position assumption selector
- Daily history for spot, flip, walls, net GEX, and put/call OI
- CSV export of the filtered chain
- Responsive layout for desktop and phone browsers
- Optional scheduled snapshots after each US market session

The app intentionally uses historical end-of-day chains. This fits medium/long-term positioning analysis and avoids spending credits on data freshness that the dashboard does not need.

## 1. Create the Supabase table

1. Open your Supabase project.
2. Go to **SQL Editor** and create a new query.
3. Paste all of [`supabase_schema.sql`](supabase_schema.sql) and run it once.
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
   - `MIN_DTE` — `7`
   - `MAX_DTE` — `365`
4. Go to **Actions → Save daily options snapshots → Run workflow** once to test it.

Start with a short watchlist and check MarketData.app credit usage before expanding it. Scheduled jobs upsert the same ticker/date/model combination, so rerunning a day does not create duplicates.

## Model definitions

### Gamma exposure

For each contract:

```text
GEX = gamma × open interest × 100 × spot² × 0.01 × assumed dealer sign
```

This produces estimated dollar gamma exposure for a 1% underlying move. Current strike GEX uses the provider's contract gamma when available. The gamma-regime curve reprices gamma with Black–Scholes using contract IV, DTE, the selected risk-free rate, and dividend yield.

### Dealer assumptions

- **Standard: calls + / puts −** is the common public-dashboard convention and can produce a gamma flip.
- **Dealers short all options** assigns both calls and puts negative gamma. Because every contract has the same gamma sign, a zero crossing normally does not exist.
- **Dealers long all options** assigns both sides positive gamma and likewise normally has no flip.

Open interest does not identify the owner or trade direction. These modes are sensitivity tests, not observed dealer books.

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
src/charts.py                  Plotly charts
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
