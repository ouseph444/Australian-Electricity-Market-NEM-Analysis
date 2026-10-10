"""
Generate a realistic SYNTHETIC dataset for practising Australian NEM analysis.

Two financial years (FY2024-25 and FY2025-26) of 30-minute data for four NEM regions:
NSW1, QLD1, SA1, VIC1.

The data is simulated, NOT real AEMO data. It is built from the main physical and
market drivers (weather, rooftop/utility solar, wind, a merit-order price curve,
gas prices, heatwaves, negative prices and price spikes) so that the patterns an
analyst would look for are present.

Run:  python generate_data.py
Out:  data/nem_trading_30min.csv   (region-level 30-min data)
      data/gas_price_daily.csv     (illustrative east-coast gas price)
"""

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, rankdata

SEED = 2026
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ time axis
# AEMO convention: SETTLEMENTDATE is the END of the 30-minute interval, in NEM time
# (AEST, UTC+10, no daylight saving).
ts_end = pd.date_range("2024-07-01 00:30", "2026-07-01 00:00", freq="30min")
ts_mid = ts_end - pd.Timedelta(minutes=15)          # middle of each interval
N = len(ts_end)
N_DAYS = N // 48
hour = ts_mid.hour.values + ts_mid.minute.values / 60
doy = ts_mid.dayofyear.values
dow = ts_mid.dayofweek.values
day_idx = np.repeat(np.arange(N_DAYS), 48)
progress = np.linspace(0, 1, N)                      # 0 -> 1 across the two years

REGIONS = ["NSW1", "QLD1", "SA1", "VIC1"]

# ------------------------------------------------------------------ region settings
CFG = {
    #            lat    lon   mean_op_demand  rooftop_MW  utilsolar_MW(start,end)  wind_MW(start,end)
    "NSW1": dict(lat=-33.9, lon=151.2, demand=8000, rooftop=4400, solar=(3300, 4300), wind=(2200, 2800),
                 t_mean=18.5, t_amp=5.5, dtr=(8, 7), cool=0.015, heat=0.010, clear=0.72, wind_rho=0.35, gas_mult=1.05, coal=1.05, spike_p=0.07),
    "QLD1": dict(lat=-27.5, lon=153.0, demand=6600, rooftop=5000, solar=(5200, 6300), wind=(1300, 1500),
                 t_mean=21.5, t_amp=4.5, dtr=(8, 9), cool=0.014, heat=0.006, clear=0.80, wind_rho=0.15, gas_mult=1.00, coal=0.95, spike_p=0.06),
    "SA1":  dict(lat=-34.9, lon=138.6, demand=1400, rooftop=1600, solar=(1000, 1300), wind=(2600, 3000),
                 t_mean=17.0, t_amp=7.0, dtr=(12, 8), cool=0.025, heat=0.012, clear=0.72, wind_rho=0.65, gas_mult=1.20, coal=1.10, spike_p=0.10),
    "VIC1": dict(lat=-37.8, lon=145.0, demand=5000, rooftop=3000, solar=(2300, 2900), wind=(3800, 4400),
                 t_mean=15.5, t_amp=6.0, dtr=(10, 7), cool=0.020, heat=0.014, clear=0.65, wind_rho=0.65, gas_mult=0.88, coal=0.80, spike_p=0.05),
}


def ou(n, phi, sd=1.0):
    """AR(1) / Ornstein-Uhlenbeck style series with unit variance (times sd)."""
    x = np.zeros(n)
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x * sd


# ------------------------------------------------------------------ shared weather drivers
shared_wind = ou(N, 0.985)
shared_cloud = ou(N_DAYS, 0.4)
shared_temp = ou(N_DAYS, 0.75)

# Heatwave episodes: 2-3 per summer, 2-4 days each (strongest in SA/VIC)
heat_shock = np.zeros(N_DAYS)
summer_windows = [(pd.Timestamp("2024-12-15"), pd.Timestamp("2025-02-20")),
                  (pd.Timestamp("2025-12-15"), pd.Timestamp("2026-02-20"))]
day0 = pd.Timestamp("2024-07-01")
for start, end in summer_windows:
    s, e = (start - day0).days, (end - day0).days
    for _ in range(rng.integers(2, 4)):
        d = rng.integers(s, e - 4)
        length = rng.integers(2, 5)
        magnitude = rng.uniform(8, 12)
        for k in range(length):
            heat_shock[d + k] = max(heat_shock[d + k], magnitude * (1 - 0.15 * k))

# ------------------------------------------------------------------ gas price (daily, A$/GJ)
days = np.arange(N_DAYS)
doy_d = (day0 + pd.to_timedelta(days, unit="D")).dayofyear.values
gas = 13.0 * np.exp(0.17 * ou(N_DAYS, 0.985) + 0.04 * np.cos(2 * np.pi * (doy_d - 200) / 365))
# two supply-constraint episodes (winter 2025, winter 2026)
for centre, size in [(int((pd.Timestamp("2025-06-25") - day0).days), 4.5),
                     (int((pd.Timestamp("2026-06-10") - day0).days), 3.0)]:
    gas += size * np.exp(-0.5 * ((days - centre) / 6.0) ** 2)
gas = np.clip(gas, 7.0, 26.0).round(2)
gas_daily = pd.DataFrame({"date": (day0 + pd.to_timedelta(days, unit="D")).date, "gas_price_aud_gj": gas})
gas_interval = np.repeat(gas, 48)

# price caps (approximate market price cap by financial year) and floor
cap = np.where(ts_mid < pd.Timestamp("2025-07-01"), 17500.0, 20300.0)
FLOOR = -1000.0

# ------------------------------------------------------------------ demand profile shape
def gauss(h, mu, s):
    return np.exp(-0.5 * ((h - mu) / s) ** 2)

profile = 0.80 + 0.14 * gauss(hour, 8.0, 2.2) + 0.30 * gauss(hour, 18.5, 2.6) + 0.10 * gauss(hour, 13.0, 3.5)
profile = profile / profile.mean()
dow_factor = np.where(dow == 5, 0.93, np.where(dow == 6, 0.91, 1.0))

# a shared price-noise component (regions move together a lot)
shared_price_noise = ou(N, 0.90, 0.15)

raw = {}
for r in REGIONS:
    c = CFG[r]

    # ---- temperature (deg C)
    season = c["t_mean"] + c["t_amp"] * np.cos(2 * np.pi * (doy - 25) / 365)
    daily_anom = (0.6 * shared_temp + 0.8 * ou(N_DAYS, 0.75)) * 2.5
    heat_w = {"SA1": 1.0, "VIC1": 0.9, "NSW1": 0.5, "QLD1": 0.2}[r]
    daily_anom = daily_anom + heat_shock * heat_w
    anom = np.interp(np.arange(N), np.arange(N_DAYS) * 48 + 24, daily_anom)   # smooth day-to-day changes
    summer_flag = (np.cos(2 * np.pi * (doy - 25) / 365) + 1) / 2          # 1 summer, 0 winter
    dtr = c["dtr"][1] + (c["dtr"][0] - c["dtr"][1]) * summer_flag          # daily temperature range
    temp = season + anom + 0.5 * dtr * np.cos(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 0.4, N)

    # ---- solar geometry (NEM time is AEST, so solar noon depends on longitude)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
    solar_hour = hour + (c["lon"] - 150.0) / 15.0
    ha = np.radians(15 * (solar_hour - 12))
    lat = np.radians(c["lat"])
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    clear_sky = np.clip(sin_elev, 0, None) ** 1.15

    # ---- cloudiness: daily clearness index + small intraday wobble
    winter = 1 - summer_flag
    mu = np.clip(c["clear"] - (0.10 * winter if r != "QLD1" else 0.02 * winter), 0.3, 0.9)
    z_cloud = 0.5 * shared_cloud + np.sqrt(1 - 0.25) * ou(N_DAYS, 0.4)
    k_daily = beta.ppf(norm.cdf(z_cloud), 5 * mu[::48][:N_DAYS], 5 * (1 - mu[::48][:N_DAYS]))
    k = np.clip(k_daily[day_idx] * (1 + ou(N, 0.9, 0.07)), 0.03, 1.0)

    # ---- utility-scale solar and rooftop solar (MW)
    solar_cap = c["solar"][0] + (c["solar"][1] - c["solar"][0]) * progress
    solar = np.clip(solar_cap * clear_sky * k * 0.92, 0, None)
    rooftop_cap = c["rooftop"] * (1 + 0.14 * progress * 2)
    rooftop = rooftop_cap * clear_sky * k * 0.70

    # ---- wind (MW): persistent weather regimes, windier in winter/spring, a bit windier at night
    z_wind = c["wind_rho"] * shared_wind + np.sqrt(1 - c["wind_rho"] ** 2) * ou(N, 0.985)
    z_wind = z_wind + 0.35 * np.cos(2 * np.pi * (doy - 220) / 365) + 0.15 * np.cos(2 * np.pi * (hour - 2) / 24)
    wind_cf = np.clip(beta.ppf(norm.cdf(z_wind), 1.35, 2.6), 0, 0.97)
    wind_cap = c["wind"][0] + (c["wind"][1] - c["wind"][0]) * progress
    wind = wind_cf * wind_cap

    # ---- operational demand (MW): gross demand minus rooftop solar
    cool = c["cool"] * np.clip(temp - 24, 0, None)
    heat = c["heat"] * np.clip(15 - temp, 0, None)
    gross = profile * dow_factor * (1 + cool + heat) * (1 + ou(N, 0.9, 0.012))
    gross = gross * (c["demand"] + rooftop.mean()) / gross.mean()
    demand = np.clip(gross - rooftop, 0.15 * c["demand"], None)

    # ---- price: position on the supply stack (residual demand) + gas price + noise
    resid = demand - wind - solar
    # 0 = very low residual demand, 1 = extremely tight. Blend of ranks over the whole sample and
    # within each financial year (the supply stack adapts a little as new plant/batteries arrive).
    u_all = rankdata(resid) / (N + 1)
    u_fy = np.empty(N)
    for m in (ts_mid < pd.Timestamp("2025-07-01"), ts_mid >= pd.Timestamp("2025-07-01")):
        u_fy[m] = rankdata(resid[m]) / (m.sum() + 1)
    u = 0.4 * u_all + 0.6 * u_fy
    u_eff = u
    gcc = 9.0 * gas_interval                        # rough gas-fired marginal cost (A$/MWh)
    knots_u = np.array([0.00, 0.17, 0.22, 0.55, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995, 1.00])
    vals = np.column_stack([
        np.zeros(N), np.full(N, 5.0), np.full(N, 42.0 * c["coal"]), np.full(N, 72.0 * c["coal"]),
        0.80 * gcc, 1.05 * gcc, 1.25 * gcc, 1.8 * gcc, 2.9 * gcc, 7 * gcc, 16 * gcc,
    ])
    vals[:, 4:] *= c["gas_mult"]               # regional tightness of the gas-fired part of the stack
    vals[:, 2:4] += 2.0 * (gas_interval[:, None] - 13.0)   # coal bids drift with the gas price (substitution)
    vals = np.maximum.accumulate(vals, axis=1)
    idx = np.clip(np.searchsorted(knots_u, u_eff, side="right") - 1, 0, len(knots_u) - 2)
    w = (u_eff - knots_u[idx]) / (knots_u[idx + 1] - knots_u[idx])
    rows = np.arange(N)
    price = vals[rows, idx] + w * (vals[rows, idx + 1] - vals[rows, idx])
    price = price * np.exp(shared_price_noise + ou(N, 0.8, 0.22))

    # negative prices when residual demand is very low (more common as solar grows)
    neg_thr = {"NSW1": 0.11, "QLD1": 0.16, "SA1": 0.17, "VIC1": 0.14}[r]
    neg = u < neg_thr
    depth = (neg_thr - u[neg]) / neg_thr
    price[neg] = -rng.gamma(1.2, 12 + 40 * depth)
    extreme = neg & (rng.random(N) < 0.004)
    price[extreme] = FLOOR

    # rare scarcity events: 1-4 consecutive intervals at very high prices
    tight = np.where(u_eff > 0.985)[0]
    starts = tight[rng.random(len(tight)) < c["spike_p"]]
    for s in starts:
        for j in range(rng.integers(1, 5)):
            if s + j < N:
                price[s + j] = np.exp(rng.normal(np.log(4500), 0.7))

    raw[r] = dict(demand=demand, wind=wind, solar=solar, temp=temp, price=price)

# ------------------------------------------------------------------ interconnector coupling
# When an interconnector is NOT congested, neighbouring regions share (roughly) one price.
# When it is congested, prices separate. Congestion is a persistent random regime.
P = {r: raw[r]["price"].copy() for r in REGIONS}


def couple(a, b, share_congested):
    congested = ou(N, 0.97) > norm.ppf(1 - share_congested)
    shared = 0.5 * (P[a] + P[b])
    P[a] = np.where(congested, P[a], shared)
    P[b] = np.where(congested, P[b], shared)


couple("SA1", "VIC1", 0.35)
couple("NSW1", "VIC1", 0.35)
couple("NSW1", "QLD1", 0.35)

frames = []
for r in REGIONS:
    d = raw[r]
    basis = {"NSW1": 1.08, "QLD1": 0.98, "SA1": 1.08, "VIC1": 0.85}[r]     # regional price level differences
    price = np.where(P[r] > 0, P[r] * basis, P[r])
    price = np.clip(price, FLOOR, cap).round(2)
    frames.append(pd.DataFrame({
        "settlement_date": ts_end, "region": r,
        "demand_mw": d["demand"].round(1), "wind_mw": d["wind"].round(1), "solar_mw": d["solar"].round(1),
        "temperature_c": d["temp"].round(1), "rrp_aud_mwh": price,
    }))

df = pd.concat(frames).sort_values(["settlement_date", "region"]).reset_index(drop=True)

# ------------------------------------------------------------------ inject realistic data problems (for cleaning practice)
miss_temp = rng.choice(len(df), 60, replace=False)
df.loc[miss_temp, "temperature_c"] = np.nan
miss_wind = rng.choice(len(df), 15, replace=False)
df.loc[miss_wind, "wind_mw"] = np.nan
dups = df.sample(6, random_state=SEED)
df = pd.concat([df, dups]).sort_values(["settlement_date", "region"]).reset_index(drop=True)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/nem_trading_30min.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
gas_daily.to_csv("data/gas_price_daily.csv", index=False)
print(f"nem_trading_30min.csv: {len(df):,} rows | gas_price_daily.csv: {len(gas_daily)} rows")
