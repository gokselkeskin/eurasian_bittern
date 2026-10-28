#!/usr/bin/env python3
"""GPS-GSM track analysis of Eurasian Bittern 216839 (Aras River, Türkiye, 2022).

Produces all tables, numbers and figures reported in the manuscript.
Usage: python bittern_216839_analysis.py [--input CSV] [--outdir DIR] [--no-show]
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

FLIGHT_SPEED_KMH = 3.0      # regular steps faster than this are movement
DAILY_MIG_KM = 30.0         # observed daily path above this = migratory day
MIG_EVENT_KM = 30.0         # movement events below this = local movement
REGULAR_MAX_H = 6.5         # max interval for an interpretable step speed
GAP_MIN_H = 24.5            # longer intervals are data gaps
STOPOVER_MIN_H = 24.0       # shorter stationary periods are halts
NIGHT_SUN_ELEV = -0.833     # standard sunrise/sunset solar elevation (deg)
EARTH_R_KM = 6371.0088

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_10m_admin_0_countries.geojson")
COUNTRY_NAMES = {"Turkey": "Türkiye"}

LON_LIM, LAT_LIM = (40.5, 66.5), (38.7, 51.1)
STOP_LABEL_OFFSETS = [(0.0, 0.0), (-1.2, 0.9), (-1.3, 1.0), (-2.6, 0.2)]


# ---------------------------------------------------------------- geometry and sun

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def solar_elevation_deg(times_utc, lat, lon):
    """Astronomical Almanac low-precision algorithm (Michalsky 1988)."""
    n = pd.DatetimeIndex(times_utc).to_julian_date().values - 2451545.0
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.asarray(lon, dtype=float)
    mean_long = np.mod(280.460 + 0.9856474 * n, 360.0)
    mean_anom = np.radians(np.mod(357.528 + 0.9856003 * n, 360.0))
    ecl_long = np.radians(mean_long + 1.915 * np.sin(mean_anom) + 0.020 * np.sin(2 * mean_anom))
    obliq = np.radians(23.439 - 0.0000004 * n)
    ra = np.arctan2(np.cos(obliq) * np.sin(ecl_long), np.cos(ecl_long))
    dec = np.arcsin(np.sin(obliq) * np.sin(ecl_long))
    gmst_h = np.mod(18.697374558 + 24.06570982441908 * n, 24.0)
    hour_angle = np.radians(gmst_h * 15.0 + lon) - ra
    elev = np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle))
    return np.degrees(elev)


def sun_event(lat, lon, t_from, t_to, kind):
    """First sunset ('set') or sunrise ('rise') at a fixed position within [t_from, t_to], UTC."""
    grid = pd.date_range(t_from, t_to, freq="1min")
    night = solar_elevation_deg(grid, np.full(len(grid), lat), np.full(len(grid), lon)) < NIGHT_SUN_ELEV
    idx = np.where(~night[:-1] & night[1:])[0] if kind == "set" else np.where(night[:-1] & ~night[1:])[0]
    return grid[idx[0] + 1] if len(idx) else pd.NaT


def interpolate_track(fixes, times):
    tt = fixes["UTC_datetime"].values.astype("int64")
    q = pd.DatetimeIndex(times).values.astype("int64")
    return (np.interp(q, tt, fixes["Latitude"].values), np.interp(q, tt, fixes["Longitude"].values))


def night_fraction(fixes, t0, t1):
    times = pd.date_range(t0, t1, freq="5min")
    lat, lon = interpolate_track(fixes, times)
    night = solar_elevation_deg(times, lat, lon) < NIGHT_SUN_ELEV
    return night.mean()


# ---------------------------------------------------------------- data

def find_input(path):
    if path:
        return path
    for cand in ("data/216839_20221011_174700.csv", "Eruasian_bittern/216839_20221011_174700.csv",
                 "216839_20221011_174700.csv"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("raw CSV not found; pass --input")


def load_fixes(path):
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw["UTC_datetime"] = pd.to_datetime(raw["UTC_datetime"])
    raw = raw.sort_values("UTC_datetime").reset_index(drop=True)
    failed = (raw["Latitude"] == 0) & (raw["Longitude"] == 0)
    fix = raw.loc[~failed].reset_index(drop=True).copy()

    fix["prev_time"] = fix["UTC_datetime"].shift()
    fix["prev_lat"] = fix["Latitude"].shift()
    fix["prev_lon"] = fix["Longitude"].shift()
    fix["step_km"] = haversine_km(fix["prev_lat"], fix["prev_lon"], fix["Latitude"], fix["Longitude"])
    fix["step_h"] = (fix["UTC_datetime"] - fix["prev_time"]).dt.total_seconds() / 3600
    fix["step_kmh"] = fix["step_km"] / fix["step_h"]
    fix["interval_class"] = np.select(
        [fix["step_h"] <= REGULAR_MAX_H, fix["step_h"] <= GAP_MIN_H, fix["step_h"] > GAP_MIN_H],
        ["regular", "coarse", "gap"], default="first")
    fix["arrival_date"] = fix["UTC_datetime"].dt.normalize()
    fix["bat_at_start"] = fix["bat_soc_pct"].shift()
    lat0, lon0 = fix.loc[0, ["Latitude", "Longitude"]]
    fix["net_disp_km"] = haversine_km(lat0, lon0, fix["Latitude"], fix["Longitude"])
    return raw, failed, fix


def add_countries(fix, cache_dir):
    import geopandas as gpd
    path = os.path.join(cache_dir, "ne_10m_admin_0_countries.geojson")
    if not os.path.exists(path):
        urllib.request.urlretrieve(NE_URL, path)
    world = gpd.read_file(path)[["NAME", "geometry"]]
    pts = gpd.GeoDataFrame(fix[["Latitude", "Longitude"]],
                           geometry=gpd.points_from_xy(fix["Longitude"], fix["Latitude"]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, world, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    fix["country"] = joined["NAME"].map(lambda x: COUNTRY_NAMES.get(x, x) if isinstance(x, str) else "Sea").values

    # distance to the nearest coast for fixes at sea (local azimuthal equidistant projection)
    fix["coast_km"] = np.nan
    for i in fix.index[fix["country"] == "Sea"]:
        lat, lon = fix.at[i, "Latitude"], fix.at[i, "Longitude"]
        crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m"
        near = world.cx[lon - 4:lon + 4, lat - 4:lat + 4].to_crs(crs)
        point = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(crs).iloc[0]
        land = near.geometry.union_all() if hasattr(near.geometry, "union_all") else near.geometry.unary_union
        fix.at[i, "coast_km"] = point.distance(land.boundary) / 1000
    return fix


# ---------------------------------------------------------------- classification

def daily_table(fix, daily_km):
    obs = fix[fix["interval_class"].isin(["regular", "coarse"])]
    d = obs.groupby("arrival_date")["step_km"].sum().rename("observed_path_km")
    g = fix[fix["interval_class"] == "gap"].groupby("arrival_date")["step_km"].sum().rename("gap_chord_km")
    n = fix.groupby("arrival_date").size().rename("n_fixes")
    out = pd.concat([n, d, g], axis=1).fillna(0.0)
    out["class"] = np.where(out["observed_path_km"] > daily_km, "migratory", "stationary")
    out.loc[out["gap_chord_km"] > MIG_EVENT_KM, "class"] = "unobserved movement (data gap)"
    return out.reset_index().rename(columns={"arrival_date": "date"})


def segment(fix, speed_kmh=FLIGHT_SPEED_KMH, daily_km=DAILY_MIG_KM):
    """Assign every step to one phase: stopover, halt, migratory or unobserved movement."""
    daily = daily_table(fix, daily_km)
    mig_days = set(daily.loc[daily["class"] == "migratory", "date"])
    moving = (((fix["interval_class"] == "regular") & (fix["step_kmh"] > speed_kmh))
              | ((fix["interval_class"] == "coarse") & fix["arrival_date"].isin(mig_days))).values
    unobs = ((fix["interval_class"] == "gap") & (fix["step_km"] > MIG_EVENT_KM)).values

    events, i = [], 1
    while i < len(fix):
        if unobs[i]:
            events.append(("unobserved movement", i, i))
            i += 1
        elif moving[i]:
            j = i
            while j + 1 < len(fix) and moving[j + 1] and not unobs[j + 1]:
                j += 1
            kind = "migratory movement" if fix["step_km"].iloc[i:j + 1].sum() >= MIG_EVENT_KM else "local"
            events.append((kind, i, j))
            i = j + 1
        else:
            i += 1

    phases, step_phase, cursor = [], np.full(len(fix), -1), 0

    def stationary(a, b):
        sub = fix.iloc[a:b + 1]
        steps = list(range(a + 1, b + 1))
        dur = (sub["UTC_datetime"].iloc[-1] - sub["UTC_datetime"].iloc[0]).total_seconds() / 3600
        lat, lon = sub["Latitude"].median(), sub["Longitude"].median()
        phases.append(dict(
            kind="stopover" if dur >= STOPOVER_MIN_H else ("single fix" if len(sub) == 1 else "halt"),
            start=sub["UTC_datetime"].iloc[0], end=sub["UTC_datetime"].iloc[-1], duration_h=dur,
            distance_km=fix["step_km"].iloc[steps].sum() if steps else 0.0, n_fixes=len(sub),
            lat=lat, lon=lon, first_fix=a, last_fix=b,
            spread_km=haversine_km(lat, lon, sub["Latitude"], sub["Longitude"]).max()))
        step_phase[steps] = len(phases) - 1

    for kind, a, b in events:
        if kind == "local":
            continue
        stationary(cursor, a - 1)
        sub = fix.iloc[a:b + 1]
        t0, t1 = fix["UTC_datetime"].iloc[a - 1], fix["UTC_datetime"].iloc[b]
        window = (t1 - t0).total_seconds() / 3600
        res = ("data gap" if kind == "unobserved movement"
               else "regular" if (sub["interval_class"] == "regular").all() else "low")
        phases.append(dict(
            kind=kind, start=t0, end=t1, duration_h=window, distance_km=sub["step_km"].sum(),
            n_fixes=b - a + 2, lat=fix["Latitude"].iloc[b], lon=fix["Longitude"].iloc[b],
            first_fix=a - 1, last_fix=b, resolution=res, max_step_h=sub["step_h"].max(),
            min_ground_speed_kmh=sub["step_km"].sum() / window))
        step_phase[a:b + 1] = len(phases) - 1
        cursor = b
    stationary(cursor, len(fix) - 1)

    phases = pd.DataFrame(phases)
    out = fix.copy()
    out["moving"], out["unobserved"], out["phase_id"] = moving, unobs, step_phase
    return out, phases, daily


def describe_flights(fix, phases):
    """Day/night composition, sun times and in-flight segment of each movement."""
    rows = []
    for k, p in phases.iterrows():
        if p["kind"] not in ("migratory movement", "unobserved movement"):
            continue
        a, b = int(p["first_fix"]), int(p["last_fix"])
        sub = fix.iloc[a:b + 1]
        dep, arr = sub.iloc[0], sub.iloc[-1]
        row = dict(phase_id=k, kind=p["kind"], resolution=p["resolution"], start=p["start"], end=p["end"],
                   window_h=p["duration_h"], distance_km=p["distance_km"],
                   min_ground_speed_kmh=p["min_ground_speed_kmh"], max_step_h=p["max_step_h"])
        if p["resolution"] == "regular":
            row["night_fraction"] = night_fraction(fix, p["start"], p["end"])
            row["day_night"] = ("Mainly nocturnal" if row["night_fraction"] >= 2 / 3 else
                                "Mainly diurnal" if row["night_fraction"] <= 1 / 3 else "Mixed")
            row["sunset_at_departure_utc"] = sun_event(dep["Latitude"], dep["Longitude"],
                                                       p["start"] - pd.Timedelta(hours=12), p["end"], "set")
            row["sunrise_at_arrival_utc"] = sun_event(arr["Latitude"], arr["Longitude"],
                                                      row["sunset_at_departure_utc"],
                                                      p["end"] + pd.Timedelta(hours=12), "rise")
            if b - a >= 3:  # fixes recorded while the bird was in flight
                inner = fix.iloc[a + 1:b]
                row["in_flight_start"] = inner["UTC_datetime"].iloc[0]
                row["in_flight_end"] = inner["UTC_datetime"].iloc[-1]
                row["in_flight_h"] = (row["in_flight_end"] - row["in_flight_start"]).total_seconds() / 3600
                row["in_flight_km"] = fix["step_km"].iloc[a + 2:b].sum()
                row["in_flight_kmh"] = row["in_flight_km"] / row["in_flight_h"]
        else:
            row["day_night"] = "ND"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables

def table1(fix, phases, flights):
    ph = phases[phases["kind"] != "single fix"].copy()
    last_stop = ph.index[ph["kind"] == "stopover"][-1]
    fl = flights.set_index("phase_id")
    rows = []
    for k, p in ph.iterrows():
        sub = fix.iloc[int(p["first_fix"]):int(p["last_fix"]) + 1]
        move = p["kind"] in ("migratory movement", "unobserved movement")
        if move:
            c0, c1 = sub["country"].iloc[0], sub["country"].iloc[-1]
            country = c0 if c0 == c1 else f"{c0} → {c1}"
            label = ("Data gap" if p["kind"] == "unobserved movement" else
                     "Migratory flight" if p["resolution"] == "regular" else
                     f"Movement ({p['max_step_h']:.0f}-h fixes)")
            speed = f"{p['min_ground_speed_kmh']:.1f}" if p["resolution"] == "regular" else "ND"
            daynight = fl.at[k, "day_night"]
        else:
            country = sub["country"].mode().iloc[0]
            label = "Summering" if k == last_stop else p["kind"].capitalize()
            speed, daynight = "–", "–"
        dur = f"{p['duration_h']:.1f}"
        if p["kind"] == "stopover":
            dur += f" ({p['duration_h'] / 24:.1f} d)"
        rows.append({
            "Start (UTC)": p["start"].strftime("%d-%m %H:%M"), "End (UTC)": p["end"].strftime("%d-%m %H:%M"),
            "Phase": label, "Duration (h)": dur, "Distance (km)": f"{p['distance_km']:,.2f}",
            "Min. ground speed (km/h)": speed, "Day/night": daynight, "Country": country,
            "Coordinates": f"{p['lat']:.3f}°N, {p['lon']:.3f}°E"})
    return pd.DataFrame(rows)


def sensitivity(fix, phases, daily):
    def moves(ph):
        m = ph[ph["kind"] == "migratory movement"]
        return [f"{s:%d-%m %H:%M}->{e:%d-%m %H:%M} ({d:.1f} km)"
                for s, e, d in zip(m["start"], m["end"], m["distance_km"])]

    def stops(ph):
        m = ph[ph["kind"] == "stopover"]
        return [f"{s:%d-%m}->{e:%d-%m}" for s, e in zip(m["start"], m["end"])]

    def mig_days(dl):
        return set(dl.loc[dl["class"] == "migratory", "date"].dt.strftime("%d-%m"))

    base_m, base_s, base_d = moves(phases), stops(phases), mig_days(daily)
    order = lambda s: sorted(s, key=lambda x: (x[3:], x[:2]))

    rows_d = []
    for d in [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]:
        _, ph, dl = segment(fix, FLIGHT_SPEED_KMH, d)
        days = mig_days(dl)
        rows_d.append({"daily threshold (km/day)": d, "n migratory days": len(days),
                       "migratory days": ", ".join(order(days)),
                       "added vs 30": ", ".join(order(days - base_d)),
                       "removed vs 30": ", ".join(order(base_d - days)),
                       "same migratory movements": moves(ph) == base_m, "same stopovers": stops(ph) == base_s})
    rows_v = []
    for v in [1, 2, 3, 4, 5, 8, 10, 12]:
        _, ph, _ = segment(fix, v, DAILY_MIG_KM)
        m = moves(ph)
        rows_v.append({"speed threshold (km/h)": v, "n migratory movements": len(m),
                       "migratory movements": " | ".join(m), "identical to 3 km/h": m == base_m,
                       "total distance (km)": round(ph.loc[ph["kind"] == "migratory movement",
                                                           "distance_km"].sum(), 1)})

    def contiguous(values, same, base):
        flags = [same(v) for v in values]
        i = int(np.argmin(np.abs(np.array(values) - base)))
        lo = hi = i
        while lo > 0 and flags[lo - 1]:
            lo -= 1
        while hi < len(values) - 1 and flags[hi + 1]:
            hi += 1
        return values[lo], values[hi]

    def same_v(v):
        return moves(segment(fix, v, DAILY_MIG_KM)[1]) == base_m

    def same_d(d):
        _, ph, dl = segment(fix, FLIGHT_SPEED_KMH, d)
        return mig_days(dl) == base_d and moves(ph) == base_m and stops(ph) == base_s

    v_range = contiguous([round(x, 1) for x in np.arange(0.5, 15.01, 0.1)], same_v, FLIGHT_SPEED_KMH)
    d_range = contiguous([round(x, 1) for x in np.arange(5, 70.01, 0.5)], same_d, DAILY_MIG_KM)
    return pd.DataFrame(rows_d), pd.DataFrame(rows_v), v_range, d_range, order(base_d)


# ---------------------------------------------------------------- figures

def make_figures(fix, phases, daily, gaps, outdir, show):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    stops = phases[phases["kind"] == "stopover"].reset_index(drop=True)
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.95)
    lat, lon = fix["Latitude"].values, fix["Longitude"].values

    # Figure 2: route
    fig2, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(LON_LIM)
    ax.set_ylim(LAT_LIM)
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution_size=6)
    except Exception as exc:
        print(f"basemap not added: {exc}")
    for i in range(1, len(fix)):
        if fix["unobserved"].iloc[i]:
            ax.plot(lon[i - 1:i + 1], lat[i - 1:i + 1], color="white", lw=2.0, ls=(0, (4, 3)), zorder=3)
        else:
            ax.plot(lon[i - 1:i + 1], lat[i - 1:i + 1], color="black", lw=2.0, alpha=0.85, zorder=3)
    ax.scatter(lon, lat, s=14, color="red", edgecolor="white", linewidth=0.4, zorder=4)
    ax.scatter([lon[0]], [lat[0]], s=260, marker="*", color="white", edgecolor="black", zorder=6)
    ax.annotate("Tagging site\n31 Mar 2022", (lon[0], lat[0]), xytext=(lon[0] - 1.6, lat[0] + 1.0),
                ha="center", va="center", fontsize=9, bbox=box, zorder=7,
                arrowprops=dict(arrowstyle="-", color="white", lw=1.0, shrinkB=8))
    for k, s in stops.iterrows():
        if k == 0:
            continue
        r = max(0.30, min(0.6, s["spread_km"] / 111.0 + 0.15))
        ax.add_patch(Circle((s["lon"], s["lat"]), r, fill=False, ec="white", lw=2.2, zorder=5))
        dx, dy = STOP_LABEL_OFFSETS[k] if k < len(STOP_LABEL_OFFSETS) else (0.0, 0.9)
        days = s["duration_h"] / 24
        length = f"(~{days:.0f} days)" if days >= 3 else f"({s['duration_h']:.0f} h)"
        label = (f"Stopover\n{s['start']:%d %b}–{s['end']:%d %b %Y}\n{length}" if k < len(stops) - 1
                 else f"Summering area\nfrom {s['start']:%d %b %Y}")
        ax.annotate(label, (s["lon"], s["lat"]), xytext=(s["lon"] + dx, s["lat"] + dy), ha="center",
                    va="center", fontsize=8.5, bbox=box, zorder=7,
                    arrowprops=dict(arrowstyle="-", color="white", lw=1.0, shrinkB=r * 40))
    for _, s in phases[phases["kind"] == "single fix"].iterrows():
        ax.annotate(f"Single fix\n{s['start']:%d %b %Y}", (s["lon"], s["lat"]),
                    xytext=(s["lon"] + 1.6, s["lat"] - 1.1), ha="center", fontsize=8.5, bbox=box,
                    arrowprops=dict(arrowstyle="-", color="white", lw=1), zorder=7)
    unobs = fix[fix["unobserved"]]
    if len(unobs):
        g = unobs.iloc[0]
        ax.annotate(f"No fixes {g['prev_time']:%d}–{g['UTC_datetime']:%d %b %Y}\n(straight line, route unknown)",
                    ((g["prev_lon"] + g["Longitude"]) / 2, (g["prev_lat"] + g["Latitude"]) / 2),
                    xytext=(56.5, 43.2), ha="center", fontsize=8.5, bbox=box,
                    arrowprops=dict(arrowstyle="-", color="white", lw=1), zorder=7)
    ax.legend(handles=[Line2D([0], [0], color="black", lw=2, label="Route between fixes"),
                       Line2D([0], [0], color="white", lw=2, ls=(0, (4, 3)), label="Data gap (no fixes)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, lw=0,
                              label="GPS fixes")],
              loc="upper left", fontsize=9, facecolor="0.85", framealpha=0.9)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    fig2.tight_layout()

    # Figure 3: net displacement
    fig3, ax = plt.subplots(figsize=(10, 4.6))
    move_gaps = gaps[gaps["chord_km"] > MIG_EVENT_KM]
    for _, g in move_gaps.iterrows():
        ax.axvspan(g["gap_start_utc"], g["gap_end_utc"], color="0.85", zorder=0, lw=0)
    ax.plot(fix["UTC_datetime"], fix["net_disp_km"], color="black", lw=1.8, zorder=2)
    for k, s in stops.iterrows():
        days = s["duration_h"] / 24
        if days < 5:
            continue
        y = fix.loc[(fix["UTC_datetime"] >= s["start"]) & (fix["UTC_datetime"] <= s["end"]), "net_disp_km"].median()
        txt = f"Stopover\n(~{days:.0f} days)" if k < len(stops) - 1 else f"Summering area\n(~{days:.0f} days)"
        ax.text(s["start"] + (s["end"] - s["start"]) / 2, y + 120 if y < 1500 else y - 250, txt,
                ha="center", va="bottom", fontsize=9)
    if len(move_gaps):
        ax.text(move_gaps["gap_start_utc"].iloc[0] - pd.Timedelta(days=1), 1150, "Data gap", ha="right",
                fontsize=9, color="0.3")
    ax.set_ylabel("Net displacement (km)")
    ax.set_xlabel("Date (UTC, 2022)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.25, ls="--")
    ax.set_ylim(-50, 2100)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()

    # Figure 4: daily distance
    fig4, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(daily["date"], daily["observed_path_km"], color="black", lw=1.2, marker="o", ms=3.5, zorder=2,
            label="Observed daily path (UTC day)")
    gd = daily[daily["gap_chord_km"] > MIG_EVENT_KM]
    ax.scatter(gd["date"], gd["gap_chord_km"], marker="D", s=50, facecolor="white", edgecolor="black", zorder=3,
               label="Straight-line distance across a data gap (not a daily value)")
    ax.axhline(DAILY_MIG_KM, color="0.5", lw=0.8, ls=":", zorder=1)
    ax.text(daily["date"].min(), DAILY_MIG_KM + 15, f"{DAILY_MIG_KM:.0f} km/day", fontsize=8, color="0.4")
    ax.set_ylabel("Distance (km)")
    ax.set_xlabel("Date (UTC, 2022)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()

    for name, fig in (("Figure2", fig2), ("Figure3", fig3), ("Figure4", fig4)):
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    raw, failed, fix = load_fixes(find_input(args.input))
    fix = add_countries(fix, args.outdir)
    fix, phases, daily = segment(fix)
    flights = describe_flights(fix, phases)

    total = fix["step_km"].sum()
    assert abs(phases["distance_km"].sum() - total) < 1e-6
    assert (fix["phase_id"].iloc[1:] >= 0).all()

    span_d = (raw["UTC_datetime"].max() - raw["UTC_datetime"].min()).total_seconds() / 86400
    expected = span_d * 24
    gaps = fix.loc[fix["interval_class"] == "gap", ["prev_time", "UTC_datetime", "step_h", "step_km"]].copy()
    gaps.columns = ["gap_start_utc", "gap_end_utc", "gap_h", "chord_km"]
    gaps["interpretation"] = np.where(gaps["chord_km"] > MIG_EVENT_KM, "unobserved movement",
                                      "same site before and after")

    bins = [0, 1.5, 2.5, 3.5, 6.5, 12.5, 24.5, 1e5]
    labels = ["~1 h", "~2 h", "~3 h", "4-6 h", "~12 h", "~24 h", ">24 h"]
    fix["interval_bin"] = pd.cut(fix["step_h"], bins, labels=labels)
    battery = (fix.dropna(subset=["step_h"]).groupby("interval_bin", observed=True)
               .agg(n=("step_h", "size"), mean_pct=("bat_at_start", "mean"),
                    min_pct=("bat_at_start", "min"), max_pct=("bat_at_start", "max")).round(1).reset_index())
    rho = fix[["bat_at_start", "step_h"]].dropna().corr(method="spearman").iloc[0, 1]

    first = fix.iloc[0]
    stops = phases[phases["kind"] == "stopover"].reset_index(drop=True)
    migr = phases[phases["kind"] == "migratory movement"]
    regular = migr[migr["resolution"] == "regular"]
    unobs = phases[phases["kind"] == "unobserved movement"]
    chords = unobs["distance_km"].sum()
    all_gaps = gaps["chord_km"].sum()
    longest = flights[flights["kind"] == "migratory movement"].sort_values("distance_km").iloc[-1]
    sea = fix[fix["country"] == "Sea"]
    sea_step = fix.loc[sea.index[1]] if len(sea) > 1 else None
    dep = phases[phases["kind"] == "migratory movement"].iloc[0]
    arr = stops.iloc[-1]
    mig_stops = stops.iloc[1:-1]
    stop_daily = daily[daily["date"].apply(lambda d: any(
        (s["start"].normalize() < d) and (d <= s["end"].normalize()) for _, s in stops.iloc[2:].iterrows()))]
    sens_d, sens_v, v_range, d_range, base_days = sensitivity(fix, phases, daily)
    tab1 = table1(fix, phases, flights)

    def km(x):
        return f"{x:,.1f}"

    numbers = [
        ("records_logged", len(raw), "records logged by the tag"),
        ("records_failed", int(failed.sum()), "records with latitude = longitude = 0"),
        ("fixes_valid", len(fix), "valid fixes"),
        ("pct_valid_of_logged", f"{100 * len(fix) / len(raw):.1f}", "valid fixes / logged records (%)"),
        ("tracking_days", f"{span_d:.1f}", "tracking span (days)"),
        ("expected_hourly", f"{expected:,.0f}", "records expected at hourly schedule"),
        ("pct_logged_of_schedule", f"{100 * len(raw) / expected:.1f}", "logged records / schedule (%)"),
        ("pct_valid_of_schedule", f"{100 * len(fix) / expected:.1f}", "valid fixes / schedule (%)"),
        ("min_satellites_valid", int(fix["satcount"].min()), "minimum satellite count of valid fixes"),
        ("battery_1h", f"{battery.iloc[0]['mean_pct']:.1f}", "mean battery at start of ~1-h intervals (%)"),
        ("battery_24h", f"{battery.iloc[-2]['mean_pct']:.1f}", "mean battery at start of ~24-h intervals (%)"),
        ("battery_gt24h", f"{battery.iloc[-1]['mean_pct']:.1f}", "mean battery at start of >24-h intervals (%)"),
        ("spearman_battery_interval", f"{rho:.2f}", "Spearman rho, battery vs interval length"),
        ("n_intervals", int(battery["n"].sum()), "intervals between valid fixes"),
        ("n_gaps", len(gaps), "intervals > 24.5 h"),
        ("n_gaps_same_site", int((gaps["chord_km"] <= MIG_EVENT_KM).sum()), "gaps without movement"),
        ("max_chord_same_site", f"{gaps.loc[gaps['chord_km'] <= MIG_EVENT_KM, 'chord_km'].max():.1f}",
         "largest chord of gaps without movement (km)"),
        ("total_path_km", km(total), "total path length incl. gap chords"),
        ("gap_chords_km", km(chords), "chords across gaps with movement"),
        ("observed_path_km", km(total - all_gaps), "path excluding all gaps"),
        ("net_displacement_km", km(fix["net_disp_km"].iloc[-1]), "final net displacement"),
        ("straightness", f"{fix['net_disp_km'].iloc[-1] / total:.2f}", "straightness index (upper bound)"),
        ("max_daily_path_km", f"{daily['observed_path_km'].max():.1f}", "longest observed path in a UTC day"),
        ("max_daily_path_date", f"{daily.loc[daily['observed_path_km'].idxmax(), 'date']:%d %b}", ""),
        ("max_daily_path_in_late_stopovers_km", f"{stop_daily['observed_path_km'].max():.1f}",
         "longest daily path within the Dagestan and Kostanay stopovers"),
        ("tag_lat", f"{first['Latitude']:.3f}", "first fix latitude"),
        ("tag_lon", f"{first['Longitude']:.3f}", "first fix longitude"),
        ("aras_stop_days", f"{stops.iloc[0]['duration_h'] / 24:.1f}", "Aras stopover (days)"),
        ("aras_end", f"{stops.iloc[0]['end']:%d %b}", "last fix of Aras stopover"),
        ("longest_flight_km", f"{longest['distance_km']:.1f}", "longest migratory flight"),
        ("longest_flight_window_h", f"{longest['window_h']:.1f}", "its observation window (h)"),
        ("longest_flight_speed", f"{longest['min_ground_speed_kmh']:.1f}", "its minimum mean ground speed"),
        ("longest_flight_night_pct", f"{100 * longest['night_fraction']:.0f}", "% of window at night"),
        ("longest_flight_in_flight_km", f"{longest['in_flight_km']:.1f}", "distance between in-flight fixes"),
        ("longest_flight_in_flight_h", f"{longest['in_flight_h']:.1f}", "time between in-flight fixes"),
        ("longest_flight_in_flight_kmh", f"{longest['in_flight_kmh']:.1f}", "speed between in-flight fixes"),
        ("sea_fix_1_coast_km", f"{sea['coast_km'].iloc[0]:.1f}" if len(sea) else "", "offshore distance"),
        ("sea_fix_2_coast_km", f"{sea['coast_km'].iloc[1]:.1f}" if len(sea) > 1 else "", "offshore distance"),
        ("sea_step_km", f"{sea_step['step_km']:.1f}" if sea_step is not None else "", "step between sea fixes"),
        ("sea_step_kmh", f"{sea_step['step_kmh']:.1f}" if sea_step is not None else "", "speed between sea fixes"),
        ("mean_min_speed_regular_flights", f"{regular['min_ground_speed_kmh'].mean():.1f}",
         f"mean of {len(regular)} regularly sampled migratory flights"),
        ("migration_days", f"{(arr['start'] - dep['start']).total_seconds() / 86400:.0f}",
         "from departure window start to arrival at summering area"),
        ("migration_stopover_days", f"{mig_stops['duration_h'].sum() / 24:.1f}", "stopovers during migration"),
        ("summering_days", f"{arr['duration_h'] / 24:.0f}", "stationary period at summering area"),
        ("summering_spread_km", f"{arr['spread_km']:.1f}", "max distance from median, summering area"),
        ("unobserved_period_h", f"{(unobs['end'].max() - unobs['start'].min()).total_seconds() / 3600:.0f}",
         "period bridged by gaps with movement"),
        ("daily_threshold_range", f"{d_range[0]}-{d_range[1]}", "km/day with identical classification"),
        ("speed_threshold_range", f"{v_range[0]}-{v_range[1]}", "km/h with identical migratory movements"),
        ("migratory_days", ", ".join(base_days), "UTC days classified as migratory"),
    ]
    numbers += [
        ("total_path_km_2dp", f"{total:,.2f}", "total path length, two decimals (Table 1 sum)"),
        ("longest_flight_in_flight_start", f"{longest['in_flight_start']:%H:%M}", "first in-flight fix (UTC)"),
        ("longest_flight_in_flight_end", f"{longest['in_flight_end']:%H:%M}", "last in-flight fix (UTC)"),
        ("sea_fix_times", ", ".join(f"{t:%d-%m %H:%M}" for t in sea["UTC_datetime"]), "fixes at sea (UTC)"),
    ]
    for d in sorted({longest["start"].normalize(), longest["end"].normalize()}):
        numbers.append((f"daily_path_{d:%d%m}", f"{daily.loc[daily['date'] == d, 'observed_path_km'].iloc[0]:.1f}",
                        "observed daily path on a day of the longest flight"))
    for k, h in phases[phases["kind"] == "halt"].reset_index(drop=True).iterrows():
        numbers.append((f"halt{k}_h", f"{h['duration_h']:.1f}", f"halt from {h['start']:%d-%m %H:%M}"))
    for k, s in stops.iterrows():
        inside = fix[(fix["phase_id"] == phases.index[phases["start"] == s["start"]][0])
                     & (fix["step_km"] > 20)]
        for _, st in inside.iterrows():
            numbers.append((f"stop{k}_move_{st['UTC_datetime']:%d%m}", f"{st['step_km']:.1f}",
                            f"local move >20 km within stopover {k} ({st['step_h']:.0f}-h interval)"))
    for k, s in stops.iterrows():
        numbers.append((f"stop{k}_days", f"{s['duration_h'] / 24:.1f}", f"stopover {k} duration (days)"))
        numbers.append((f"stop{k}_spread_km", f"{s['spread_km']:.1f}", f"stopover {k} max distance from median"))
        numbers.append((f"stop{k}_path_km", f"{s['distance_km']:.1f}", f"stopover {k} local path"))
    for _, f in flights.iterrows():
        tag = f"{f['start']:%d%m}"
        for col in ("window_h", "distance_km", "min_ground_speed_kmh"):
            numbers.append((f"flight_{tag}_{col}", f"{f[col]:.1f}", f["kind"]))
        if f.get("resolution") == "regular":
            numbers.append((f"flight_{tag}_night_pct", f"{100 * f['night_fraction']:.0f}", "% window at night"))
            numbers.append((f"flight_{tag}_sunset_utc", f"{f['sunset_at_departure_utc']:%d-%m %H:%M}", ""))
            numbers.append((f"flight_{tag}_sunrise_utc", f"{f['sunrise_at_arrival_utc']:%d-%m %H:%M}", ""))
    numbers = pd.DataFrame(numbers, columns=["key", "value", "description"])

    outputs = {
        "01_fix_statistics.csv": numbers.iloc[:9],
        "02_interval_vs_battery.csv": battery,
        "03_data_gaps.csv": gaps,
        "04_daily_summary.csv": daily,
        "05_phases.csv": phases,
        "06_table1.csv": tab1,
        "07_flights.csv": flights,
        "08_sensitivity_daily_threshold.csv": sens_d,
        "09_sensitivity_speed_threshold.csv": sens_v,
        "10_steps_labelled.csv": fix.drop(columns=["interval_bin"]),
        "11_manuscript_numbers.csv": numbers,
    }
    for name, df in outputs.items():
        df.to_csv(os.path.join(args.outdir, name), index=False, encoding="utf-8-sig")

    print(numbers.to_string(index=False))
    print("\nTable 1\n", tab1.to_string(index=False))
    shown = sum(float(x.replace(",", "")) for x in tab1["Distance (km)"])
    print(f"\nTable 1 distance sum: {shown:.2f} km (total path {total:.2f} km)")
    assert abs(shown - round(total, 2)) < 0.005, "Table 1 rows do not add up to the total"

    make_figures(fix, phases, daily, gaps, args.outdir, show=not args.no_show)
    print(f"\nOutputs written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
