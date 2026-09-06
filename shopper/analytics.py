"""Journey analytics: the questions a detector cannot answer.

A person detector answers "where are people in this frame". That is a fact about
one frame and has no memory. Everything here needs **identity across time**:

* did the person who *walked past* this display also *stop* at it (stop rate);
* where did they go next (flow between regions);
* were they browsing or did they come for one thing (shopper type);
* were they with someone (groups).

Detection is the input to this, not a competitor to it: swap the detector and
every number below still computes. That is the whole argument for the module.

No cv2, no events. These would need new `event_type`s in docs/SHARED.md §5 to
reach the dashboard, and that table is frozen -- for now they are report-only.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class AnalyticsParams:
    stop_s: float = 2.0             # standing this long in one place is a "stop"
    min_region_s: float = 0.3       # shorter than this in a region is clipping a corner
    region_debounce_s: float = 0.5  # continuous presence before a region change is real
    walk_min_px: float = 2.0        # per-frame movement below this is jitter, not walking
    passing_s: float = 3.0          # in view less than this and never stopping = passing
    group_dist_scale: float = 1.5   # x mean body width: how close counts as "together"
    group_min_s: float = 3.0        # together this long to be a group
    group_min_frac: float = 0.5     # ...and for this fraction of your shared time
    max_path_segments: int = 8      # trim the printed path; the count is still exact


class Segment:
    """One uninterrupted stay in one region."""

    __slots__ = ("region", "first_f", "last_f", "still")

    def __init__(self, region, first_f):
        self.region, self.first_f, self.last_f, self.still = region, first_f, first_f, 0

    @property
    def frames(self):
        return self.last_f - self.first_f + 1


class Journey:
    """One person's whole visit."""

    __slots__ = ("id", "first_f", "last_f", "present", "still", "walked_px",
                 "segments", "stops", "_foot", "_run", "_pending", "_pending_n",
                 "_pending_still")

    def __init__(self, track_id, f):
        self.id = track_id
        self.first_f = self.last_f = f
        self.present = self.still = self.stops = 0
        self.walked_px = 0.0
        self.segments = []
        self._foot = None
        self._run = 0               # current run of standing frames
        self._pending = None        # region change waiting to be believed
        self._pending_n = self._pending_still = 0

    @property
    def path(self):
        return [s.region for s in self.segments]


class Journeys:
    """Feed it every tracked person every frame; ask it questions at the end."""

    def __init__(self, fps, region_of, params: AnalyticsParams | None = None):
        self.fps = float(fps)
        self._region_of = region_of
        self.p = params or AnalyticsParams()
        self.people: dict[int, Journey] = {}
        self._together = Counter()      # (a, b) -> frames close enough to be together
        self._overlap = Counter()       # (a, b) -> frames both were on screen

    # --- accumulation -----------------------------------------------------
    def observe(self, f, entries):
        """One frame. `entries` is [(track_id, (u, v), width, standing), ...]."""
        for track_id, foot, _w, standing in entries:
            self._observe_one(f, track_id, foot, bool(standing))
        self._observe_pairs(entries)

    def _observe_one(self, f, track_id, foot, standing):
        p = self.p
        j = self.people.get(track_id)
        if j is None:
            j = self.people[track_id] = Journey(track_id, f)
        j.last_f = f
        j.present += 1

        if j._foot is not None and not standing:
            step = ((foot[0] - j._foot[0]) ** 2 + (foot[1] - j._foot[1]) ** 2) ** 0.5
            # Below the jitter floor it is the box breathing, not a person walking.
            # Counting it would give a shopper who never moved a 300 px "journey".
            if step >= p.walk_min_px:
                j.walked_px += step
        j._foot = foot

        # A stop is a *run* of standing frames, so a browser who shifts their
        # weight once is one stop, not two.
        if standing:
            j.still += 1
            j._run += 1
            if j._run == int(round(p.stop_s * self.fps)):
                j.stops += 1
        else:
            j._run = 0

        self._advance_region(j, f, self._region_of(*foot), standing)

    def _advance_region(self, j, f, region, standing):
        """Region changes are debounced: someone on a boundary flickers at frame
        rate, and an undebounced path reads A>B>A>B>A instead of A>B."""
        cur = j.segments[-1] if j.segments else None
        if cur is not None and region == cur.region:
            cur.last_f = f
            cur.still += int(standing)
            j._pending, j._pending_n, j._pending_still = None, 0, 0
            return
        if region != j._pending:
            j._pending, j._pending_n, j._pending_still = region, 1, int(standing)
            return
        j._pending_n += 1
        j._pending_still += int(standing)
        if j._pending_n < int(round(self.p.region_debounce_s * self.fps)):
            return
        # The debounce is a delay in *believing* the change, not lost time: the
        # frames it waited belong to the segment, standing ones included, or a
        # 4 s stop is reported as 3.6 s.
        seg = Segment(region, f - j._pending_n + 1)
        seg.last_f = f
        seg.still = j._pending_still
        j.segments.append(seg)
        j._pending, j._pending_n, j._pending_still = None, 0, 0

    def _observe_pairs(self, entries):
        for i in range(len(entries)):
            for k in range(i + 1, len(entries)):
                (a, fa, wa, _), (b, fb, wb, _) = entries[i], entries[k]
                key = (min(a, b), max(a, b))
                self._overlap[key] += 1
                # Scale-adaptive: "a body width apart" means what it says whether
                # they are near the camera or far from it.
                near = self.p.group_dist_scale * max((wa + wb) / 2.0, 1.0)
                if ((fa[0] - fb[0]) ** 2 + (fa[1] - fb[1]) ** 2) ** 0.5 <= near:
                    self._together[key] += 1

    # --- 1. engagement: who passed, who stopped ---------------------------
    def engagement_rows(self):
        """Per region: passed, stopped, stop rate, mean dwell of those who stopped.

        The retail KPI a per-frame detector cannot produce: it needs the same
        identity seen walking past *and* standing still.
        """
        stop_f = self.p.stop_s * self.fps
        min_f = self.p.min_region_s * self.fps
        passed, stopped, dwell = Counter(), Counter(), defaultdict(list)
        for j in self.people.values():
            seen, stood = set(), set()
            per_region_still = Counter()
            for s in j.segments:
                if s.frames >= min_f:
                    seen.add(s.region)
                per_region_still[s.region] += s.still
            for region, still in per_region_still.items():
                if still >= stop_f:
                    stood.add(region)
                    dwell[region].append(still / self.fps)
            for region in seen | stood:
                passed[region] += 1
            for region in stood:
                stopped[region] += 1
        # Sorted by who actually stopped, not who walked past. The busiest area is
        # usually the doorway; the useful one is where people stayed.
        rows = []
        for region in sorted(passed, key=lambda r: (-stopped[r], -sum(dwell.get(r, [])), r)):
            n, k = passed[region], stopped[region]
            d = dwell.get(region, [])
            rows.append({"region": region, "passed": n, "stopped": k,
                         "stop_rate_pct": round(100.0 * k / n, 1) if n else 0.0,
                         "mean_stop_s": round(sum(d) / len(d), 1) if d else 0.0,
                         "total_stop_s": round(sum(d), 1)})
        return rows

    # --- 2. flow ----------------------------------------------------------
    def transition_rows(self):
        """Region -> region moves, aggregated. "70% turn left on entry"."""
        moves = Counter()
        for j in self.people.values():
            path = j.path
            for a, b in zip(path, path[1:]):
                moves[(a, b)] += 1
        total = sum(moves.values())
        return [{"from_region": a, "to_region": b, "moves": n,
                 "share_pct": round(100.0 * n / total, 1) if total else 0.0}
                for (a, b), n in moves.most_common()]

    # --- 3. shopper type --------------------------------------------------
    def shopper_type(self, j):
        """browsing / considered / direct / passing, by how many times they stopped.

        Stops, not time standing: someone who studies one display for a minute
        considered *one* thing, which is a different shopper from one who sampled
        four. Both are useful to a store and the distinction is lost if long
        dwell alone counts as browsing.
        """
        if j.stops >= 2:
            return "browsing"
        if j.stops == 1:
            return "considered"
        if j.present / self.fps < self.p.passing_s:
            return "passing"
        return "direct"

    def person_rows(self):
        rows = []
        for j in sorted(self.people.values(), key=lambda j: -j.present):
            walking_s = max((j.present - j.still) / self.fps, 1e-9)
            path = j.path
            shown = path if len(path) <= self.p.max_path_segments else \
                path[: self.p.max_path_segments] + ["..."]
            rows.append({"person": j.id,
                         "shopper_type": self.shopper_type(j),
                         "present_s": round(j.present / self.fps, 2),
                         "standing_still_s": round(j.still / self.fps, 2),
                         "stops": j.stops,
                         "walked_px": round(j.walked_px),
                         "mean_walking_speed_px_s": round(j.walked_px / walking_s, 1),
                         "regions_visited": len(path),
                         "path": " > ".join(shown)})
        return rows

    def type_counts(self):
        return Counter(self.shopper_type(j) for j in self.people.values())

    # --- 4. groups --------------------------------------------------------
    def group_rows(self):
        """People who moved around together: a couple, a family, colleagues.

        Shoppers are not the same as shopping *parties*, and a party of three
        buys once. Pairs are merged transitively, so A-with-B and B-with-C is one
        group of three.
        """
        p = self.p
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        pairs = []
        for key, n in self._together.items():
            shared = self._overlap[key]
            if n >= p.group_min_s * self.fps and shared and n / shared >= p.group_min_frac:
                pairs.append((key, n))
                union(*key)
        if not pairs:
            return []
        seconds = defaultdict(float)
        members = defaultdict(set)
        for (a, b), n in pairs:
            root = find(a)
            members[root] |= {a, b}
            seconds[root] = max(seconds[root], n / self.fps)
        rows = []
        for i, (root, who) in enumerate(sorted(members.items(),
                                               key=lambda kv: -len(kv[1])), start=1):
            rows.append({"group": i, "size": len(who),
                         "members": " ".join(f"#{m}" for m in sorted(who)),
                         "together_s": round(seconds[root], 1)})
        return rows
