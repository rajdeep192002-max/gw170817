"""
Test suite for multi-messenger GW170817 event timeline and coordinator (Task 012).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from gw170817.config import SimConfig
from gw170817.constants import day, Mpc
from gw170817.simulation.event_timeline import EventTimeline, MessengerEvent, EventPhase


def test_event_timeline():
    # 1. Construction
    config = SimConfig()
    timeline = EventTimeline(config)

    # 14. Validation Report Header Printout
    print("=== TASK 012 MULTI-MESSENGER EVENT TIMELINE VALIDATION ===")
    print(f"\nEvent:         {timeline.event_name}")
    print(f"Reference UTC: {timeline.reference_utc}")
    print(f"Distance:      {timeline.distance / Mpc:.1f} Mpc")
    print(f"\nMerger/GW time:  {timeline.merger_time:.3f} s")
    print(f"GRB 170817A:     {timeline.grb_time:.3f} s")
    print(f"GW -> GRB delay: {timeline.grb_delay:.3f} s")
    print(f"Afterglow peak:  {timeline.afterglow_peak_days:.1f} days")

    events = timeline.get_events()
    print("\nEvent sequence:")
    for ev in events:
        t_str = f"{ev.time:.3f} s" if ev.time < day else f"{ev.time / day:.1f} days"
        print(f"  {t_str:12s} {ev.messenger:10s} {ev.name}")

    # B. Metadata Check
    assert timeline.event_name == "GW170817"
    assert abs(timeline.distance / Mpc - 40.0) < 1e-5
    assert timeline.grb_delay == 1.7
    assert timeline.merger_time == 0.0
    assert timeline.grb_time == 1.7
    assert timeline.afterglow_peak_time == 150.0 * day

    # C. Chronological Ordering Check
    for i in range(len(events) - 1):
        assert events[i].time <= events[i+1].time, "Events must be strictly sorted by time"

    # G. Phase Classification Check
    print("\nPhase checks:")
    phases_to_check = [
        (-10.0, "INSPIRAL"),
        (0.0, "MERGER"),
        (1.0, "POST_MERGER"),
        (1.7, "GRB"),
        (150.0 * day, "AFTERGLOW")
    ]
    for t_val, expected_phase in phases_to_check:
        actual_phase = timeline.current_phase(t_val)
        print(f"  - t = {t_val:12.1f} s -> Phase: {actual_phase:12s} (Expected: {expected_phase})")
        assert actual_phase == expected_phase, f"Phase mismatch at t={t_val}: got {actual_phase}, expected {expected_phase}"

    # H. Messenger Activation Check
    assert not timeline.is_grb_active(1.0), "GRB must be inactive before 1.7 s"
    assert timeline.is_grb_active(1.7), "GRB must be active at 1.7 s"
    assert timeline.is_grb_active(2.5), "GRB must be active during 1.7-3.7 s interval"
    assert not timeline.is_grb_active(5.0), "GRB must be inactive after duration"

    assert not timeline.is_afterglow_active(100.0), "Afterglow must be inactive at early seconds"
    assert timeline.is_afterglow_active(1.0 * day), "Afterglow must be active at 1 day post-merger"
    assert timeline.is_afterglow_active(150.0 * day), "Afterglow must be active at peak"

    # I. Time Conversions Check
    assert timeline.time_since_merger(10.0) == 10.0
    assert abs(timeline.time_since_grb(2.7) - 1.0) < 1e-6
    assert abs(timeline.afterglow_time_days(150.0 * day) - 150.0) < 1e-6

    # J. events_up_to() Check
    ev_before = timeline.events_up_to(-1.0)
    assert len(ev_before) == 0, "No events before merger"

    ev_0 = timeline.events_up_to(0.0)
    assert len(ev_0) == 2, "2 events at t=0.0 (GW merger & Kilonova)"

    ev_grb = timeline.events_up_to(1.7)
    assert len(ev_grb) == 3, "3 events up to GRB (GW, Kilonova, GRB)"

    ev_all = timeline.events_up_to(200.0 * day)
    assert len(ev_all) == 4, "All 4 events included after 150 days"

    # KeyError check for get_event
    ev_gw = timeline.get_event("GW170817 merger")
    assert ev_gw.messenger == "GW"
    try:
        timeline.get_event("Nonexistent Event")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # L. Determinism & Safety Check
    t2 = EventTimeline(config)
    assert timeline.get_events() == t2.get_events()
    assert timeline.current_phase(100.0) == t2.current_phase(100.0)

    print("\nALL TASK 012 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_event_timeline()
