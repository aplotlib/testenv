from datetime import datetime
import pytest
from timeline.data import EVENTS, PHASES, COLORS, MARKERS

REQUIRED_FIELDS = {'date', 'label', 'type', 'note', 'positive', 'phase'}
VALID_TYPES = {'detection', 'rca', 'decision', 'containment'}
VALID_PHASES = {'early', 'investigation', 'decisions', 'containment'}


def test_all_events_have_required_fields():
    for i, event in enumerate(EVENTS):
        missing = REQUIRED_FIELDS - set(event.keys())
        assert not missing, f"Event {i} missing fields: {missing}"


def test_all_event_types_are_valid():
    for i, event in enumerate(EVENTS):
        assert event['type'] in VALID_TYPES, \
            f"Event {i} ('{event['label'][:20]}') has invalid type: {event['type']}"


def test_all_event_phases_are_valid():
    for i, event in enumerate(EVENTS):
        assert event['phase'] in VALID_PHASES, \
            f"Event {i} has invalid phase: {event['phase']}"


def test_all_dates_are_datetime():
    for i, event in enumerate(EVENTS):
        assert isinstance(event['date'], datetime), \
            f"Event {i} date is not a datetime: {type(event['date'])}"


def test_events_cover_expected_date_range():
    dates = [e['date'] for e in EVENTS]
    assert min(dates) >= datetime(2025, 5, 1)
    assert max(dates) <= datetime(2026, 6, 30)


def test_positive_events_have_notes():
    for i, event in enumerate(EVENTS):
        if event['positive']:
            assert event['note'] and event['note'].strip(), \
                f"Positive event {i} must have a note (it gets a star badge)"


def test_seventeen_events():
    assert len(EVENTS) == 17, f"Expected 17 events, got {len(EVENTS)}"


def test_colors_has_required_keys():
    required = {
        'detection', 'rca', 'decision', 'containment',
        'positive_badge', 'spine', 'text', 'background',
    }
    missing = required - set(COLORS.keys())
    assert not missing, f"COLORS missing keys: {missing}"


def test_markers_has_required_keys():
    assert set(MARKERS.keys()) == {'detection', 'rca', 'decision', 'containment'}


def test_four_phases_defined():
    assert len(PHASES) == 4
    assert {p['key'] for p in PHASES} == VALID_PHASES


def test_each_phase_has_at_least_one_event():
    for phase in PHASES:
        events_in_phase = [e for e in EVENTS if e['phase'] == phase['key']]
        assert events_in_phase, f"Phase '{phase['key']}' has no events"
