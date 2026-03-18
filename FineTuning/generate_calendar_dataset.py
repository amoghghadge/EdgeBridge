"""
generate_calendar_dataset.py  (v4 — Final Production Version)

CRITICAL FIXES OVER v3:
1. FORMAT MATCH: Training data now matches the EXACT runtime token format.
   - No system message (Jinja template defaults to "You are Qwen...")
   - Calendar context wrapped in [Context: ...] and prepended to user message
   - This matches what LiteRT-LM actually produces at inference time.
2. FULL PARAMETER COVERAGE: create_event now trains on location, notes,
   calendar_name in all combinations. modify_event trains on title, time,
   and location changes. search_events uses varying days_ahead.
3. PROMPT DIVERSITY: 8-10 user prompt variants per generator (was 3-4).
4. RICHER TOOL RESULTS: find_free_slots returns multiple slots.
"""

import json
import random
from datetime import datetime, timedelta

# ============================================================================
# Calendar Tool Schemas (1:1 with ToolDeclarations.swift)
# ============================================================================

CALENDAR_TOOLS = [
    {"type": "function", "function": {
        "name": "get_events",
        "description": "Returns all calendar events on a specific date, including their titles, times, locations, notes, and attendees. Accepts 'today', 'tomorrow', 'yesterday', 'next monday', or a specific date in YYYY-MM-DD format. ALWAYS use this tool when the user asks about their schedule on any day. The results include full event details like location and notes.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "The date to check. Accepts: 'today', 'tomorrow', 'yesterday', 'next monday' through 'next sunday', or YYYY-MM-DD format. Defaults to 'today' if not specified."}
        }, "required": ["date"]}
    }},
    {"type": "function", "function": {
        "name": "get_week_events",
        "description": "Returns a weekly overview of all calendar events, grouped by day. Use this when the user asks about their week, weekly schedule, or wants to see multiple days at once.",
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "The first day of the week to show. Accepts 'today', 'tomorrow', 'next monday', or YYYY-MM-DD. Defaults to 'today'."}
        }, "required": ["start_date"]}
    }},
    {"type": "function", "function": {
        "name": "find_free_slots",
        "description": "Finds available time slots of a given duration on a specific date, between 8:00 AM and 8:00 PM. Use this when the user wants to know when they're free, asks about availability, or wants to find time for a meeting.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "The date to search for free slots. Accepts 'today', 'tomorrow', or YYYY-MM-DD."},
            "duration_minutes": {"type": "integer", "description": "The minimum duration in minutes needed (e.g., 30, 60, 90, 120)."}
        }, "required": ["date", "duration_minutes"]}
    }},
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Creates a new event on the user's calendar. Use this when the user explicitly asks to schedule, book, or create an event. You can include a location and notes.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "The title of the event."},
            "start_time": {"type": "string", "description": "Start time in YYYY-MM-DDTHH:MM format (e.g., 2026-03-06T14:00)."},
            "end_time": {"type": "string", "description": "End time in YYYY-MM-DDTHH:MM format (e.g., 2026-03-06T15:00)."},
            "location": {"type": "string", "description": "Optional: the location for the event."},
            "notes": {"type": "string", "description": "Optional: notes or description for the event."},
            "calendar_name": {"type": "string", "description": "Optional: specific calendar to add to. Uses default if omitted."}
        }, "required": ["title", "start_time", "end_time"]}
    }},
    {"type": "function", "function": {
        "name": "modify_event",
        "description": "Modifies an existing calendar event. Can change the title, time, or location. Use this when the user wants to reschedule, rename, or update an event. You need the event_id from a previous get_events or search_events call.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string", "description": "The unique identifier of the event to modify (from get_events results)."},
            "new_title": {"type": "string", "description": "Optional: new title for the event."},
            "new_start_time": {"type": "string", "description": "Optional: new start time in YYYY-MM-DDTHH:MM format."},
            "new_end_time": {"type": "string", "description": "Optional: new end time in YYYY-MM-DDTHH:MM format."},
            "new_location": {"type": "string", "description": "Optional: new location for the event."}
        }, "required": ["event_id"]}
    }},
    {"type": "function", "function": {
        "name": "delete_event",
        "description": "Deletes a calendar event. Only use when the user explicitly asks to remove or cancel an event. Requires the event_id from a previous query. Always confirm with the user before deleting.",
        "parameters": {"type": "object", "properties": {
            "event_id": {"type": "string", "description": "The unique identifier of the event to delete."}
        }, "required": ["event_id"]}
    }},
    {"type": "function", "function": {
        "name": "search_events",
        "description": "Searches for calendar events by keyword in the title, location, or notes. Searches across the next N days. Use this when the user asks about a specific event by name or topic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search keyword to look for in event titles, locations, and notes."},
            "days_ahead": {"type": "integer", "description": "How many days ahead to search (default: 30)."}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "check_conflicts",
        "description": "Checks if a proposed time slot conflicts with any existing events. Use this before creating an event to ensure there are no scheduling overlaps.",
        "parameters": {"type": "object", "properties": {
            "start_time": {"type": "string", "description": "Proposed start time in YYYY-MM-DDTHH:MM format."},
            "end_time": {"type": "string", "description": "Proposed end time in YYYY-MM-DDTHH:MM format."}
        }, "required": ["start_time", "end_time"]}
    }}
]

# ============================================================================
# Sample Data Pools
# ============================================================================

EVENT_TITLES = [
    "Team Standup", "CS 4501", "Lunch with Sarah", "Project Review",
    "1:1 with Manager", "Design Sprint", "CS 4720", "Gym",
    "Dentist Appointment", "Coffee with Alex", "Board Meeting",
    "Study Group", "Yoga", "Interview Prep", "Client Call",
    "Budget Review", "Lab Session", "Office Hours", "Presentation Prep",
    "Team Happy Hour", "Sprint Planning", "Code Review", "Workshop",
    "Networking Event", "Thesis Meeting", "Research Seminar",
    "Doctor Appointment", "Piano Lesson", "Running Club",
    "Career Fair", "TA Office Hours", "Group Project Meeting"
]

LOCATIONS = [
    "Thornton Hall A120", "Olsson Hall 018", "Rice Hall 340",
    "Zoom", "Conference Room B", "Main Library", "Student Center",
    "virtual", "TBD", "Newcomb Hall", "Google Meet",
    "Building 5 Room 201", "Coffee Shop on Main St",
    "Downtown Fitness", "Alderman Library Room 304", "Clark Hall 108",
    "The Corner", "Bodo's Bagels"
]

NOTES = [
    "Discuss project updates", "Bring laptop", "Spec Top: Computer Science",
    "Quarterly review", "Prepare slides beforehand", "Cancel if raining",
    "Review PR #42 before meeting", "Agenda: roadmap planning",
    "Important: bring signed forms", "Optional attendance",
    "Bring charger", "Wear business casual", "RSVP required",
    "Zoom link in email", "Prepare demo for client",
    "Mobile Application Development", "Read chapters 5-7 before class"
]

CALENDARS = ["Work", "Personal", "School", "Health"]

# ============================================================================
# Formatting Functions — Tool args use ISO, display uses human-readable
# ============================================================================

def iso_time(dt):
    """For TOOL CALL ARGUMENTS — matches Swift's parseDateTime()."""
    return dt.strftime("%Y-%m-%dT%H:%M")

def display_time(dt):
    """For HUMAN-READABLE text in assistant responses and tool results."""
    return dt.strftime("%-I:%M %p")

def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def display_date(dt):
    return dt.strftime("%b %-d, %Y")

def tool_msg(name, content):
    return {"role": "tool", "name": name, "content": content}

# ============================================================================
# Temporal Helpers
# ============================================================================

def random_simulated_datetime():
    start = datetime(2025, 1, 1)
    end = datetime(2027, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_hours = random.randint(8, 18)
    random_minutes = random.choice([0, 15, 30, 45])
    return start + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)

def get_context_block(simulated_now):
    """
    CRITICAL: This generates the [Context: ...] block that gets PREPENDED
    to the user message, matching the exact runtime format from EngineViewModel.
    """
    return (
        f"[Context: Current date and time: {simulated_now.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Day of week: {simulated_now.strftime('%A')}\n"
        "You are an intelligent calendar assistant running entirely on this device. "
        "You have access to the user's real calendar through the provided tools. "
        "Important rules: "
        "- ALWAYS use the get_events tool to check the calendar before answering schedule questions. Never guess. "
        "- When the user asks about \"today\", call get_events with date=\"today\". "
        "- When the user asks about \"tomorrow\", call get_events with date=\"tomorrow\". "
        "- When asking about a specific day, use the appropriate date string. "
        "- Event results include location, notes, attendees, and duration — use this information in your answers. "
        "- When finding free time, use find_free_slots. "
        "- Before creating events, optionally use check_conflicts to verify no overlaps. "
        "- For weekly overviews, use get_week_events. "
        "- To find a specific event by name, use search_events. "
        "- For create_event, modify_event, and check_conflicts, use YYYY-MM-DDTHH:MM format for times. "
        "- Format times naturally (e.g., \"2:00 PM\" not \"14:00\") when speaking to the user. "
        "- Be concise but thorough. Include location and notes when they exist. "
        "- For non-calendar questions, respond normally without tools.]"
    )

def get_relative_date_word_and_dt(simulated_now):
    """Returns (date_word_for_tool_args, target_datetime)."""
    offset = random.choice([-1, 0, 0, 1, 1, random.randint(2, 7)])  # bias toward today/tomorrow
    target_dt = simulated_now + timedelta(days=offset)
    if offset == 0:
        word = "today"
    elif offset == 1:
        word = "tomorrow"
    elif offset == -1:
        word = "yesterday"
    else:
        # Mix between YYYY-MM-DD and "next <day>" for variety
        if offset <= 7 and random.random() > 0.5:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            word = f"next {days[target_dt.weekday()]}"
        else:
            word = format_date(target_dt)
    return word, target_dt

def random_event(target_dt, idx=0):
    hour = random.choice([8, 9, 10, 11, 13, 14, 15, 16, 17])
    duration = random.choice([30, 45, 50, 60, 75, 90, 120])
    start = target_dt.replace(hour=hour, minute=random.choice([0, 15, 30]), second=0, microsecond=0)
    end = start + timedelta(minutes=duration)
    title = random.choice(EVENT_TITLES)
    has_location = random.random() > 0.25
    has_notes = random.random() > 0.4
    return {
        "title": title,
        "start_time": display_time(start),
        "end_time": display_time(end),
        "date": display_date(target_dt),
        "location": random.choice(LOCATIONS) if has_location else "",
        "notes": random.choice(NOTES) if has_notes else "",
        "calendar": random.choice(CALENDARS),
        "event_id": f"EVT-{idx:04d}-{random.randint(1000,9999)}",
        "duration_minutes": duration,
        "all_day": False,
        "recurring": random.choice([True, False]),
        "_start_dt": start,
        "_end_dt": end,
    }

def events_to_tool_result(events, tool_name="get_events", date_str="today", dt_obj=None):
    clean_events = []
    for e in events:
        d = {
            "title": e["title"], "start_time": e["start_time"],
            "end_time": e["end_time"], "calendar": e["calendar"],
            "event_id": e["event_id"], "duration_minutes": e["duration_minutes"],
            "all_day": False, "recurring": e.get("recurring", False),
            "date": e["date"],
        }
        if e.get("location"):
            d["location"] = e["location"]
        if e.get("notes"):
            d["notes"] = e["notes"]
        clean_events.append(d)
    result = {
        "tool_name": tool_name,
        "date": date_str,
        "event_count": len(clean_events),
        "events": clean_events
    }
    if dt_obj:
        result["day_of_week"] = dt_obj.strftime("%A")
    return json.dumps(result)

def user_msg(context_block, query):
    """Create user message with [Context: ...] prefix matching runtime format."""
    return {"role": "user", "content": f"{context_block}\n{query}"}

# ============================================================================
# Generators
# NO system message — the Jinja template defaults to "You are Qwen..."
# which matches the runtime behavior exactly.
# ============================================================================

def gen_single_get_events():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    num_events = random.randint(0, 4)
    events = [random_event(target_dt, i) for i in range(num_events)]

    prompts = [
        f"What's on my calendar {date_word}?",
        f"What events do I have {date_word}?",
        f"Show me my schedule for {date_word}",
        f"Do I have anything planned {date_word}?",
        f"Am I free {date_word}?",
        f"What's happening {date_word}?",
        f"Check my calendar for {date_word}",
        f"Any meetings {date_word}?",
        f"Pull up my {date_word} schedule",
        f"What do I have going on {date_word}?",
    ]

    tool_result = events_to_tool_result(events, "get_events", display_date(target_dt), target_dt)

    if num_events == 0:
        responses = [
            f"You have no events scheduled for {date_word}. Your day is completely free!",
            f"Your calendar is clear {date_word} — no events at all.",
            f"Nothing on your calendar {date_word}. You're free all day!",
        ]
        assistant_text = random.choice(responses)
    else:
        lines = [f"You have {num_events} event{'s' if num_events > 1 else ''} {date_word}:\n"]
        for i, e in enumerate(events, 1):
            line = f"{i}. **{e['title']}** at {e['start_time']} - {e['end_time']}"
            if e.get("location"):
                line += f" ({e['location']})"
            if e.get("notes"):
                line += f" — {e['notes']}"
            lines.append(line)
        assistant_text = "\n".join(lines)

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, random.choice(prompts)),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_events", "arguments": {"date": date_word}}}
            ]},
            tool_msg("get_events", tool_result),
            {"role": "assistant", "content": assistant_text}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_single_create_event():
    """Creates events with FULL parameter coverage: location, notes, calendar_name."""
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt)

    # Decide which optional params to include (weighted for full coverage)
    r = random.random()
    include_location = r > 0.3
    include_notes = r > 0.5
    include_calendar = r > 0.75

    # Build user prompt with natural phrasing of optional params
    base_prompts = [
        f"Schedule '{event['title']}' {date_word} at {display_time(event['_start_dt'])}",
        f"Create an event called '{event['title']}' {date_word} from {display_time(event['_start_dt'])} to {display_time(event['_end_dt'])}",
        f"Add '{event['title']}' to my calendar for {date_word} at {display_time(event['_start_dt'])}",
        f"Book '{event['title']}' {date_word} starting at {display_time(event['_start_dt'])} for {event['duration_minutes']} minutes",
        f"Put '{event['title']}' on my schedule {date_word} at {display_time(event['_start_dt'])}",
        f"I need to schedule '{event['title']}' {date_word} from {display_time(event['_start_dt'])} to {display_time(event['_end_dt'])}",
        f"Set up '{event['title']}' {date_word} at {display_time(event['_start_dt'])}",
        f"Can you create '{event['title']}' for {date_word} at {display_time(event['_start_dt'])}?",
    ]
    prompt = random.choice(base_prompts)

    loc = random.choice(LOCATIONS) if include_location else None
    note = random.choice(NOTES) if include_notes else None
    cal = random.choice(CALENDARS) if include_calendar else None

    if loc:
        prompt += f" in {loc}"
    if note:
        prompt += f". Note: {note}"
    if cal:
        prompt += f". Add it to my {cal} calendar"

    # Build tool call arguments
    create_args = {
        "title": event["title"],
        "start_time": iso_time(event["_start_dt"]),
        "end_time": iso_time(event["_end_dt"]),
    }
    if loc:
        create_args["location"] = loc
    if note:
        create_args["notes"] = note
    if cal:
        create_args["calendar_name"] = cal

    create_result = json.dumps({
        "tool_name": "create_event", "success": True,
        "title": event["title"],
        "start_time": display_time(event["_start_dt"]),
        "end_time": display_time(event["_end_dt"]),
        "event_id": event["event_id"]
    })

    resp = f"Done! I've created \"{event['title']}\" on {display_date(target_dt)} from {display_time(event['_start_dt'])} to {display_time(event['_end_dt'])}"
    if loc:
        resp += f" at {loc}"
    resp += "."

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, prompt),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": create_args}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": resp}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_find_free_and_book():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    duration = random.choice([30, 60, 90, 120])
    title = random.choice(EVENT_TITLES)

    # Generate 2-3 free slots (model should pick the first)
    slot_count = random.randint(1, 3)
    free_slots = []
    slot_hours = random.sample([8, 10, 12, 14, 16], k=min(slot_count, 5))
    slot_hours.sort()
    for h in slot_hours[:slot_count]:
        s = target_dt.replace(hour=h, minute=0, second=0, microsecond=0)
        e = s + timedelta(minutes=duration + random.choice([0, 30, 60]))
        free_slots.append({
            "start": display_time(s), "end": display_time(e),
            "duration_minutes": int((e - s).total_seconds() / 60)
        })

    free_slots_result = json.dumps({
        "tool_name": "find_free_slots",
        "date": display_date(target_dt),
        "duration_requested_minutes": duration,
        "search_window": "8:00 AM - 8:00 PM",
        "free_slots_count": len(free_slots),
        "free_slots": free_slots
    })

    # Book using the FIRST available slot
    first_slot_start = target_dt.replace(hour=slot_hours[0], minute=0, second=0, microsecond=0)
    book_end = first_slot_start + timedelta(minutes=duration)

    create_result = json.dumps({
        "tool_name": "create_event", "success": True, "title": title,
        "start_time": display_time(first_slot_start), "end_time": display_time(book_end),
        "event_id": f"EVT-NEW-{random.randint(1000,9999)}"
    })

    prompts = [
        f"Find a {duration}-minute slot {date_word} and book a '{title}'",
        f"Find a {duration}-minute opening {date_word} and book a '{title}' session",
        f"When am I free for {duration} minutes {date_word}? Book '{title}' in the first slot",
        f"I need {duration} minutes {date_word} for '{title}'. Find an opening and schedule it",
        f"Check my availability {date_word} for {duration} minutes and create a '{title}' event",
        f"Find me a free {duration}-minute window {date_word} and add '{title}'",
        f"Look for a {duration}-minute gap {date_word} and put '{title}' on my calendar",
        f"Is there a {duration}-minute opening {date_word}? If so, book '{title}'",
    ]

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, random.choice(prompts)),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "find_free_slots", "arguments": {"date": date_word, "duration_minutes": duration}}}
            ]},
            tool_msg("find_free_slots", free_slots_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": {
                    "title": title,
                    "start_time": iso_time(first_slot_start),
                    "end_time": iso_time(book_end)
                }}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": f"I found a free slot and booked \"{title}\" {date_word} from {display_time(first_slot_start)} to {display_time(book_end)}."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_search_and_modify():
    """Covers ALL modify_event parameter combinations: location, title, time."""
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    _, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt, 0)

    # Choose what to modify
    mod_type = random.choice(["location", "title", "time", "location_and_time", "title_and_location"])

    new_location = random.choice(LOCATIONS)
    new_title = random.choice(EVENT_TITLES)
    new_start = event["_start_dt"].replace(hour=random.choice([9, 11, 14, 16]))
    new_end = new_start + timedelta(minutes=event["duration_minutes"])
    days_ahead = random.choice([7, 14, 30, 60])

    if mod_type == "location":
        prompt = random.choice([
            f"Find my '{event['title']}' event and change the location to '{new_location}'",
            f"Update the location of '{event['title']}' to {new_location}",
            f"Move '{event['title']}' to {new_location}",
            f"Change where '{event['title']}' is happening to {new_location}",
        ])
        modify_args = {"event_id": event["event_id"], "new_location": new_location}
        resp = f"Updated! \"{event['title']}\" is now located at {new_location}."
    elif mod_type == "title":
        prompt = random.choice([
            f"Rename my '{event['title']}' event to '{new_title}'",
            f"Change the name of '{event['title']}' to '{new_title}'",
            f"Update '{event['title']}' — rename it to '{new_title}'",
        ])
        modify_args = {"event_id": event["event_id"], "new_title": new_title}
        resp = f"Done! \"{event['title']}\" has been renamed to \"{new_title}\"."
    elif mod_type == "time":
        prompt = random.choice([
            f"Reschedule '{event['title']}' to {display_time(new_start)}",
            f"Move '{event['title']}' to {display_time(new_start)}",
            f"Change the time of '{event['title']}' to {display_time(new_start)}",
            f"Push '{event['title']}' to {display_time(new_start)}",
        ])
        modify_args = {
            "event_id": event["event_id"],
            "new_start_time": iso_time(new_start),
            "new_end_time": iso_time(new_end)
        }
        resp = f"Done! \"{event['title']}\" has been moved to {display_time(new_start)}."
    elif mod_type == "location_and_time":
        prompt = f"Move '{event['title']}' to {display_time(new_start)} at {new_location}"
        modify_args = {
            "event_id": event["event_id"],
            "new_start_time": iso_time(new_start),
            "new_end_time": iso_time(new_end),
            "new_location": new_location
        }
        resp = f"Updated! \"{event['title']}\" is now at {display_time(new_start)} in {new_location}."
    else:  # title_and_location
        prompt = f"Rename '{event['title']}' to '{new_title}' and change location to {new_location}"
        modify_args = {
            "event_id": event["event_id"],
            "new_title": new_title,
            "new_location": new_location
        }
        resp = f"Updated! Event renamed to \"{new_title}\" and moved to {new_location}."

    search_result = json.dumps({
        "tool_name": "search_events", "query": event["title"], "match_count": 1,
        "events": [{
            "title": event["title"], "start_time": event["start_time"],
            "end_time": event["end_time"], "date": event["date"],
            "location": event.get("location", ""), "event_id": event["event_id"],
        }]
    })
    modify_result = json.dumps({"tool_name": "modify_event", "success": True})

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, prompt),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_events", "arguments": {"query": event["title"], "days_ahead": days_ahead}}}
            ]},
            tool_msg("search_events", search_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "modify_event", "arguments": modify_args}}
            ]},
            tool_msg("modify_event", modify_result),
            {"role": "assistant", "content": resp}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_check_conflicts_then_create():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    title = random.choice(EVENT_TITLES)
    start = target_dt.replace(hour=random.choice([9, 10, 13, 14, 15]), minute=0, second=0, microsecond=0)
    duration = random.choice([60, 90, 120])
    end = start + timedelta(minutes=duration)

    has_conflict = random.choice([True, False])
    conflict_result = json.dumps({
        "tool_name": "check_conflicts", "has_conflicts": has_conflict,
        "checked_start": display_time(start), "checked_end": display_time(end),
        "conflicting_events": [{"title": random.choice(EVENT_TITLES), "start_time": display_time(start)}] if has_conflict else []
    })

    prompts = [
        f"Create a '{title}' {date_word} at {display_time(start)} but check for conflicts first",
        f"I want to add '{title}' {date_word} at {display_time(start)}. Any conflicts?",
        f"Schedule '{title}' {date_word} from {display_time(start)} to {display_time(end)}, but make sure nothing overlaps",
        f"Check if {display_time(start)} is free {date_word} and if so, book '{title}'",
    ]

    messages = [
        user_msg(ctx, random.choice(prompts)),
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "check_conflicts", "arguments": {
                "start_time": iso_time(start), "end_time": iso_time(end)
            }}}
        ]},
        tool_msg("check_conflicts", conflict_result),
    ]

    if has_conflict:
        messages.append({"role": "assistant", "content": f"There's a scheduling conflict at {display_time(start)} {date_word}. Would you like me to find an alternative time?"})
    else:
        create_result = json.dumps({"tool_name": "create_event", "success": True, "event_id": f"EVT-NEW-{random.randint(1000,9999)}"})
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": {
                    "title": title, "start_time": iso_time(start), "end_time": iso_time(end)
                }}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": f"No conflicts found! Created \"{title}\" {date_word} at {display_time(start)}."}
        ])

    return {"tools": CALENDAR_TOOLS, "messages": messages, "metadata": random.choice(["train"] * 9 + ["eval"])}

def gen_duplicate_events():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    source_word, source_dt = get_relative_date_word_and_dt(simulated_now)
    target_dt = source_dt + timedelta(days=random.randint(1, 4))
    target_word = "tomorrow" if target_dt.date() == (simulated_now + timedelta(days=1)).date() else format_date(target_dt)

    events = [random_event(source_dt, i) for i in range(random.randint(1, 3))]
    get_result = events_to_tool_result(events, "get_events", display_date(source_dt), source_dt)

    prompts = [
        f"Copy all my events from {source_word} to {target_word}",
        f"Duplicate my {source_word} schedule to {target_word}",
        f"Clone my events from {source_word} onto {target_word}",
    ]

    messages = [
        user_msg(ctx, random.choice(prompts)),
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_events", "arguments": {"date": source_word}}}
        ]},
        tool_msg("get_events", get_result),
    ]

    created_titles = []
    for event in events:
        new_start = event["_start_dt"].replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        new_end = event["_end_dt"].replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        create_args = {
            "title": event["title"],
            "start_time": iso_time(new_start),
            "end_time": iso_time(new_end),
        }
        create_result = json.dumps({"tool_name": "create_event", "success": True, "title": event["title"], "event_id": f"EVT-COPY-{random.randint(1000,9999)}"})
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "create_event", "arguments": create_args}}
        ]})
        messages.append(tool_msg("create_event", create_result))
        created_titles.append(event["title"])

    messages.append({
        "role": "assistant",
        "content": f"Done! I've copied {len(events)} event{'s' if len(events) > 1 else ''} to {display_date(target_dt)}: {', '.join(created_titles)}."
    })

    return {"tools": CALENDAR_TOOLS, "messages": messages, "metadata": random.choice(["train"] * 9 + ["eval"])}

def gen_reschedule_event():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    _, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt, 0)
    new_hour = random.choice([9, 10, 11, 14, 15, 16])
    new_start = target_dt.replace(hour=new_hour, minute=0, second=0, microsecond=0)
    new_end = new_start + timedelta(minutes=event["duration_minutes"])

    prompts = [
        f"Move my '{event['title']}' to {display_time(new_start)}",
        f"Reschedule '{event['title']}' to {display_time(new_start)}",
        f"Push '{event['title']}' back to {display_time(new_start)}",
        f"Change the time of '{event['title']}' to {display_time(new_start)}",
    ]

    search_result = json.dumps({
        "tool_name": "search_events", "query": event["title"], "events": [{
            "title": event["title"], "start_time": event["start_time"],
            "end_time": event["end_time"], "event_id": event["event_id"]
        }]
    })
    modify_result = json.dumps({"tool_name": "modify_event", "success": True})

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, random.choice(prompts)),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_events", "arguments": {"query": event["title"]}}}
            ]},
            tool_msg("search_events", search_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "modify_event", "arguments": {
                    "event_id": event["event_id"],
                    "new_start_time": iso_time(new_start),
                    "new_end_time": iso_time(new_end)
                }}}
            ]},
            tool_msg("modify_event", modify_result),
            {"role": "assistant", "content": f"Done! \"{event['title']}\" has been moved to {display_time(new_start)}."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_delete_event():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    _, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt, 0)

    prompts = [
        f"Cancel my '{event['title']}' event",
        f"Delete '{event['title']}' from my calendar",
        f"Remove the '{event['title']}' event",
        f"I need to cancel '{event['title']}'",
        f"Take '{event['title']}' off my calendar",
        f"Get rid of '{event['title']}'",
    ]

    search_result = json.dumps({
        "tool_name": "search_events", "query": event["title"], "match_count": 1,
        "events": [{
            "title": event["title"], "start_time": event["start_time"],
            "end_time": event["end_time"], "date": event["date"],
            "event_id": event["event_id"]
        }]
    })
    delete_result = json.dumps({"tool_name": "delete_event", "success": True, "deleted_title": event["title"]})

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, random.choice(prompts)),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_events", "arguments": {"query": event["title"]}}}
            ]},
            tool_msg("search_events", search_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "delete_event", "arguments": {"event_id": event["event_id"]}}}
            ]},
            tool_msg("delete_event", delete_result),
            {"role": "assistant", "content": f"Done! I've canceled and deleted \"{event['title']}\"."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_weekly_overview():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    day1 = simulated_now.replace(second=0, microsecond=0)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=3)

    prompts = [
        "What does my week look like?",
        "Show me my weekly schedule",
        "Give me an overview of my week",
        "What's planned for this week?",
        "How busy am I this week?",
    ]

    schedule = [
        {"date": display_date(day1), "day_of_week": day1.strftime("%A"), "events": [
            {"title": random.choice(EVENT_TITLES), "start_time": "10:00 AM", "end_time": "11:00 AM"}
        ]},
        {"date": display_date(day2), "day_of_week": day2.strftime("%A"), "events": [
            {"title": random.choice(EVENT_TITLES), "start_time": "2:00 PM", "end_time": "3:30 PM"}
        ]},
        {"date": display_date(day3), "day_of_week": day3.strftime("%A"), "events": [
            {"title": random.choice(EVENT_TITLES), "start_time": "9:00 AM", "end_time": "10:00 AM"}
        ]},
    ]

    week_result = json.dumps({
        "tool_name": "get_week_events",
        "start_date": display_date(day1),
        "end_date": display_date(day1 + timedelta(days=6)),
        "total_events": 3, "days_with_events": 3,
        "schedule": schedule
    })

    lines = ["Here's your week:\n"]
    for day in schedule:
        for ev in day["events"]:
            lines.append(f"**{day['day_of_week']}, {day['date']}:** {ev['title']} at {ev['start_time']}")

    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, random.choice(prompts)),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_week_events", "arguments": {"start_date": "today"}}}
            ]},
            tool_msg("get_week_events", week_result),
            {"role": "assistant", "content": "\n".join(lines)}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_no_tool_needed():
    simulated_now = random_simulated_datetime()
    ctx = get_context_block(simulated_now)
    qa_pairs = [
        ("What's the weather like?", "I'm a calendar assistant, so I can't check the weather. But I can help you manage your schedule!"),
        ("Thank you!", "You're welcome! Let me know if you need anything else with your calendar."),
        ("What can you do?", "I can help you manage your calendar! I can check your schedule, find free time, create events, modify or delete events, and search for specific events."),
        ("Hello!", "Hi there! I'm your calendar assistant. How can I help you with your schedule today?"),
        ("Tell me a joke", "I'm focused on calendar management, but I'm happy to help you schedule some fun! Want me to check your availability?"),
        ("How does the stock market work?", "That's outside my expertise — I'm your calendar assistant. I can help with scheduling, events, and time management though!"),
        ("Goodbye", "Goodbye! Feel free to come back whenever you need help with your calendar."),
        ("Thanks for your help!", "Happy to help! Don't hesitate to ask if you need to manage your schedule again."),
    ]
    q, a = random.choice(qa_pairs)
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            user_msg(ctx, q),
            {"role": "assistant", "content": a}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

# ============================================================================
# Generation
# ============================================================================

GENERATORS = [
    (gen_single_get_events, 200),
    (gen_single_create_event, 200),     # up from 150, now covers all param combos
    (gen_find_free_and_book, 200),
    (gen_search_and_modify, 200),       # up from 150, now covers title/time/location/combos
    (gen_check_conflicts_then_create, 100),
    (gen_reschedule_event, 100),
    (gen_delete_event, 100),
    (gen_duplicate_events, 100),
    (gen_weekly_overview, 80),
    (gen_no_tool_needed, 80),
]

def generate_dataset(output_path="calendar_training_data.jsonl"):
    examples = []
    for generator, count in GENERATORS:
        for _ in range(count):
            try:
                examples.append(generator())
            except Exception as e:
                print(f"Error in {generator.__name__}: {e}")
    random.shuffle(examples)
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    train_count = sum(1 for e in examples if e.get("metadata") == "train")
    eval_count = sum(1 for e in examples if e.get("metadata") == "eval")
    multi_step = sum(1 for e in examples
                     if sum(1 for m in e["messages"] if m.get("tool_calls")) >= 2)
    print(f"Generated {len(examples)} examples")
    print(f"  Train: {train_count}")
    print(f"  Eval:  {eval_count}")
    print(f"  Multi-step: {multi_step}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    random.seed(42)
    generate_dataset()