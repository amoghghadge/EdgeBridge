"""
generate_calendar_dataset.py

Generates a JSONL training dataset for fine-tuning FunctionGemma-270M
on multi-step calendar tool-calling operations with dynamic temporal grounding.

Each example contains:
  - tools: The calendar function schemas
  - messages: A conversation with a dynamically generated system prompt, 
    user query, and the expected assistant response (including tool_calls 
    and tool results mapped to the simulated current date).

Usage:
  python generate_calendar_dataset.py
"""

import json
import random
from datetime import datetime, timedelta

def tool_msg(name, content):
    """Create a properly formatted tool response message."""
    return {"role": "tool", "name": name, "content": content}

# ============================================================================
# Calendar Tool Schemas
# ============================================================================

CALENDAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": "Returns all calendar events on a specific date, including titles, times, locations, notes, and attendees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check. Accepts: 'today', 'tomorrow', 'yesterday', or YYYY-MM-DD format."
                    }
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_week_events",
            "description": "Returns a weekly overview of all calendar events grouped by day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "The first day of the week to show. Defaults to 'today'."
                    }
                },
                "required": ["start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_free_slots",
            "description": "Finds available time slots of a given duration on a specific date between 8 AM and 8 PM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to search for free slots."
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Minimum duration in minutes needed for the free slot."
                    }
                },
                "required": ["date", "duration_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Creates a new event on the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the event."},
                    "start_time": {"type": "string", "description": "Start time in YYYY-MM-DDTHH:MM format."},
                    "end_time": {"type": "string", "description": "End time in YYYY-MM-DDTHH:MM format."},
                    "location": {"type": "string", "description": "Optional location."},
                    "notes": {"type": "string", "description": "Optional notes."},
                    "calendar_name": {"type": "string", "description": "Optional calendar name."}
                },
                "required": ["title", "start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_event",
            "description": "Modifies an existing calendar event. Can change title, time, or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to modify."},
                    "new_title": {"type": "string", "description": "Optional new title."},
                    "new_start_time": {"type": "string", "description": "Optional new start time."},
                    "new_end_time": {"type": "string", "description": "Optional new end time."},
                    "new_location": {"type": "string", "description": "Optional new location."}
                },
                "required": ["event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Deletes a calendar event by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to delete."}
                },
                "required": ["event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Searches for events by keyword in title, location, or notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword."},
                    "days_ahead": {"type": "integer", "description": "How many days ahead to search."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_conflicts",
            "description": "Checks if a proposed time slot conflicts with existing events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Proposed start time."},
                    "end_time": {"type": "string", "description": "Proposed end time."}
                },
                "required": ["start_time", "end_time"]
            }
        }
    }
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
    "Networking Event", "Thesis Meeting", "Research Seminar"
]

LOCATIONS = [
    "Thornton Hall A120", "Olsson Hall 018", "Rice Hall 340",
    "Zoom", "Conference Room B", "Main Library", "Student Center",
    "virtual", "TBD", "Newcomb Hall", "Google Meet",
    "Building 5 Room 201", "Coffee Shop on Main St"
]

NOTES = [
    "Discuss project updates", "Bring laptop", "Spec Top: Computer Science",
    "Quarterly review", "Prepare slides beforehand", "Cancel if raining",
    "Review PR #42 before meeting", "Agenda: roadmap planning",
    "Important: bring signed forms", "Optional attendance"
]

CALENDARS = ["Work", "Personal", "School", "Health"]

# ============================================================================
# Core Temporal Helper Functions
# ============================================================================

def random_simulated_datetime():
    """Generates a random mock 'current' date and time between 2025 and 2027."""
    start = datetime(2025, 1, 1)
    end = datetime(2027, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_hours = random.randint(8, 18)
    random_minutes = random.randint(0, 59)
    return start + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)

def get_system_prompt(simulated_now, extra_instructions=""):
    """Dynamically builds the system prompt with temporal grounding."""
    base_prompt = (
        f"Current date and time: {simulated_now.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Day of week: {simulated_now.strftime('%A')}\n"
        f"You are an intelligent calendar assistant. Use the provided tools to help manage the user's schedule."
    )
    if extra_instructions:
        base_prompt += f" {extra_instructions}"
    return base_prompt

def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def format_time(dt):
    return dt.strftime("%Y-%m-%dT%H:%M")

def get_relative_date_word_and_dt(simulated_now):
    """Returns a tuple of (user_friendly_word, actual_datetime_object)."""
    offset = random.choice([-1, 0, 1, random.randint(2, 7)])
    target_dt = simulated_now + timedelta(days=offset)
    
    if offset == 0:
        word = "today"
    elif offset == 1:
        word = "tomorrow"
    elif offset == -1:
        word = "yesterday"
    else:
        word = format_date(target_dt)
        
    return word, target_dt

def random_event(target_dt, idx=0):
    """Generates a random event structurally bound to a specific target date."""
    hour = random.choice([8, 9, 10, 11, 13, 14, 15, 16])
    duration = random.choice([30, 45, 60, 75, 90, 120])
    start = target_dt.replace(hour=hour, minute=0)
    end = start + timedelta(minutes=duration)
    title = random.choice(EVENT_TITLES)
    return {
        "title": title,
        "start_time": start.strftime("%I:%M %p"),
        "end_time": end.strftime("%I:%M %p"),
        "date": target_dt.strftime("%b %d, %Y"),
        "location": random.choice(LOCATIONS) if random.random() > 0.3 else "",
        "notes": random.choice(NOTES) if random.random() > 0.5 else "",
        "calendar": random.choice(CALENDARS),
        "event_id": f"EVT-{idx:04d}-{random.randint(1000,9999)}",
        "duration_minutes": duration,
        "all_day": False,
        "_start_dt": start,
        "_end_dt": end,
    }

def events_to_tool_result(events, tool_name="get_events", date_str="today"):
    """Format events as a tool response JSON string."""
    clean_events = []
    for e in events:
        d = {"title": e["title"], "start_time": e["start_time"],
             "end_time": e["end_time"], "calendar": e["calendar"],
             "event_id": e["event_id"]}
        if e.get("location"):
            d["location"] = e["location"]
        if e.get("notes"):
            d["notes"] = e["notes"]
        d["duration_minutes"] = e["duration_minutes"]
        d["all_day"] = False
        clean_events.append(d)
    return json.dumps({
        "tool_name": tool_name,
        "date": date_str,
        "event_count": len(clean_events),
        "events": clean_events
    })

# ============================================================================
# Generators
# ============================================================================

def gen_single_get_events():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    num_events = random.randint(0, 4)
    events = [random_event(target_dt, i) for i in range(num_events)]
    
    user_prompts = [
        f"What's on my calendar {date_word}?",
        f"What events do I have {date_word}?",
        f"Show me my schedule for {date_word}",
        f"Do I have anything planned {date_word}?",
    ]
    
    tool_result = events_to_tool_result(events, "get_events", target_dt.strftime("%b %d, %Y"))
    
    if num_events == 0:
        assistant_text = f"You have no events scheduled for {date_word}. Your day is completely free!"
    else:
        lines = [f"You have {num_events} event{'s' if num_events > 1 else ''} {date_word}:\n"]
        for i, e in enumerate(events, 1):
            line = f"{i}. **{e['title']}** at {e['start_time']} - {e['end_time']}"
            if e.get("location"):
                line += f" at {e['location']}"
            lines.append(line)
        assistant_text = "\n".join(lines)
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": random.choice(user_prompts)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_events", "arguments": {"date": date_word}}}
            ]},
            tool_msg("get_events", tool_result),
            {"role": "assistant", "content": assistant_text}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_single_create_event():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt)
    
    user_prompts = [
        f"Schedule a {event['title']} {date_word} at {event['_start_dt'].strftime('%I:%M %p')}",
        f"Create an event called {event['title']} {date_word} from {event['start_time']} to {event['end_time']}",
        f"Add {event['title']} to my calendar for {date_word} at {event['_start_dt'].strftime('%I:%M %p')}",
    ]
    
    create_args = {
        "title": event["title"],
        "start_time": format_time(event["_start_dt"]),
        "end_time": format_time(event["_end_dt"]),
    }
    if event.get("location") and random.random() > 0.5:
        create_args["location"] = event["location"]
    
    create_result = json.dumps({
        "tool_name": "create_event", "success": True,
        "title": event["title"], "start_time": event["start_time"],
        "end_time": event["end_time"], "event_id": event["event_id"]
    })
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": random.choice(user_prompts)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": create_args}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": f"Done! I've created \"{event['title']}\" on {target_dt.strftime('%b %d')} from {event['start_time']} to {event['end_time']}."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_find_free_and_book():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    duration = random.choice([30, 60, 90])
    title = random.choice(EVENT_TITLES)
    
    free_start = target_dt.replace(hour=random.choice([8, 12, 15]), minute=0)
    free_end = free_start + timedelta(minutes=duration + 60)
    
    free_slots_result = json.dumps({
        "tool_name": "find_free_slots", "date": target_dt.strftime("%b %d, %Y"),
        "duration_requested_minutes": duration, "free_slots_count": 1,
        "free_slots": [
            {"start": free_start.strftime("%I:%M %p"), "end": free_end.strftime("%I:%M %p"), "duration_minutes": str(duration + 60)}
        ]
    })
    
    book_start = free_start
    book_end = free_start + timedelta(minutes=duration)
    
    create_result = json.dumps({
        "tool_name": "create_event", "success": True, "title": title,
        "start_time": book_start.strftime("%I:%M %p"), "end_time": book_end.strftime("%I:%M %p"),
        "event_id": f"EVT-NEW-{random.randint(1000,9999)}"
    })
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Find a {duration}-minute slot {date_word} and book a {title}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "find_free_slots", "arguments": {"date": date_word, "duration_minutes": duration}}}
            ]},
            tool_msg("find_free_slots", free_slots_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": {
                    "title": title, "start_time": format_time(book_start), "end_time": format_time(book_end)
                }}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": f"I found a free slot and booked \"{title}\" {date_word} from {book_start.strftime('%I:%M %p')} to {book_end.strftime('%I:%M %p')}."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_duplicate_events():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    source_word, source_dt = get_relative_date_word_and_dt(simulated_now)
    # Ensure target is different from source
    target_dt = source_dt + timedelta(days=random.randint(1, 4))
    
    # Calculate if target is naturally tomorrow relative to sim_now
    if target_dt.date() == (simulated_now + timedelta(days=1)).date():
        target_word = "tomorrow"
    else:
        target_word = format_date(target_dt)

    events = [random_event(source_dt, i) for i in range(random.randint(1, 3))]
    get_result = events_to_tool_result(events, "get_events", source_dt.strftime("%b %d, %Y"))
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Copy all my events from {source_word} to {target_word}"},
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
            "title": event["title"], "start_time": format_time(new_start), "end_time": format_time(new_end),
        }
        create_result = json.dumps({"tool_name": "create_event", "success": True, "title": event["title"], "event_id": f"EVT-COPY-{random.randint(1000,9999)}"})
        
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "create_event", "arguments": create_args}}
        ]})
        messages.append(tool_msg("create_event", create_result))
        created_titles.append(event["title"])
    
    messages.append({
        "role": "assistant",
        "content": f"Done! I've copied {len(events)} events to {target_dt.strftime('%b %d')}: {', '.join(created_titles)}."
    })
    
    return {"tools": CALENDAR_TOOLS, "messages": messages, "metadata": random.choice(["train"] * 9 + ["eval"])}

def gen_search_and_modify():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    _, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt, 0)
    query = event["title"].split()[0]
    
    search_result = json.dumps({
        "tool_name": "search_events", "query": query, "match_count": 1, "events": [{
            "title": event["title"], "start_time": event["start_time"], "end_time": event["end_time"],
            "date": event["date"], "location": event.get("location", ""), "event_id": event["event_id"],
        }]
    })
    
    new_location = random.choice(LOCATIONS)
    modify_result = json.dumps({"tool_name": "modify_event", "success": True, "original_title": event["title"]})
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Search for {event['title']} and move it to {new_location}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_events", "arguments": {"query": query, "days_ahead": 30}}}
            ]},
            tool_msg("search_events", search_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "modify_event", "arguments": {
                    "event_id": event["event_id"], "new_location": new_location
                }}}
            ]},
            tool_msg("modify_event", modify_result),
            {"role": "assistant", "content": f"Updated! \"{event['title']}\" is now located at {new_location}."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_check_conflicts_then_create():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now, extra_instructions="Always check for conflicts before creating events.")
    
    date_word, target_dt = get_relative_date_word_and_dt(simulated_now)
    title = random.choice(EVENT_TITLES)
    start = target_dt.replace(hour=14, minute=0)
    end = start + timedelta(hours=1)
    
    has_conflict = random.choice([True, False])
    conflict_result = json.dumps({
        "tool_name": "check_conflicts", "has_conflicts": has_conflict,
        "conflicting_events": [{"title": "Existing Appt"}] if has_conflict else []
    })
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Create a {title} {date_word} at {start.strftime('%I:%M %p')}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "check_conflicts", "arguments": {
                "start_time": format_time(start), "end_time": format_time(end)
            }}}
        ]},
        tool_msg("check_conflicts", conflict_result),
    ]
    
    if has_conflict:
        messages.append({"role": "assistant", "content": f"There's a scheduling conflict at {start.strftime('%I:%M %p')}. Would you like me to find an alternative time?"})
    else:
        create_result = json.dumps({"tool_name": "create_event", "success": True, "event_id": "EVT-NEW-1234"})
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "create_event", "arguments": {
                    "title": title, "start_time": format_time(start), "end_time": format_time(end)
                }}}
            ]},
            tool_msg("create_event", create_result),
            {"role": "assistant", "content": f"No conflicts found. Created \"{title}\"."}
        ])
    
    return {"tools": CALENDAR_TOOLS, "messages": messages, "metadata": random.choice(["train"] * 9 + ["eval"])}

def gen_reschedule_event():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    _, target_dt = get_relative_date_word_and_dt(simulated_now)
    event = random_event(target_dt, 0)
    new_start = target_dt.replace(hour=16, minute=0)
    new_end = new_start + timedelta(minutes=event["duration_minutes"])
    
    search_result = json.dumps({
        "tool_name": "search_events", "query": event["title"], "events": [{
            "title": event["title"], "start_time": event["start_time"], "end_time": event["end_time"], "event_id": event["event_id"]
        }]
    })
    
    modify_result = json.dumps({"tool_name": "modify_event", "success": True})
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Move my {event['title']} to {new_start.strftime('%I:%M %p')}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "search_events", "arguments": {"query": event["title"]}}}
            ]},
            tool_msg("search_events", search_result),
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "modify_event", "arguments": {
                    "event_id": event["event_id"], "new_start_time": format_time(new_start), "new_end_time": format_time(new_end)
                }}}
            ]},
            tool_msg("modify_event", modify_result),
            {"role": "assistant", "content": f"Done! \"{event['title']}\" has been moved."}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_weekly_overview():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now)
    
    day1 = simulated_now
    day2 = simulated_now + timedelta(days=1)
    
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "What does my week look like?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_week_events", "arguments": {"start_date": "today"}}}
            ]},
            tool_msg("get_week_events", json.dumps({
                "tool_name": "get_week_events", "total_events": 2,
                "schedule": [
                    {"date": day1.strftime("%b %-d, %Y"), "day_of_week": day1.strftime("%A"), "events": [{"title": "CS 4501", "start_time": "11:00 AM", "end_time": "12:15 PM"}]},
                    {"date": day2.strftime("%b %-d, %Y"), "day_of_week": day2.strftime("%A"), "events": [{"title": "Lab Session", "start_time": "2:00 PM", "end_time": "3:30 PM"}]}
                ]
            })),
            {"role": "assistant", "content": f"Here's your week:\n\n**{day1.strftime('%A, %b %-d')}:** CS 4501 at 11:00 AM\n**{day2.strftime('%A, %b %-d')}:** Lab Session at 2:00 PM"}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

def gen_no_tool_needed():
    simulated_now = random_simulated_datetime()
    sys_prompt = get_system_prompt(simulated_now, extra_instructions="For non-calendar questions, respond helpfully without using tools.")
    
    qa_pairs = [
        ("What's the weather like?", "I'm a calendar assistant, so I can't check the weather. But I can help you manage your schedule!"),
        ("Thank you!", "You're welcome! Let me know if you need anything else with your calendar."),
    ]
    q, a = random.choice(qa_pairs)
    return {
        "tools": CALENDAR_TOOLS,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ],
        "metadata": random.choice(["train"] * 9 + ["eval"])
    }

# ============================================================================
# Generation Execution
# ============================================================================

GENERATORS = [
    (gen_single_get_events, 200),
    (gen_single_create_event, 150),
    (gen_find_free_and_book, 200),
    (gen_duplicate_events, 150),
    (gen_search_and_modify, 150),
    (gen_check_conflicts_then_create, 100),
    (gen_reschedule_event, 150),
    (gen_weekly_overview, 80),
    (gen_no_tool_needed, 80),
]

def generate_dataset(output_path="calendar_training_data.jsonl"):
    examples = []
    for generator, count in GENERATORS:
        for _ in range(count):
            try:
                example = generator()
                examples.append(example)
            except Exception as e:
                print(f"Error in {generator.__name__}: {e}")
    
    random.shuffle(examples)
    
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    train_count = sum(1 for e in examples if e.get("metadata") == "train")
    eval_count = sum(1 for e in examples if e.get("metadata") == "eval")
    
    print(f"Generated {len(examples)} examples")
    print(f"  Train: {train_count}")
    print(f"  Eval:  {eval_count}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    random.seed(42)
    generate_dataset()