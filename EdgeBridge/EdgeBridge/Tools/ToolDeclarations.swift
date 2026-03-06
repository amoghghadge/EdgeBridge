//
//  ToolDeclarations.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// ToolDeclarations.swift  (v2 — Enhanced Calendar Tools)
//
// IMPROVEMENTS OVER v1:
// 1. Merged get_todays_events into get_events — accepts "today", "tomorrow",
//    or any YYYY-MM-DD date. This prevents the model from choosing the wrong
//    tool when the user asks about a different day.
// 2. Added get_week_events for weekly overview.
// 3. Added search_events for keyword search across future events.
// 4. Added check_conflicts to detect scheduling overlaps.
// 5. Added modify_event to reschedule or rename existing events.
// 6. create_event now accepts location and notes.
// 7. All tool descriptions emphasize that results include location and notes.
// 8. System prompt explicitly tells model to use get_events with "tomorrow"
//    parameter (not get_todays_events) for non-today queries.
// ============================================================================

import Foundation

enum ToolDeclarations {
    
    static let calendarToolsJSON: String = """
    [
      {
        "name": "get_events",
        "description": "Returns all calendar events on a specific date, including their titles, times, locations, notes, and attendees. Accepts 'today', 'tomorrow', 'yesterday', 'next monday', or a specific date in YYYY-MM-DD format. ALWAYS use this tool when the user asks about their schedule on any day. The results include full event details like location and notes.",
        "parameters": {
          "type": "object",
          "properties": {
            "date": {
              "type": "string",
              "description": "The date to check. Accepts: 'today', 'tomorrow', 'yesterday', 'next monday' through 'next sunday', or YYYY-MM-DD format. Defaults to 'today' if not specified."
            }
          },
          "required": ["date"]
        }
      },
      {
        "name": "get_week_events",
        "description": "Returns a weekly overview of all calendar events, grouped by day. Use this when the user asks about their week, weekly schedule, or wants to see multiple days at once.",
        "parameters": {
          "type": "object",
          "properties": {
            "start_date": {
              "type": "string",
              "description": "The first day of the week to show. Accepts 'today', 'tomorrow', 'next monday', or YYYY-MM-DD. Defaults to 'today'."
            }
          },
          "required": ["start_date"]
        }
      },
      {
        "name": "find_free_slots",
        "description": "Finds available time slots of a given duration on a specific date, between 8:00 AM and 8:00 PM. Use this when the user wants to know when they're free, asks about availability, or wants to find time for a meeting.",
        "parameters": {
          "type": "object",
          "properties": {
            "date": {
              "type": "string",
              "description": "The date to search for free slots. Accepts 'today', 'tomorrow', or YYYY-MM-DD."
            },
            "duration_minutes": {
              "type": "integer",
              "description": "The minimum duration in minutes needed (e.g., 30, 60, 90, 120)."
            }
          },
          "required": ["date", "duration_minutes"]
        }
      },
      {
        "name": "create_event",
        "description": "Creates a new event on the user's calendar. Use this when the user explicitly asks to schedule, book, or create an event. You can include a location and notes.",
        "parameters": {
          "type": "object",
          "properties": {
            "title": {
              "type": "string",
              "description": "The title of the event."
            },
            "start_time": {
              "type": "string",
              "description": "Start time in YYYY-MM-DDTHH:MM format (e.g., 2026-03-06T14:00)."
            },
            "end_time": {
              "type": "string",
              "description": "End time in YYYY-MM-DDTHH:MM format (e.g., 2026-03-06T15:00)."
            },
            "location": {
              "type": "string",
              "description": "Optional: the location for the event."
            },
            "notes": {
              "type": "string",
              "description": "Optional: notes or description for the event."
            },
            "calendar_name": {
              "type": "string",
              "description": "Optional: specific calendar to add to. Uses default if omitted."
            }
          },
          "required": ["title", "start_time", "end_time"]
        }
      },
      {
        "name": "modify_event",
        "description": "Modifies an existing calendar event. Can change the title, time, or location. Use this when the user wants to reschedule, rename, or update an event. You need the event_id from a previous get_events or search_events call.",
        "parameters": {
          "type": "object",
          "properties": {
            "event_id": {
              "type": "string",
              "description": "The unique identifier of the event to modify (from get_events results)."
            },
            "new_title": {
              "type": "string",
              "description": "Optional: new title for the event."
            },
            "new_start_time": {
              "type": "string",
              "description": "Optional: new start time in YYYY-MM-DDTHH:MM format."
            },
            "new_end_time": {
              "type": "string",
              "description": "Optional: new end time in YYYY-MM-DDTHH:MM format."
            },
            "new_location": {
              "type": "string",
              "description": "Optional: new location for the event."
            }
          },
          "required": ["event_id"]
        }
      },
      {
        "name": "delete_event",
        "description": "Deletes a calendar event. Only use when the user explicitly asks to remove or cancel an event. Requires the event_id from a previous query. Always confirm with the user before deleting.",
        "parameters": {
          "type": "object",
          "properties": {
            "event_id": {
              "type": "string",
              "description": "The unique identifier of the event to delete."
            }
          },
          "required": ["event_id"]
        }
      },
      {
        "name": "search_events",
        "description": "Searches for calendar events by keyword in the title, location, or notes. Searches across the next N days. Use this when the user asks about a specific event by name or topic.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search keyword to look for in event titles, locations, and notes."
            },
            "days_ahead": {
              "type": "integer",
              "description": "How many days ahead to search (default: 30)."
            }
          },
          "required": ["query"]
        }
      },
      {
        "name": "check_conflicts",
        "description": "Checks if a proposed time slot conflicts with any existing events. Use this before creating an event to ensure there are no scheduling overlaps.",
        "parameters": {
          "type": "object",
          "properties": {
            "start_time": {
              "type": "string",
              "description": "Proposed start time in YYYY-MM-DDTHH:MM format."
            },
            "end_time": {
              "type": "string",
              "description": "Proposed end time in YYYY-MM-DDTHH:MM format."
            }
          },
          "required": ["start_time", "end_time"]
        }
      }
    ]
    """
    
    static func getDynamicSystemPrompt() -> String {
        let now = Date()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        dateFormatter.locale = Locale(identifier: "en_US_POSIX")
        let dateString = dateFormatter.string(from: now)
        
        let dayFormatter = DateFormatter()
        dayFormatter.dateFormat = "EEEE"
        dayFormatter.locale = Locale(identifier: "en_US_POSIX")
        let dayString = dayFormatter.string(from: now)
        
        return """
            Current date and time: \(dateString)
            Day of week: \(dayString)
            You are an intelligent calendar assistant running entirely on this device. \
            You have access to the user's real calendar through the provided tools. \
            Important rules: \
            - ALWAYS use the get_events tool to check the calendar before answering schedule questions. Never guess. \
            - When the user asks about "today", call get_events with date="today". \
            - When the user asks about "tomorrow", call get_events with date="tomorrow". \
            - When asking about a specific day, use the appropriate date string. \
            - Event results include location, notes, attendees, and duration — use this information in your answers. \
            - When finding free time, use find_free_slots. \
            - Before creating events, optionally use check_conflicts to verify no overlaps. \
            - For weekly overviews, use get_week_events. \
            - To find a specific event by name, use search_events. \
            - Format times naturally (e.g., "2:00 PM" not "14:00"). \
            - Be concise but thorough. Include location and notes when they exist. \
            - For non-calendar questions, respond normally without tools.
            """
    }
}
