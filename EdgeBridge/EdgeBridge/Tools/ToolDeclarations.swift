//
//  ToolDeclarations.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// ToolDeclarations.swift
//
// Defines the calendar tool schemas as a JSON string that gets passed to
// litert_conversation_create() via the tools_json parameter. The model
// sees these declarations in its prompt context and uses them to decide
// when and how to call each function.
//
// The schema follows the Gemini API FunctionDeclaration format, which is
// what LiteRT-LM's Conversation API expects. Each tool has a name,
// description, and a parameters object defining expected argument types.
//
// IMPORTANT: The descriptions matter for model behavior — they're included
// in the prompt template and guide the model's decision about when to use
// each tool. Be specific and clear about what each function does.
// ============================================================================

import Foundation

enum ToolDeclarations {
    
    /// The JSON array of calendar tool declarations.
    /// This string is passed directly to litert_conversation_create().
    static let calendarToolsJSON: String = """
    [
      {
        "name": "get_todays_events",
        "description": "Returns all calendar events happening today, including their titles, start times, end times, and which calendar they belong to. Use this when the user asks about their schedule today or what they have going on.",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      },
      {
        "name": "get_events_for_date",
        "description": "Returns all calendar events on a specific date. Use this when the user asks about their schedule on a particular day.",
        "parameters": {
          "type": "object",
          "properties": {
            "date": {
              "type": "string",
              "description": "The date to check, in YYYY-MM-DD format (e.g., 2026-03-06). Also accepts 'today' or 'tomorrow'."
            }
          },
          "required": ["date"]
        }
      },
      {
        "name": "find_free_slots",
        "description": "Finds available time slots of a given duration on a specific date, between 8:00 AM and 8:00 PM. Use this when the user wants to know when they're free or asks about availability for a meeting.",
        "parameters": {
          "type": "object",
          "properties": {
            "date": {
              "type": "string",
              "description": "The date to search for free slots, in YYYY-MM-DD format."
            },
            "duration_minutes": {
              "type": "integer",
              "description": "The minimum duration in minutes needed for the free slot (e.g., 60 for a one-hour meeting, 120 for two hours)."
            }
          },
          "required": ["date", "duration_minutes"]
        }
      },
      {
        "name": "create_event",
        "description": "Creates a new event on the user's calendar. Use this when the user explicitly asks to schedule, book, or create a calendar event.",
        "parameters": {
          "type": "object",
          "properties": {
            "title": {
              "type": "string",
              "description": "The title of the event (e.g., 'Team Standup', 'Lunch with Sarah')."
            },
            "start_time": {
              "type": "string",
              "description": "The start time in ISO format: YYYY-MM-DDTHH:MM (e.g., 2026-03-06T14:00 for 2:00 PM)."
            },
            "end_time": {
              "type": "string",
              "description": "The end time in ISO format: YYYY-MM-DDTHH:MM (e.g., 2026-03-06T15:00 for 3:00 PM)."
            },
            "calendar_name": {
              "type": "string",
              "description": "Optional: the name of a specific calendar to add the event to. If omitted, uses the default calendar."
            }
          },
          "required": ["title", "start_time", "end_time"]
        }
      },
      {
        "name": "delete_event",
        "description": "Deletes a calendar event by its ID. Only use this when the user explicitly asks to remove or cancel a specific event. Always confirm with the user before deleting.",
        "parameters": {
          "type": "object",
          "properties": {
            "event_id": {
              "type": "string",
              "description": "The unique identifier of the event to delete. Get this from the results of get_todays_events or get_events_for_date."
            }
          },
          "required": ["event_id"]
        }
      },
      {
        "name": "get_upcoming_events",
        "description": "Returns the next N upcoming events across all calendars, looking ahead up to 30 days. Use this when the user asks about what's coming up or their upcoming schedule.",
        "parameters": {
          "type": "object",
          "properties": {
            "count": {
              "type": "integer",
              "description": "The number of upcoming events to return (default: 5, max recommended: 10)."
            }
          },
          "required": ["count"]
        }
      }
    ]
    """
    
    /// System prompt that instructs the model to act as a calendar assistant.
    /// This sets the behavioral context for how the model uses the tools.
    static let calendarSystemPrompt: String = """
    You are an intelligent calendar assistant running entirely on this device. \
    You have access to the user's real calendar through the provided tools. \
    When the user asks about their schedule, availability, or wants to manage events, \
    use the appropriate calendar tools to help them. \
    \
    Guidelines: \
    - Always check the calendar before answering schedule questions — don't guess. \
    - When the user asks about availability, use find_free_slots to find open times. \
    - When creating events, confirm the details with the user first unless they were very specific. \
    - Format times in a natural, readable way (e.g., "3:00 PM" not "15:00"). \
    - If you need to chain multiple operations (e.g., check schedule then find free time), \
      make the tool calls in sequence. \
    - For general conversation unrelated to the calendar, respond normally without using tools. \
    - Be concise but helpful in your responses.
    """
}
