//
//  CalendarToolExecutor.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// CalendarToolExecutor.swift
//
// Interfaces with Apple's EventKit framework to provide real calendar
// operations for the agentic tool-calling loop. Each method corresponds
// to a tool declared in ToolDeclarations.swift and returns a JSON-
// serializable result that gets sent back to the model via
// litert_conversation_send_tool_response().
//
// PRIVACY: All calendar access happens on-device. No data leaves the phone.
// The user must grant calendar permission the first time.
//
// ARCHITECTURE:
//   Model produces tool_calls JSON
//     → EngineViewModel parses function name + arguments
//       → CalendarToolExecutor.execute(name, args) dispatches here
//         → EventKit API reads/writes the user's real calendar
//           → Result JSON sent back to model via C bridge
// ============================================================================

import EventKit
import Foundation

class CalendarToolExecutor {
    
    // EventKit's entry point — shared across all tool calls.
    private let eventStore = EKEventStore()
    
    // Whether we've been granted calendar access.
    private var hasAccess = false
    
    // ISO date formatter for parsing model-provided date strings.
    // The model will output dates like "2026-03-06" or "2026-03-06T15:00".
    private let isoDateFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
    
    // Simpler date-only formatter for when the model provides just a date.
    private let dateOnlyFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()
    
    // Display formatter for human-readable times in tool results.
    private let displayTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()
    
    private let displayDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()
    
    private let displayDateTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()
    
    // MARK: - Permission Handling
    
    /// Request calendar access. Must be called before any tool execution.
    /// Returns true if access was granted.
    func requestAccess() async -> Bool {
        do {
            let granted = try await eventStore.requestFullAccessToEvents()
            hasAccess = granted
            return granted
        } catch {
            hasAccess = false
            return false
        }
    }
    
    // MARK: - Unified Dispatch
    
    /// Main entry point for the agentic loop. The EngineViewModel calls this
    /// with the function name and arguments parsed from the model's tool_calls
    /// JSON. Returns a JSON string that will be sent back to the model as the
    /// tool response.
    func execute(functionName: String, arguments: [String: Any]) async -> String {
        // Ensure we have calendar permission.
        if !hasAccess {
            let granted = await requestAccess()
            if !granted {
                return encodeResult([
                    "error": "Calendar access denied. Please grant permission in Settings.",
                    "tool_name": functionName
                ])
            }
        }
        
        switch functionName {
        case "get_todays_events":
            return getTodaysEvents()
            
        case "get_events_for_date":
            let dateStr = arguments["date"] as? String ?? ""
            return getEventsForDate(dateString: dateStr)
            
        case "find_free_slots":
            let dateStr = arguments["date"] as? String ?? ""
            let duration = arguments["duration_minutes"] as? Int ?? 60
            return findFreeSlots(dateString: dateStr, durationMinutes: duration)
            
        case "create_event":
            let title = arguments["title"] as? String ?? "New Event"
            let startStr = arguments["start_time"] as? String ?? ""
            let endStr = arguments["end_time"] as? String ?? ""
            let calendarName = arguments["calendar_name"] as? String
            return createEvent(title: title, startTime: startStr, endTime: endStr, calendarName: calendarName)
            
        case "delete_event":
            let eventId = arguments["event_id"] as? String ?? ""
            return deleteEvent(eventId: eventId)
            
        case "get_upcoming_events":
            let count = arguments["count"] as? Int ?? 5
            return getUpcomingEvents(count: count)
            
        default:
            return encodeResult([
                "error": "Unknown tool: \(functionName)",
                "tool_name": functionName
            ])
        }
    }
    
    // MARK: - Tool Implementations
    
    /// Returns all events happening today.
    private func getTodaysEvents() -> String {
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: Date())
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: startOfDay,
            end: endOfDay,
            calendars: nil  // Search all calendars.
        )
        let events = eventStore.events(matching: predicate)
        
        let eventList = events.map { event -> [String: String] in
            return [
                "title": event.title ?? "Untitled",
                "start_time": displayTimeFormatter.string(from: event.startDate),
                "end_time": displayTimeFormatter.string(from: event.endDate),
                "calendar": event.calendar?.title ?? "Unknown",
                "event_id": event.eventIdentifier ?? ""
            ]
        }
        
        return encodeResult([
            "tool_name": "get_todays_events",
            "date": displayDateFormatter.string(from: Date()),
            "event_count": eventList.count,
            "events": eventList
        ])
    }
    
    /// Returns all events on a specific date.
    private func getEventsForDate(dateString: String) -> String {
        guard let date = parseDate(dateString) else {
            return encodeResult([
                "tool_name": "get_events_for_date",
                "error": "Invalid date format: \(dateString). Use YYYY-MM-DD."
            ])
        }
        
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: startOfDay,
            end: endOfDay,
            calendars: nil
        )
        let events = eventStore.events(matching: predicate)
        
        let eventList = events.map { event -> [String: String] in
            return [
                "title": event.title ?? "Untitled",
                "start_time": displayTimeFormatter.string(from: event.startDate),
                "end_time": displayTimeFormatter.string(from: event.endDate),
                "calendar": event.calendar?.title ?? "Unknown",
                "event_id": event.eventIdentifier ?? ""
            ]
        }
        
        return encodeResult([
            "tool_name": "get_events_for_date",
            "date": displayDateFormatter.string(from: date),
            "event_count": eventList.count,
            "events": eventList
        ])
    }
    
    /// Finds available time slots of a given duration on a specific date.
    /// Analyzes gaps between existing events during business hours (8am-8pm).
    private func findFreeSlots(dateString: String, durationMinutes: Int) -> String {
        guard let date = parseDate(dateString) else {
            return encodeResult([
                "tool_name": "find_free_slots",
                "error": "Invalid date format: \(dateString). Use YYYY-MM-DD."
            ])
        }
        
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        
        // Define search window: 8am to 8pm.
        let windowStart = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: startOfDay)!
        let windowEnd = calendar.date(bySettingHour: 20, minute: 0, second: 0, of: startOfDay)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: windowStart,
            end: windowEnd,
            calendars: nil
        )
        let events = eventStore.events(matching: predicate)
            .sorted { $0.startDate < $1.startDate }
        
        // Find gaps between events.
        var freeSlots: [[String: String]] = []
        var currentTime = windowStart
        let requiredDuration = TimeInterval(durationMinutes * 60)
        
        for event in events {
            // If there's a gap before this event that's long enough...
            let gap = event.startDate.timeIntervalSince(currentTime)
            if gap >= requiredDuration {
                freeSlots.append([
                    "start": displayTimeFormatter.string(from: currentTime),
                    "end": displayTimeFormatter.string(from: event.startDate),
                    "duration_minutes": "\(Int(gap / 60))"
                ])
            }
            // Move current time to after this event (or keep it if event
            // ends before our current position, e.g. overlapping events).
            if event.endDate > currentTime {
                currentTime = event.endDate
            }
        }
        
        // Check the gap after the last event until end of window.
        let finalGap = windowEnd.timeIntervalSince(currentTime)
        if finalGap >= requiredDuration {
            freeSlots.append([
                "start": displayTimeFormatter.string(from: currentTime),
                "end": displayTimeFormatter.string(from: windowEnd),
                "duration_minutes": "\(Int(finalGap / 60))"
            ])
        }
        
        return encodeResult([
            "tool_name": "find_free_slots",
            "date": displayDateFormatter.string(from: date),
            "duration_requested_minutes": durationMinutes,
            "free_slots_count": freeSlots.count,
            "free_slots": freeSlots
        ])
    }
    
    /// Creates a new calendar event.
    private func createEvent(title: String, startTime: String, endTime: String, calendarName: String?) -> String {
        guard let startDate = parseDateTime(startTime) else {
            return encodeResult([
                "tool_name": "create_event",
                "error": "Invalid start_time format: \(startTime). Use ISO 8601 (e.g., 2026-03-06T15:00)."
            ])
        }
        guard let endDate = parseDateTime(endTime) else {
            return encodeResult([
                "tool_name": "create_event",
                "error": "Invalid end_time format: \(endTime). Use ISO 8601 (e.g., 2026-03-06T17:00)."
            ])
        }
        
        let event = EKEvent(eventStore: eventStore)
        event.title = title
        event.startDate = startDate
        event.endDate = endDate
        
        // Use specified calendar or default.
        if let calName = calendarName,
           let targetCalendar = eventStore.calendars(for: .event).first(where: { $0.title == calName }) {
            event.calendar = targetCalendar
        } else {
            event.calendar = eventStore.defaultCalendarForNewEvents
        }
        
        do {
            try eventStore.save(event, span: .thisEvent)
            return encodeResult([
                "tool_name": "create_event",
                "success": true,
                "title": title,
                "start_time": displayDateTimeFormatter.string(from: startDate),
                "end_time": displayDateTimeFormatter.string(from: endDate),
                "calendar": event.calendar?.title ?? "Default",
                "event_id": event.eventIdentifier ?? ""
            ])
        } catch {
            return encodeResult([
                "tool_name": "create_event",
                "error": "Failed to create event: \(error.localizedDescription)"
            ])
        }
    }
    
    /// Deletes a calendar event by its identifier.
    private func deleteEvent(eventId: String) -> String {
        guard let event = eventStore.event(withIdentifier: eventId) else {
            return encodeResult([
                "tool_name": "delete_event",
                "error": "Event not found with id: \(eventId)"
            ])
        }
        
        let title = event.title ?? "Untitled"
        
        do {
            try eventStore.remove(event, span: .thisEvent)
            return encodeResult([
                "tool_name": "delete_event",
                "success": true,
                "deleted_title": title
            ])
        } catch {
            return encodeResult([
                "tool_name": "delete_event",
                "error": "Failed to delete event: \(error.localizedDescription)"
            ])
        }
    }
    
    /// Returns the next N upcoming events across all calendars.
    private func getUpcomingEvents(count: Int) -> String {
        let now = Date()
        // Look ahead 30 days.
        let futureDate = Calendar.current.date(byAdding: .day, value: 30, to: now)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: now,
            end: futureDate,
            calendars: nil
        )
        let events = eventStore.events(matching: predicate)
            .prefix(count)
        
        let eventList = events.map { event -> [String: String] in
            return [
                "title": event.title ?? "Untitled",
                "date": displayDateFormatter.string(from: event.startDate),
                "start_time": displayTimeFormatter.string(from: event.startDate),
                "end_time": displayTimeFormatter.string(from: event.endDate),
                "calendar": event.calendar?.title ?? "Unknown",
                "event_id": event.eventIdentifier ?? ""
            ]
        }
        
        return encodeResult([
            "tool_name": "get_upcoming_events",
            "count_requested": count,
            "count_returned": eventList.count,
            "events": Array(eventList)
        ])
    }
    
    // MARK: - Date Parsing Helpers
    
    /// Parses a date string, trying multiple formats.
    /// Handles: "2026-03-06", "2026-03-06T15:00", "2026-03-06T15:00:00Z", "tomorrow", "today"
    private func parseDate(_ string: String) -> Date? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        
        // Handle relative dates that the model might produce.
        if trimmed == "today" {
            return Date()
        }
        if trimmed == "tomorrow" {
            return Calendar.current.date(byAdding: .day, value: 1, to: Date())
        }
        
        // Try date-only format first.
        if let date = dateOnlyFormatter.date(from: string) {
            return date
        }
        
        // Try ISO 8601 with time.
        if let date = isoDateFormatter.date(from: string) {
            return date
        }
        
        // Try common format without timezone: "2026-03-06T15:00"
        let flexFormatter = DateFormatter()
        flexFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm"
        flexFormatter.locale = Locale(identifier: "en_US_POSIX")
        if let date = flexFormatter.date(from: string) {
            return date
        }
        
        return nil
    }
    
    /// Parses a datetime string for event creation. Same as parseDate but
    /// ensures we get a time component.
    private func parseDateTime(_ string: String) -> Date? {
        return parseDate(string)
    }
    
    // MARK: - JSON Encoding
    
    /// Encodes a dictionary as a JSON string for sending back to the model.
    private func encodeResult(_ dict: [String: Any]) -> String {
        do {
            let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
            return String(data: data, encoding: .utf8) ?? "{\"error\": \"JSON encoding failed\"}"
        } catch {
            return "{\"error\": \"JSON encoding failed: \(error.localizedDescription)\"}"
        }
    }
}
