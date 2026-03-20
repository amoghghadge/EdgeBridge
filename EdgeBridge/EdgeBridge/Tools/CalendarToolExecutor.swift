//
//  CalendarToolExecutor.swift
//  EdgeBridge
//
//  Created by Amogh Ghadge on 3/4/26.
//

// ============================================================================
// CalendarToolExecutor.swift  (v2 — Enhanced Calendar Agent)
//
// IMPROVEMENTS OVER v1:
// 1. Events now include location, notes, URL, and attendee info
// 2. Merged get_todays_events into get_events_for_date (accepts "today"/"tomorrow")
// 3. Added get_week_events for weekly overview
// 4. Added search_events to find events by title keyword
// 5. Added check_conflicts to detect scheduling overlaps
// 6. Added modify_event to reschedule existing events
// 7. Better date parsing with relative dates ("next monday", etc.)
// 8. All-day events handled properly
// ============================================================================

import EventKit
import Foundation

class CalendarToolExecutor {
    
    private let eventStore = EKEventStore()
    private var hasAccess = false
    
    // Formatters
    private let dateOnlyFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()
    
    private let displayTimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()
    
    private let displayDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .none
        return f
    }()
    
    private let displayDateTimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()
    
    // MARK: - Permission
    
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
    
    func execute(functionName: String, arguments: [String: Any]) async -> String {
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
        case "get_events":
            let dateStr = arguments["date"] as? String ?? "today"
            return getEvents(dateString: dateStr)
            
        case "get_week_events":
            let startDateStr = arguments["start_date"] as? String ?? "today"
            return getWeekEvents(startDateString: startDateStr)
            
        case "find_free_slots":
            let dateStr = arguments["date"] as? String ?? "today"
            let duration = arguments["duration_minutes"] as? Int ?? 60
            return findFreeSlots(dateString: dateStr, durationMinutes: duration)
            
        case "create_event":
            let title = arguments["title"] as? String ?? "New Event"
            let startStr = arguments["start_time"] as? String ?? ""
            let endStr = arguments["end_time"] as? String ?? ""
            let location = arguments["location"] as? String
            let notes = arguments["notes"] as? String
            let calendarName = arguments["calendar_name"] as? String
            return createEvent(
                title: title, startTime: startStr, endTime: endStr,
                location: location, notes: notes, calendarName: calendarName
            )
            
        case "modify_event":
            let eventId = arguments["event_id"] as? String ?? ""
            let newTitle = arguments["new_title"] as? String
            let newStart = arguments["new_start_time"] as? String
            let newEnd = arguments["new_end_time"] as? String
            let newLocation = arguments["new_location"] as? String
            return modifyEvent(
                eventId: eventId, newTitle: newTitle,
                newStart: newStart, newEnd: newEnd, newLocation: newLocation
            )
            
        case "delete_event":
            let eventId = arguments["event_id"] as? String ?? ""
            return deleteEvent(eventId: eventId)
            
        case "search_events":
            let query = arguments["query"] as? String ?? ""
            var days = arguments["days_ahead"] as? Int ?? 30
            if days == 0 {
                days = 30
            }
            return searchEvents(query: query, daysAhead: days)
            
        case "check_conflicts":
            let startStr = arguments["start_time"] as? String ?? ""
            let endStr = arguments["end_time"] as? String ?? ""
            return checkConflicts(startTime: startStr, endTime: endStr)
            
        // Legacy support for v1 tool names
        case "get_todays_events":
            return getEvents(dateString: "today")
            
        case "get_events_for_date":
            let dateStr = arguments["date"] as? String ?? "today"
            return getEvents(dateString: dateStr)
            
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
    
    /// Returns all events on a specific date. Accepts "today", "tomorrow",
    /// "yesterday", or a YYYY-MM-DD date string.
    private func getEvents(dateString: String) -> String {
        guard let date = parseDate(dateString) else {
            return encodeResult([
                "tool_name": "get_events",
                "error": "Invalid date: \(dateString). Use YYYY-MM-DD, 'today', or 'tomorrow'."
            ])
        }
        
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let endOfDay = calendar.date(byAdding: .day, value: 1, to: startOfDay)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: startOfDay, end: endOfDay, calendars: nil
        )
        let events = eventStore.events(matching: predicate)
        let eventList = events.map { formatEvent($0) }
        
        return encodeResult([
            "tool_name": "get_events",
            "date": displayDateFormatter.string(from: date),
            "day_of_week": dayOfWeekString(date),
            "event_count": eventList.count,
            "events": eventList
        ])
    }
    
    /// Returns events for an entire week starting from the given date.
    private func getWeekEvents(startDateString: String) -> String {
        guard let startDate = parseDate(startDateString) else {
            return encodeResult([
                "tool_name": "get_week_events",
                "error": "Invalid date: \(startDateString)."
            ])
        }
        
        let calendar = Calendar.current
        let startOfWeek = calendar.startOfDay(for: startDate)
        let endOfWeek = calendar.date(byAdding: .day, value: 7, to: startOfWeek)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: startOfWeek, end: endOfWeek, calendars: nil
        )
        let events = eventStore.events(matching: predicate)
        
        // Group events by day for a clear weekly overview.
        var dayGroups: [[String: Any]] = []
        for dayOffset in 0..<7 {
            let day = calendar.date(byAdding: .day, value: dayOffset, to: startOfWeek)!
            let nextDay = calendar.date(byAdding: .day, value: 1, to: day)!
            
            let dayEvents = events.filter { event in
                event.startDate >= day && event.startDate < nextDay
            }
            
            if !dayEvents.isEmpty {
                dayGroups.append([
                    "date": displayDateFormatter.string(from: day),
                    "day_of_week": dayOfWeekString(day),
                    "events": dayEvents.map { formatEvent($0) }
                ])
            }
        }
        
        return encodeResult([
            "tool_name": "get_week_events",
            "start_date": displayDateFormatter.string(from: startOfWeek),
            "end_date": displayDateFormatter.string(from: calendar.date(byAdding: .day, value: 6, to: startOfWeek)!),
            "total_events": events.count,
            "days_with_events": dayGroups.count,
            "schedule": dayGroups
        ])
    }
    
    /// Finds available time slots of a given duration on a specific date.
    private func findFreeSlots(dateString: String, durationMinutes: Int) -> String {
        guard let date = parseDate(dateString) else {
            return encodeResult([
                "tool_name": "find_free_slots",
                "error": "Invalid date: \(dateString)."
            ])
        }
        
        let calendar = Calendar.current
        let startOfDay = calendar.startOfDay(for: date)
        let windowStart = calendar.date(bySettingHour: 8, minute: 0, second: 0, of: startOfDay)!
        let windowEnd = calendar.date(bySettingHour: 20, minute: 0, second: 0, of: startOfDay)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: windowStart, end: windowEnd, calendars: nil
        )
        let events = eventStore.events(matching: predicate)
            .filter { !$0.isAllDay }  // Exclude all-day events from free slot computation.
            .sorted { $0.startDate < $1.startDate }
        
        var freeSlots: [[String: String]] = []
        var currentTime = windowStart
        let requiredDuration = TimeInterval(durationMinutes * 60)
        
        for event in events {
            let gap = event.startDate.timeIntervalSince(currentTime)
            if gap >= requiredDuration {
                freeSlots.append([
                    "start": displayTimeFormatter.string(from: currentTime),
                    "end": displayTimeFormatter.string(from: event.startDate),
                    "duration_minutes": "\(Int(gap / 60))"
                ])
            }
            if event.endDate > currentTime {
                currentTime = event.endDate
            }
        }
        
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
            "day_of_week": dayOfWeekString(date),
            "duration_requested_minutes": durationMinutes,
            "search_window": "8:00 AM - 8:00 PM",
            "free_slots_count": freeSlots.count,
            "free_slots": freeSlots
        ])
    }
    
    /// Creates a new calendar event with full metadata.
    private func createEvent(
        title: String, startTime: String, endTime: String,
        location: String?, notes: String?, calendarName: String?
    ) -> String {
        guard let startDate = parseDateTime(startTime) else {
            return encodeResult([
                "tool_name": "create_event",
                "error": "Invalid start_time: \(startTime). Use YYYY-MM-DDTHH:MM format."
            ])
        }
        guard let endDate = parseDateTime(endTime) else {
            return encodeResult([
                "tool_name": "create_event",
                "error": "Invalid end_time: \(endTime). Use YYYY-MM-DDTHH:MM format."
            ])
        }
        
        let event = EKEvent(eventStore: eventStore)
        event.title = title
        event.startDate = startDate
        event.endDate = endDate
        
        if let loc = location, !loc.isEmpty {
            event.location = loc
        }
        if let n = notes, !n.isEmpty {
            event.notes = n
        }
        
        if let calName = calendarName,
           let cal = eventStore.calendars(for: .event).first(where: { $0.title == calName }) {
            event.calendar = cal
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
                "location": location ?? "",
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
    
    /// Modifies an existing event (reschedule, rename, change location).
    private func modifyEvent(
        eventId: String, newTitle: String?,
        newStart: String?, newEnd: String?, newLocation: String?
    ) -> String {
        guard let event = eventStore.event(withIdentifier: eventId) else {
            return encodeResult([
                "tool_name": "modify_event",
                "error": "Event not found with id: \(eventId)"
            ])
        }
        
        let originalTitle = event.title ?? "Untitled"
        
        if let t = newTitle, !t.isEmpty { event.title = t }
        if let s = newStart, let d = parseDateTime(s) { event.startDate = d }
        if let e = newEnd, let d = parseDateTime(e) { event.endDate = d }
        if let l = newLocation { event.location = l }
        
        do {
            try eventStore.save(event, span: .thisEvent)
            return encodeResult([
                "tool_name": "modify_event",
                "success": true,
                "original_title": originalTitle,
                "updated_event": formatEvent(event)
            ])
        } catch {
            return encodeResult([
                "tool_name": "modify_event",
                "error": "Failed to modify event: \(error.localizedDescription)"
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
                "error": "Failed to delete: \(error.localizedDescription)"
            ])
        }
    }
    
    /// Searches for events by title keyword within the next N days.
    private func searchEvents(query: String, daysAhead: Int) -> String {
        let now = Date()
        let future = Calendar.current.date(byAdding: .day, value: daysAhead, to: now)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: now, end: future, calendars: nil
        )
        let allEvents = eventStore.events(matching: predicate)
        let matches = allEvents.filter { event in
            let title = event.title ?? ""
            let location = event.location ?? ""
            let notes = event.notes ?? ""
            let q = query.lowercased()
            return title.lowercased().contains(q) ||
                   location.lowercased().contains(q) ||
                   notes.lowercased().contains(q)
        }
        
        return encodeResult([
            "tool_name": "search_events",
            "query": query,
            "days_searched": daysAhead,
            "match_count": matches.count,
            "events": matches.prefix(10).map { formatEvent($0) }
        ])
    }
    
    /// Checks if a proposed time slot conflicts with existing events.
    private func checkConflicts(startTime: String, endTime: String) -> String {
        guard let startDate = parseDateTime(startTime) else {
            return encodeResult([
                "tool_name": "check_conflicts",
                "error": "Invalid start_time: \(startTime)."
            ])
        }
        guard let endDate = parseDateTime(endTime) else {
            return encodeResult([
                "tool_name": "check_conflicts",
                "error": "Invalid end_time: \(endTime)."
            ])
        }
        
        let predicate = eventStore.predicateForEvents(
            withStart: startDate, end: endDate, calendars: nil
        )
        let conflicts = eventStore.events(matching: predicate)
            .filter { !$0.isAllDay }
        
        return encodeResult([
            "tool_name": "check_conflicts",
            "proposed_start": displayDateTimeFormatter.string(from: startDate),
            "proposed_end": displayDateTimeFormatter.string(from: endDate),
            "has_conflicts": !conflicts.isEmpty,
            "conflict_count": conflicts.count,
            "conflicting_events": conflicts.map { formatEvent($0) }
        ])
    }
    
    /// Returns the next N upcoming events.
    private func getUpcomingEvents(count: Int) -> String {
        let now = Date()
        let future = Calendar.current.date(byAdding: .day, value: 30, to: now)!
        
        let predicate = eventStore.predicateForEvents(
            withStart: now, end: future, calendars: nil
        )
        let events = Array(eventStore.events(matching: predicate).prefix(count))
        
        return encodeResult([
            "tool_name": "get_upcoming_events",
            "count_requested": count,
            "count_returned": events.count,
            "events": events.map { formatEvent($0) }
        ])
    }
    
    // MARK: - Event Formatting
    
    /// Formats an EKEvent into a comprehensive dictionary including
    /// location, notes, attendees, and all-day status.
    private func formatEvent(_ event: EKEvent) -> [String: Any] {
        var dict: [String: Any] = [
            "title": event.title ?? "Untitled",
            "calendar": event.calendar?.title ?? "Unknown",
            "event_id": event.eventIdentifier ?? ""
        ]
        
        // Time info — handle all-day events differently.
        if event.isAllDay {
            dict["all_day"] = true
            dict["date"] = displayDateFormatter.string(from: event.startDate)
        } else {
            dict["all_day"] = false
            dict["start_time"] = displayTimeFormatter.string(from: event.startDate)
            dict["end_time"] = displayTimeFormatter.string(from: event.endDate)
            dict["date"] = displayDateFormatter.string(from: event.startDate)
            
            // Duration in minutes for convenience.
            let durationMinutes = Int(event.endDate.timeIntervalSince(event.startDate) / 60)
            dict["duration_minutes"] = durationMinutes
        }
        
        // Location — this was missing in v1!
        if let location = event.location, !location.isEmpty {
            dict["location"] = location
        }
        
        // Notes
        if let notes = event.notes, !notes.isEmpty {
            dict["notes"] = notes
        }
        
        // URL
        if let url = event.url {
            dict["url"] = url.absoluteString
        }
        
        // Attendees (if any)
        if let attendees = event.attendees, !attendees.isEmpty {
            dict["attendees"] = attendees.compactMap { participant -> [String: String]? in
                guard let name = participant.name else { return nil }
                return [
                    "name": name,
                    "status": participantStatusString(participant.participantStatus)
                ]
            }
        }
        
        // Recurrence info
        if event.hasRecurrenceRules {
            dict["recurring"] = true
        }
        
        return dict
    }
    
    // MARK: - Date Parsing
    
    private func parseDate(_ string: String) -> Date? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let calendar = Calendar.current
        
        // Relative dates
        switch trimmed {
        case "today":
            return Date()
        case "tomorrow":
            return calendar.date(byAdding: .day, value: 1, to: Date())
        case "yesterday":
            return calendar.date(byAdding: .day, value: -1, to: Date())
        case "next monday", "next mon":
            return nextWeekday(.monday)
        case "next tuesday", "next tue":
            return nextWeekday(.tuesday)
        case "next wednesday", "next wed":
            return nextWeekday(.wednesday)
        case "next thursday", "next thu":
            return nextWeekday(.thursday)
        case "next friday", "next fri":
            return nextWeekday(.friday)
        case "next saturday", "next sat":
            return nextWeekday(.saturday)
        case "next sunday", "next sun":
            return nextWeekday(.sunday)
        default:
            break
        }
        
        // YYYY-MM-DD
        if let date = dateOnlyFormatter.date(from: string) {
            return date
        }
        
        // ISO 8601 variants
        let flexFormatter = DateFormatter()
        flexFormatter.locale = Locale(identifier: "en_US_POSIX")
        
        for format in ["yyyy-MM-dd'T'HH:mm:ssZ", "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd'T'HH:mm"] {
            flexFormatter.dateFormat = format
            if let date = flexFormatter.date(from: string) {
                return date
            }
        }
        
        return nil
    }
    
    private func parseDateTime(_ string: String) -> Date? {
        return parseDate(string)
    }
    
    private func nextWeekday(_ target: Weekday) -> Date {
        let calendar = Calendar.current
        var date = Date()
        for _ in 1...7 {
            date = calendar.date(byAdding: .day, value: 1, to: date)!
            let weekday = calendar.component(.weekday, from: date)
            if weekday == target.rawValue {
                return date
            }
        }
        return date
    }
    
    private enum Weekday: Int {
        case sunday = 1, monday = 2, tuesday = 3, wednesday = 4
        case thursday = 5, friday = 6, saturday = 7
    }
    
    // MARK: - Helpers
    
    private func dayOfWeekString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE"
        return formatter.string(from: date)
    }
    
    private func participantStatusString(_ status: EKParticipantStatus) -> String {
        switch status {
        case .accepted: return "accepted"
        case .declined: return "declined"
        case .tentative: return "tentative"
        case .pending: return "pending"
        default: return "unknown"
        }
    }
    
    private func encodeResult(_ dict: [String: Any]) -> String {
        do {
            let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
            return String(data: data, encoding: .utf8) ?? "{\"error\": \"JSON encoding failed\"}"
        } catch {
            return "{\"error\": \"JSON encoding failed: \(error.localizedDescription)\"}"
        }
    }
}
