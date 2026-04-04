# Iced Migration Assessment

## Current State

- GUI is ~1,546 lines in a single `gui/app.rs` file using `egui`/`eframe` 0.27
- Immediate-mode rendering (UI rebuilt every frame)
- 7 modal dialogs, dynamic config grids, real-time system monitoring bars, download queue with progress, color-coded log viewer

## What Makes Iced Different

- **Retained-mode** (Elm architecture) — state changes produce messages, messages update state, state produces new UI
- Requires splitting your single `App` struct into `Model`, `Message`, and `update()`/`view()` functions
- Async is built-in via `Command` (no manual thread spawning + `request_repaint()` needed)
- More boilerplate but better type safety and testability

## Key Migration Challenges

1. **Architecture shift** — immediate-mode → Elm architecture requires restructuring all UI logic
2. **Threading model** — current `std::thread::spawn` + `Arc<RwLock<...>>` + `ctx.request_repaint()` pattern becomes `Command::perform` + `Message` dispatch
3. **Dynamic config grid** — egui's immediate mode makes dynamic field rendering trivial; iced requires more careful widget composition
4. **Real-time throttling** — egui's frame-caching pattern doesn't translate directly; iced uses subscriptions
5. **Ecosystem maturity** — iced has fewer ready-made widgets than egui (e.g., no built-in color-coded text viewer, fewer layout options)

## Estimated Effort

40-80 hours for a full migration given the 7 dialogs, provider setup flows, download manager, and real-time monitoring.

## Is It Worth It?

- **Stay with egui** if you value rapid iteration, less boilerplate, and the current code works well
- **Migrate to iced** if you want better compile-time guarantees, testable UI logic, and a more "idiomatic Rust" architecture

---

*Assessed: Fri Apr 03 2026*
*Decision: No migration planned at this time.*
