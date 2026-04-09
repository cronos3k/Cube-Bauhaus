//! Input Manager — centralized input handling for all devices.
//!
//! Separates raw input events from game/editor logic.
//! Designed for extensibility: keyboard, mouse, gamepad, flight stick, 3D space mouse.
//!
//! Architecture:
//!   winit events → InputManager::handle_*() → InputState updated
//!   Per-frame:    InputManager::flush() → produces InputFrame
//!   Consumers:    FlyCamera, Editor, Game read InputFrame
//!
//! The InputManager does NOT know about cameras, editors, or game logic.
//! It only tracks what buttons are pressed and what the mouse did.

use std::collections::HashSet;
use winit::keyboard::KeyCode;
use winit::event::{ElementState, MouseButton};

/// Raw input state — updated by events, read by consumers.
pub struct InputState {
    // Keyboard
    pub keys_held: HashSet<KeyCode>,
    pub keys_pressed: HashSet<KeyCode>,   // just pressed THIS frame (cleared each flush)
    pub keys_released: HashSet<KeyCode>,  // just released THIS frame

    // Modifiers (always tracked, even across focus changes)
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,

    // Mouse buttons
    pub lmb: bool,
    pub rmb: bool,
    pub mmb: bool,
    pub lmb_just_pressed: bool,
    pub lmb_just_released: bool,
    pub rmb_just_pressed: bool,
    pub rmb_just_released: bool,
    pub mmb_just_pressed: bool,
    pub mmb_just_released: bool,

    // Cursor position (physical pixels)
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub cursor_dx: f32,  // delta since last flush
    pub cursor_dy: f32,

    // Raw mouse motion (hardware mickeys, for camera look)
    pub raw_mouse_dx: f64,
    pub raw_mouse_dy: f64,

    // Scroll
    pub scroll_y: f32,  // accumulated since last flush

    // Window
    pub window_w: u32,
    pub window_h: u32,
}

impl InputState {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            keys_held: HashSet::new(),
            keys_pressed: HashSet::new(),
            keys_released: HashSet::new(),
            shift: false, ctrl: false, alt: false,
            lmb: false, rmb: false, mmb: false,
            lmb_just_pressed: false, lmb_just_released: false,
            rmb_just_pressed: false, rmb_just_released: false,
            mmb_just_pressed: false, mmb_just_released: false,
            cursor_x: 0.0, cursor_y: 0.0,
            cursor_dx: 0.0, cursor_dy: 0.0,
            raw_mouse_dx: 0.0, raw_mouse_dy: 0.0,
            scroll_y: 0.0,
            window_w: w, window_h: h,
        }
    }

    /// Call once per frame AFTER all consumers have read the state.
    /// Clears per-frame deltas and "just pressed/released" flags.
    pub fn flush(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.lmb_just_pressed = false;
        self.lmb_just_released = false;
        self.rmb_just_pressed = false;
        self.rmb_just_released = false;
        self.mmb_just_pressed = false;
        self.mmb_just_released = false;
        self.cursor_dx = 0.0;
        self.cursor_dy = 0.0;
        self.raw_mouse_dx = 0.0;
        self.raw_mouse_dy = 0.0;
        self.scroll_y = 0.0;
    }

    // ── Event handlers (called from winit event loop) ──

    pub fn handle_key(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if self.keys_held.insert(code) {
                    self.keys_pressed.insert(code);  // only on first press, not repeat
                }
            }
            ElementState::Released => {
                self.keys_held.remove(&code);
                self.keys_released.insert(code);
            }
        }
        // Track modifiers
        match code {
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.shift = state == ElementState::Pressed;
            }
            KeyCode::ControlLeft | KeyCode::ControlRight => {
                self.ctrl = state == ElementState::Pressed;
            }
            KeyCode::AltLeft | KeyCode::AltRight => {
                self.alt = state == ElementState::Pressed;
            }
            _ => {}
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => {
                if pressed && !self.lmb { self.lmb_just_pressed = true; }
                if !pressed && self.lmb { self.lmb_just_released = true; }
                self.lmb = pressed;
            }
            MouseButton::Right => {
                if pressed && !self.rmb { self.rmb_just_pressed = true; }
                if !pressed && self.rmb { self.rmb_just_released = true; }
                self.rmb = pressed;
            }
            MouseButton::Middle => {
                if pressed && !self.mmb { self.mmb_just_pressed = true; }
                if !pressed && self.mmb { self.mmb_just_released = true; }
                self.mmb = pressed;
            }
            _ => {}
        }
    }

    pub fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        let new_x = x as f32;
        let new_y = y as f32;
        self.cursor_dx += new_x - self.cursor_x;
        self.cursor_dy += new_y - self.cursor_y;
        self.cursor_x = new_x;
        self.cursor_y = new_y;
    }

    pub fn handle_raw_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.raw_mouse_dx += dx;
        self.raw_mouse_dy += dy;
    }

    pub fn handle_scroll(&mut self, delta_y: f32) {
        self.scroll_y += delta_y;
    }

    // ── Query helpers ──

    /// Was this key just pressed this frame?
    pub fn just_pressed(&self, code: KeyCode) -> bool {
        self.keys_pressed.contains(&code)
    }

    /// Is this key currently held down?
    pub fn held(&self, code: KeyCode) -> bool {
        self.keys_held.contains(&code)
    }

    /// Was this key just released this frame?
    pub fn just_released(&self, code: KeyCode) -> bool {
        self.keys_released.contains(&code)
    }

    /// Check for a key combo like Ctrl+Z.
    pub fn ctrl_pressed(&self, code: KeyCode) -> bool {
        self.ctrl && self.just_pressed(code)
    }

    /// Check for Alt+key combo.
    pub fn alt_pressed(&self, code: KeyCode) -> bool {
        self.alt && self.just_pressed(code)
    }

    /// WASD movement vector (unit or zero). Only meaningful when consumed for movement.
    pub fn wasd_dir(&self) -> [f32; 3] {
        let mut dir = [0.0f32; 3];
        if self.held(KeyCode::KeyW) { dir[2] -= 1.0; } // forward = -Z
        if self.held(KeyCode::KeyS) { dir[2] += 1.0; } // backward = +Z
        if self.held(KeyCode::KeyA) { dir[0] -= 1.0; } // left = -X
        if self.held(KeyCode::KeyD) { dir[0] += 1.0; } // right = +X
        if self.held(KeyCode::KeyE) || self.held(KeyCode::Space) { dir[1] += 1.0; } // up
        if self.held(KeyCode::KeyQ) || self.held(KeyCode::ControlLeft) { dir[1] -= 1.0; } // down
        dir
    }
}
