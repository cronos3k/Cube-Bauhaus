//! Camera uniform data and simple fly-through camera controller.

use glam::{Mat4, Vec3, Vec4};

/// GPU-side camera uniform — 128 bytes, std140 compatible.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj:  [[f32; 4]; 4],  // 64 bytes
    pub camera_pos: [f32; 4],       // 16  (xyz = position, w unused)
    pub sun_dir:    [f32; 4],       // 16  (xyz = world-space dir toward light, w = intensity)
    pub sun_color:  [f32; 4],       // 16  (rgb = color, a unused)
    pub ambient:    [f32; 4],       // 16  (rgb = color, w = strength)
}

impl Default for CameraUniform {
    fn default() -> Self {
        let ident = Mat4::IDENTITY.to_cols_array_2d();
        Self {
            view_proj:  ident,
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            sun_dir:    [0.4, 0.8, 0.5, 1.2],   // warm afternoon sun
            sun_color:  [1.0, 0.92, 0.8, 0.0],
            ambient:    [0.4, 0.5, 0.7, 0.25],   // blue-ish sky ambient
        }
    }
}

/// Simple first-person fly camera.
pub struct FlyCamera {
    pub pos:   Vec3,
    pub yaw:   f32,   // radians, horizontal
    pub pitch: f32,   // radians, vertical (clamped ±89°)
    pub speed: f32,   // units/second

    // Movement keys held
    pub fwd:   bool,
    pub back:  bool,
    pub left:  bool,
    pub right: bool,
    pub up:    bool,
    pub down:  bool,

    // Mouse-look active (held right-mouse)
    pub mouse_look: bool,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            pos:        Vec3::new(512.0, 400.0, 512.0),
            yaw:        -std::f32::consts::FRAC_PI_4,
            pitch:      -0.3,
            speed:      80.0,
            fwd: false, back: false, left: false,
            right: false, up: false, down: false,
            mouse_look: false,
        }
    }
}

impl FlyCamera {
    /// Forward vector (ignores pitch for lateral movement keys)
    pub fn forward_flat(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    /// Full look direction (with pitch)
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward_flat().cross(Vec3::Y).normalize()
    }

    /// Apply mouse delta (in pixels) when mouse look is active.
    pub fn apply_mouse(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.002;
        self.yaw   -= dx * SENSITIVITY;
        self.pitch -= dy * SENSITIVITY;
        self.pitch  = self.pitch.clamp(-1.553, 1.553); // ±89°
    }

    /// Integrate movement over `dt` seconds (legacy boolean-field path).
    pub fn update(&mut self, dt: f32) {
        let fwd   = self.forward_flat();
        let right = self.right();
        let mut vel = Vec3::ZERO;
        if self.fwd   { vel += fwd;    }
        if self.back  { vel -= fwd;    }
        if self.right { vel += right;  }
        if self.left  { vel -= right;  }
        if self.up    { vel += Vec3::Y; }
        if self.down  { vel -= Vec3::Y; }
        if vel.length_squared() > 0.0 {
            self.pos += vel.normalize() * self.speed * dt;
        }
    }

    /// Move from a pre-computed direction vector `[x, y, z]` (from InputState::wasd_dir).
    /// `sprint` doubles speed.  Compatible with woi2 input style.
    pub fn update_from_wasd(&mut self, dir: [f32; 3], dt: f32, sprint: bool) {
        // dir[2] = forward/back (-Z = forward in woi2 convention), dir[0] = strafe, dir[1] = up
        let fwd   = self.forward_flat();
        let right = self.right();
        let vel   = right * dir[0] - fwd * dir[2] + Vec3::Y * dir[1];
        if vel.length_squared() > 0.0 {
            let speed = if sprint { self.speed * 3.0 } else { self.speed };
            self.pos += vel.normalize() * speed * dt;
        }
    }

    /// Adjust speed by scroll wheel delta (same as woi2 adjust_speed).
    pub fn adjust_speed(&mut self, scroll: f32) {
        self.speed = (self.speed * (1.0 + scroll * 0.15)).clamp(5.0, 16384.0);
    }

    /// Build the GPU uniform for this frame.
    pub fn build_uniform(&self, width: u32, height: u32) -> CameraUniform {
        let aspect = width as f32 / height.max(1) as f32;
        let view   = Mat4::look_at_rh(self.pos, self.pos + self.forward(), Vec3::Y);
        let proj   = Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.5, 65536.0);
        // Vulkan clip: flip Y
        let clip_fix = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        let view_proj = clip_fix * proj * view;

        CameraUniform {
            view_proj:  view_proj.to_cols_array_2d(),
            camera_pos: [self.pos.x, self.pos.y, self.pos.z, 1.0],
            ..CameraUniform::default()
        }
    }
}
