//! BBC — BananaBread / Cube2 octree geometry viewer + editor.
//!
//! Controls (fly mode):
//!   WASD=move, Q/E or Space/Ctrl=up/down, Shift=sprint
//!   Mouse=look (click to grab, Esc to release)
//!   Scroll=speed, F=wireframe, Esc×2=quit
//!
//! Editor mode (press E to toggle):
//!   LMB=select, Space=cancel, Delete=delete cubes
//!   Scroll=fill/empty, F+Scroll=push/pull face, .+Scroll=push corner
//!   G+Scroll=grid, 1=tex slot, 2=tex rotate, 3=tex scale, 4=tex offset, 5=material, R+Scroll=rotate
//!   X=flip, C=copy, V=paste, Z/U=undo, I=redo

mod input;
mod editor;
mod screenshot;

use std::time::Instant;

use winit::{
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

use bbc_renderer::{FlyCamera, GpuMesh, Renderer, Vertex};
use cube_world::{build_mesh_with_textures, build_wireframe, load_ogz, octree::{Cube, OctreeWorld}};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("bbc=info".parse().unwrap())
                .add_directive("bbc_renderer=info".parse().unwrap())
                .add_directive("cube_world=info".parse().unwrap()),
        )
        .init();

    let map_path = std::env::args().nth(1);

    // ── Window ────────────────────────────────────────────────────────────────
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = WindowBuilder::new()
        .with_title("BBC — Cube2 Editor")
        .with_inner_size(winit::dpi::LogicalSize::new(1600u32, 900u32))
        .build(&event_loop)
        .unwrap();

    // ── Renderer ──────────────────────────────────────────────────────────────
    let mut renderer = Renderer::new(&window);

    // ── World + Editor ────────────────────────────────────────────────────────
    let world = match &map_path {
        Some(path) => {
            println!("Loading OGZ: {path}");
            match load_ogz(path) {
                Ok(w) => {
                    println!(
                        "Loaded: world_scale={} ({} units), {} entities, ogz_version={}",
                        w.world_scale, w.world_size(), w.entities.len(), w.ogz_version,
                    );
                    w
                }
                Err(e) => {
                    eprintln!("Failed to load OGZ: {e} — falling back to test world");
                    make_test_world()
                }
            }
        }
        None => {
            println!("No map specified — generating test world.  Pass a .ogz path to load one.");
            make_test_world()
        }
    };

    let mut editor = editor::EditorState::new(world);

    // ── Load map textures ────────────────────────────────────────────────────
    if let Some(ref path) = map_path {
        editor.current_map_path = Some(path.clone());
        load_map_textures(path, &mut editor, &mut renderer);
    }

    // ── Build initial meshes ──────────────────────────────────────────────────
    let (verts, idxs) = build_mesh_with_textures(&editor.edit_world.world, Some(&editor.tex_registry));
    println!("Mesh: {} vertices, {} triangles", verts.len(), idxs.len() / 3);
    let gpu_verts: &[Vertex] = bytemuck::cast_slice(&verts);
    let mut solid_mesh = renderer.upload_mesh(gpu_verts, &idxs);

    let (wverts, widxs) = build_wireframe(&editor.edit_world.world);
    let gpu_wverts: &[Vertex] = bytemuck::cast_slice(&wverts);
    let mut wire_mesh = renderer.upload_mesh(gpu_wverts, &widxs);

    // Editor overlay mesh (rebuilt each frame when in edit mode)
    let mut overlay_mesh: Option<GpuMesh> = None;
    let mut crosshair_mesh: Option<GpuMesh> = None;

    // Frame-delayed deletion queue: with 2 frames in flight, we must keep
    // meshes alive for at least 2 frames before destroying them.
    let mut delete_queue: [Vec<GpuMesh>; 3] = [vec![], vec![], vec![]];
    let mut delete_frame: usize = 0;

    // ── Camera ────────────────────────────────────────────────────────────────
    let ws = editor.edit_world.world.world_size() as f32;
    let mut camera = FlyCamera {
        pos:   glam::Vec3::new(ws * 0.5, ws * 0.9, ws * 0.2),
        yaw:   0.0,
        pitch: -0.8,
        speed: (ws * 0.25).max(80.0).min(4096.0),
        ..FlyCamera::default()
    };

    // ── egui integration ──────────────────────────────────────────────────────
    let egui_ctx = egui::Context::default();
    egui_ctx.set_visuals(egui::Visuals::dark());
    let mut egui_winit_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None, // max texture size
    );
    let mut egui_renderer = bbc_renderer::EguiRenderer::new(
        &renderer.device,
        renderer.swapchain.format.format,
    );
    let mut ui_mode = false; // false = camera mode, true = menu/UI mode

    // ── FPS counter ──────────────────────────────────────────────────────────
    let mut frame_count = 0u32;
    let mut fps_timer = std::time::Instant::now();
    let mut current_fps = 0.0f32;

    // ── Mesh stats ───────────────────────────────────────────────────────────
    let mut vert_count = verts.len();
    let mut tri_count = idxs.len() / 3;

    // Window title
    let map_stem = map_path.as_deref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("test-world");
    window.set_title(&format!(
        "BBC — {map_stem}  [E=edit mode | WASD=move | Scroll=speed | F=wire]"
    ));
    println!("Controls: E=edit mode | WASD move | mouse look (click grab, Esc release) | Scroll=speed | Shift=sprint | F=wireframe | Esc=quit");
    println!("Editor:   LMB=select face | RMB=select vertices | Scroll=fill/push | F+Scroll=edge push | G+Scroll=grid | Space=cancel | Del=delete | X=flip | R+Scroll=rotate | C=copy V=paste | Z=undo I=redo");
    println!("Texture:  1+Scroll=slot | 2+Scroll=rotate | 3+Scroll=scale | 4+Scroll=offset (Shift=2nd axis) | 5+Scroll=material");
    println!("File:     Ctrl+S=save | Ctrl+O=open | Ctrl+N=new map | Ctrl+E=export GLB | Ctrl+Shift+E=export FBX");

    // ── Input + state ─────────────────────────────────────────────────────────
    let sz = window.inner_size();
    let mut inp  = input::InputState::new(sz.width, sz.height);
    let mut show_wire       = false;
    let mut mouse_captured  = false;
    let mut resize_pending: Option<(u32, u32)> = None;
    let mut last_frame      = Instant::now();
    let mut last_cursor_pos: Option<(f64, f64)> = None;
    let mut skip_next_cursor = false;
    let mut screenshot_requested = false;
    let mut screenshot_counter = 0u32;

    // Grab the mouse immediately on start
    mouse_captured = true;
    camera.mouse_look = true;
    window.set_cursor_visible(false);
    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
    let _ = window.set_cursor_position(winit::dpi::LogicalPosition::new(
        sz.width as f64 / 2.0, sz.height as f64 / 2.0,
    ));
    skip_next_cursor = true;

    // ── Event loop ────────────────────────────────────────────────────────────
    event_loop.run(move |event, elwt| {
        use winit::event::*;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                unsafe { renderer.device.device_wait_idle().unwrap() };
                for slot in delete_queue.iter_mut() {
                    for mut m in slot.drain(..) {
                        m.destroy(&renderer.device, &renderer.memory);
                    }
                }
                if let Some(mut m) = overlay_mesh.take() {
                    m.destroy(&renderer.device, &renderer.memory);
                }
                if let Some(mut m) = crosshair_mesh.take() {
                    m.destroy(&renderer.device, &renderer.memory);
                }
                solid_mesh.destroy(&renderer.device, &renderer.memory);
                wire_mesh.destroy(&renderer.device, &renderer.memory);
                std::process::exit(0);
            }

            // ── Route ALL window events to egui first ─────────────────────────
            Event::WindowEvent { ref event, .. }
                if !matches!(event, WindowEvent::RedrawRequested
                    | WindowEvent::CloseRequested
                    | WindowEvent::Resized(_)
                    | WindowEvent::Focused(_))
            => {
                // Toggle ui_mode on Right Alt press
                if let WindowEvent::KeyboardInput { event: ref key_event, .. } = event {
                    if let PhysicalKey::Code(KeyCode::AltRight) = key_event.physical_key {
                        if key_event.state == winit::event::ElementState::Pressed
                            && !key_event.repeat
                        {
                            ui_mode = !ui_mode;
                            if ui_mode {
                                // Enter UI mode — release mouse
                                mouse_captured = false;
                                camera.mouse_look = false;
                                last_cursor_pos = None;
                                skip_next_cursor = false;
                                window.set_cursor_visible(true);
                                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                            } else {
                                // Leave UI mode — re-capture mouse
                                mouse_captured = true;
                                camera.mouse_look = true;
                                last_cursor_pos = None;
                                skip_next_cursor = true;
                                window.set_cursor_visible(false);
                                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                                let sz = window.inner_size();
                                let _ = window.set_cursor_position(winit::dpi::LogicalPosition::new(
                                    sz.width as f64 / 2.0, sz.height as f64 / 2.0,
                                ));
                            }
                        }
                    }
                }

                // Pass event to egui
                let egui_response = egui_winit_state.on_window_event(&window, event);

                // If egui consumed the event, don't pass to editor/camera
                if egui_response.consumed {
                    // Still need to handle Resize events etc. but those are excluded above
                } else if !ui_mode {
                    // Pass to the game's input system as before
                    match event {
                        WindowEvent::KeyboardInput { event: key_event, .. } => {
                            if let PhysicalKey::Code(code) = key_event.physical_key {
                                inp.handle_key(code, key_event.state);
                            }
                        }
                        WindowEvent::MouseInput { button, state, .. } => {
                            inp.handle_mouse_button(*button, *state);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            inp.handle_cursor_moved(position.x, position.y);
                            if mouse_captured {
                                if skip_next_cursor {
                                    skip_next_cursor = false;
                                    last_cursor_pos = Some((position.x, position.y));
                                } else {
                                    if let Some((lx, ly)) = last_cursor_pos {
                                        inp.handle_raw_mouse_motion(position.x - lx, position.y - ly);
                                    }
                                    last_cursor_pos = Some((position.x, position.y));
                                    skip_next_cursor = true;
                                    let sz = window.inner_size();
                                    let cx = sz.width as f64 / 2.0;
                                    let cy = sz.height as f64 / 2.0;
                                    let _ = window.set_cursor_position(winit::dpi::LogicalPosition::new(cx, cy));
                                }
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let scroll = match delta {
                                MouseScrollDelta::LineDelta(_, y) => *y,
                                MouseScrollDelta::PixelDelta(p)   => p.y as f32 * 0.1,
                            };
                            inp.handle_scroll(scroll);
                        }
                        _ => {}
                    }
                }
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(sz), ..
            } => {
                if sz.width > 0 && sz.height > 0 {
                    resize_pending = Some((sz.width, sz.height));
                }
            }

            // ── Focus lost → release mouse capture ───────────────────────────
            Event::WindowEvent {
                event: WindowEvent::Focused(false), ..
            } => {
                if mouse_captured {
                    mouse_captured = false;
                    camera.mouse_look = false;
                    last_cursor_pos = None;
                    skip_next_cursor = false;
                    window.set_cursor_visible(true);
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                }
            }

            // ── Main frame ────────────────────────────────────────────────────
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested, ..
            } => {
                if let Some((w, h)) = resize_pending.take() {
                    renderer.resize(w, h);
                }

                let now = Instant::now();
                let dt  = (now - last_frame).as_secs_f32().min(0.1);
                last_frame = now;

                // ── FPS counter ───────────────────────────────────────────────
                frame_count += 1;
                let elapsed = fps_timer.elapsed().as_secs_f32();
                if elapsed >= 0.5 {
                    current_fps = frame_count as f32 / elapsed;
                    frame_count = 0;
                    fps_timer = std::time::Instant::now();
                }

                // ── Process global input ──────────────────────────────────────

                // Toggle wireframe (only when NOT holding F for face edit)
                if inp.just_pressed(KeyCode::KeyF) && !editor.edit_mode {
                    show_wire = !show_wire;
                    renderer.wireframe = show_wire;
                }

                // F12: request screenshot
                if inp.just_pressed(KeyCode::F12) {
                    screenshot_requested = true;
                }

                // Escape: clean up ALL GPU resources and quit
                if inp.just_pressed(KeyCode::Escape) {
                    unsafe { renderer.device.device_wait_idle().unwrap() };
                    // Flush all pending GPU mesh deletions
                    for slot in delete_queue.iter_mut() {
                        for mut m in slot.drain(..) {
                            m.destroy(&renderer.device, &renderer.memory);
                        }
                    }
                    if let Some(mut m) = overlay_mesh.take() {
                        m.destroy(&renderer.device, &renderer.memory);
                    }
                    if let Some(mut m) = crosshair_mesh.take() {
                        m.destroy(&renderer.device, &renderer.memory);
                    }
                    // Destroy main world meshes
                    solid_mesh.destroy(&renderer.device, &renderer.memory);
                    wire_mesh.destroy(&renderer.device, &renderer.memory);
                    // Force exit to avoid gpu-allocator Drop panic on leaked internal state
                    std::process::exit(0);
                }

                // Left or right click to re-grab (only when not in edit mode, or always for look)
                if (inp.lmb_just_pressed || inp.rmb_just_pressed) && !mouse_captured {
                    mouse_captured = true;
                    camera.mouse_look = true;
                    last_cursor_pos = None;
                    skip_next_cursor = true;
                    window.set_cursor_visible(false);
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    let sz = window.inner_size();
                    let _ = window.set_cursor_position(winit::dpi::LogicalPosition::new(
                        sz.width as f64 / 2.0, sz.height as f64 / 2.0,
                    ));
                }

                // Mouse look
                if mouse_captured {
                    let dx = inp.raw_mouse_dx as f32;
                    let dy = inp.raw_mouse_dy as f32;
                    if dx != 0.0 || dy != 0.0 {
                        camera.apply_mouse(dx, dy);
                    }
                }

                // Speed scroll (only when not in edit mode — edit mode uses scroll for editing)
                if !editor.edit_mode && inp.scroll_y != 0.0 {
                    camera.speed = (camera.speed * (1.0 + inp.scroll_y * 0.15)).clamp(5.0, 16384.0);
                }

                // WASD movement — with Shift=sprint
                let dir = inp.wasd_dir();
                if dir[0] != 0.0 || dir[1] != 0.0 || dir[2] != 0.0 {
                    let sprint = inp.shift;
                    camera.update_from_wasd(dir, dt, sprint);
                }

                // ── Editor update ─────────────────────────────────────────────
                let mut mesh_needs_rebuild = editor.update(&inp, &camera);

                // ── File operations ──────────────────────────────────────────
                if editor.request_save {
                    editor.request_save = false;
                    handle_save(&editor);
                }
                if editor.request_load {
                    editor.request_load = false;
                    if let Some(rebuild) = handle_load(&mut editor, &mut renderer) {
                        mesh_needs_rebuild = rebuild;
                    }
                }
                if editor.request_newmap {
                    editor.request_newmap = false;
                    handle_newmap(&mut editor);
                    mesh_needs_rebuild = true;
                }
                if editor.request_export_glb {
                    editor.request_export_glb = false;
                    handle_export_glb(&editor);
                }
                if editor.request_export_fbx {
                    editor.request_export_fbx = false;
                    handle_export_fbx(&editor);
                }

                inp.flush();

                // ── Rebuild world mesh if edited ──────────────────────────────
                if mesh_needs_rebuild {
                    // Wait for all GPU work to finish before destroying in-use meshes
                    unsafe { renderer.device.device_wait_idle().unwrap() };

                    // Destroy old meshes
                    solid_mesh.destroy(&renderer.device, &renderer.memory);
                    wire_mesh.destroy(&renderer.device, &renderer.memory);

                    // Rebuild
                    let (verts, idxs) = build_mesh_with_textures(&editor.edit_world.world, Some(&editor.tex_registry));
                    let gpu_verts: &[Vertex] = bytemuck::cast_slice(&verts);
                    solid_mesh = renderer.upload_mesh(gpu_verts, &idxs);

                    let (wverts, widxs) = build_wireframe(&editor.edit_world.world);
                    let gpu_wverts: &[Vertex] = bytemuck::cast_slice(&wverts);
                    wire_mesh = renderer.upload_mesh(gpu_wverts, &widxs);

                    vert_count = verts.len();
                    tri_count = idxs.len() / 3;
                    println!("Mesh rebuilt: {} verts, {} tris", vert_count, tri_count);
                }

                // ── Flush old GPU meshes (delayed by 3 frames for safety) ──
                {
                    let slot = delete_frame % 3;
                    for mut m in delete_queue[slot].drain(..) {
                        m.destroy(&renderer.device, &renderer.memory);
                    }
                    delete_frame = delete_frame.wrapping_add(1);
                }

                // ── Rebuild editor overlay ────────────────────────────────────
                if let Some(old) = overlay_mesh.take() {
                    delete_queue[delete_frame % 3].push(old);
                }

                if editor.edit_mode {
                    let (ov_verts, ov_idxs) = editor.build_overlay();
                    if !ov_verts.is_empty() {
                        overlay_mesh = Some(renderer.upload_mesh(&ov_verts, &ov_idxs));
                    }
                } else {
                    // Edit mode off: also queue crosshair for deletion
                    if let Some(old_ch) = crosshair_mesh.take() {
                        delete_queue[delete_frame % 3].push(old_ch);
                    }
                }

                // ── Build egui frame ──────────────────────────────────────────
                let egui_input = egui_winit_state.take_egui_input(&window);
                let egui_output = egui_ctx.run(egui_input, |ctx| {
                    build_editor_ui(
                        ctx, &mut editor, &camera, ui_mode, show_wire,
                        vert_count, tri_count, current_fps,
                    );
                });
                let egui_primitives = egui_ctx.tessellate(
                    egui_output.shapes,
                    egui_output.pixels_per_point,
                );

                // Handle platform output (cursor changes etc.)
                egui_winit_state.handle_platform_output(&window, egui_output.platform_output);

                // ── Render ────────────────────────────────────────────────────
                let sz = window.inner_size();
                let cam_uniform = camera.build_uniform(sz.width, sz.height);

                let Some((cmd, frame, img_idx)) = renderer.begin_frame() else {
                    resize_pending = Some((sz.width, sz.height));
                    return;
                };

                renderer.upload_camera(frame, &cam_uniform);
                renderer.begin_rendering(cmd, img_idx);

                // Draw solid geometry
                renderer.bind_pipeline(cmd, frame);
                renderer.draw_mesh(cmd, &solid_mesh, &glam::Mat4::IDENTITY);

                // Draw wireframe overlay (map edges)
                if show_wire {
                    renderer.bind_line_pipeline(cmd, frame);
                    renderer.draw_mesh(cmd, &wire_mesh, &glam::Mat4::IDENTITY);
                }

                // Draw editor overlay (cursor, selection, grid)
                if let Some(ref ov) = overlay_mesh {
                    renderer.bind_line_pipeline(cmd, frame);
                    renderer.draw_mesh(cmd, ov, &glam::Mat4::IDENTITY);
                }

                // Draw crosshair in edit mode — rendered as a 3D cross
                // positioned in front of the camera
                if editor.edit_mode {
                    if let Some(old_ch) = crosshair_mesh.take() {
                        delete_queue[delete_frame % 3].push(old_ch);
                    }
                    let cross = build_crosshair_mesh(&camera, sz.width, sz.height);
                    if !cross.0.is_empty() {
                        let ch = renderer.upload_mesh(&cross.0, &cross.1);
                        renderer.bind_line_pipeline(cmd, frame);
                        renderer.draw_mesh(cmd, &ch, &glam::Mat4::IDENTITY);
                        crosshair_mesh = Some(ch);
                    }
                }

                // Handle egui texture updates (font atlas etc.)
                if !egui_output.textures_delta.set.is_empty()
                    || !egui_output.textures_delta.free.is_empty()
                {
                    egui_renderer.update_texture(
                        &renderer.device, &renderer.memory,
                        renderer.commands.pool, renderer.graphics_queue,
                        &egui_output.textures_delta,
                    );
                }

                // Draw egui UI overlay
                let screen_size = [sz.width as f32, sz.height as f32];
                egui_renderer.render(&renderer.device, &renderer.memory, cmd, frame, &egui_primitives, screen_size);

                renderer.end_rendering(cmd, img_idx);
                if !renderer.end_frame(frame, img_idx) {
                    resize_pending = Some((sz.width, sz.height));
                }

                // ── F12 screenshot capture ───────────────────────────────
                if screenshot_requested {
                    screenshot_requested = false;
                    screenshot_counter += 1;

                    let (w, h, mut rgba) = renderer.capture_screenshot(img_idx);

                    // Burn camera info text onto the image
                    let cam_info = format!(
                        "pos=({:.1}, {:.1}, {:.1})  yaw={:.1}  pitch={:.1}  fwd=({:.3}, {:.3}, {:.3})",
                        camera.pos.x, camera.pos.y, camera.pos.z,
                        camera.yaw.to_degrees(), camera.pitch.to_degrees(),
                        camera.forward().x, camera.forward().y, camera.forward().z,
                    );
                    screenshot::burn_text(&mut rgba, w, h, 4, 4, &cam_info, [255, 255, 0, 255]);

                    // Save as PNG
                    let filename = format!(
                        "screenshot_{:04}_x{:.0}_y{:.0}_z{:.0}.png",
                        screenshot_counter, camera.pos.x, camera.pos.y, camera.pos.z,
                    );
                    match image::save_buffer(
                        &filename, &rgba, w, h, image::ColorType::Rgba8,
                    ) {
                        Ok(()) => println!("Screenshot saved: {filename}"),
                        Err(e) => eprintln!("Screenshot failed: {e}"),
                    }
                    println!("  Camera: {cam_info}");
                }

                window.request_redraw();
            }

            _ => {}
        }
    }).unwrap();
}

// ── egui UI builder ──────────────────────────────────────────────────────────

fn build_editor_ui(
    ctx: &egui::Context,
    editor: &mut editor::EditorState,
    _camera: &bbc_renderer::FlyCamera,
    ui_mode: bool,
    _show_wire: bool,
    vert_count: usize,
    tri_count: usize,
    fps: f32,
) {
    // Semi-transparent dark frame for panels
    let panel_frame = egui::Frame::none()
        .fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 200))
        .inner_margin(egui::Margin::same(4.0));

    // ── Top menu bar ─────────────────────────────────────────────────────────
    egui::TopBottomPanel::top("menu_bar")
        .frame(panel_frame)
        .show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.add(egui::Button::new("New          Ctrl+N")).clicked() {
                        editor.request_newmap = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Open         Ctrl+O")).clicked() {
                        editor.request_load = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Save         Ctrl+S")).clicked() {
                        editor.request_save = true;
                        ui.close_menu();
                    }
                    if ui.button("Save As...").clicked() {
                        editor.request_save = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Export GLB    Ctrl+E")).clicked() {
                        editor.request_export_glb = true;
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Export FBX    Ctrl+Shift+E")).clicked() {
                        editor.request_export_fbx = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        println!("Quit requested via menu");
                        std::process::exit(0);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui.add(egui::Button::new("Undo         Z")).clicked() {
                        println!("[menu] Undo");
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Redo         I")).clicked() {
                        println!("[menu] Redo");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Copy         C")).clicked() {
                        println!("[menu] Copy");
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Paste        V")).clicked() {
                        println!("[menu] Paste");
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Delete       Del")).clicked() {
                        println!("[menu] Delete");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Flip         X")).clicked() {
                        println!("[menu] Flip");
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui.add(egui::Button::new("Wireframe    F")).clicked() {
                        println!("[menu] Toggle wireframe");
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Edit Mode    E")).clicked() {
                        println!("[menu] Toggle edit mode");
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label(format!("Grid Size: {} (2^{})", editor.grid_size, editor.grid_power));
                });

                ui.menu_button("Texture", |ui| {
                    ui.label("1+Scroll = Slot");
                    ui.label("2+Scroll = Rotate");
                    ui.label("3+Scroll = Scale");
                    ui.label("4+Scroll = Offset");
                    ui.label("5+Scroll = Material");
                });
            });
        });

    // ── Bottom status bar ────────────────────────────────────────────────────
    let status_frame = egui::Frame::none()
        .fill(egui::Color32::from_rgba_unmultiplied(20, 20, 20, 220))
        .inner_margin(egui::Margin::same(4.0));

    egui::TopBottomPanel::bottom("status_bar")
        .frame(status_frame)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Edit mode indicator
                let edit_text = if editor.edit_mode { "Edit: ON" } else { "Edit: OFF" };
                let edit_color = if editor.edit_mode {
                    egui::Color32::from_rgb(100, 255, 100)
                } else {
                    egui::Color32::from_rgb(180, 180, 180)
                };
                ui.colored_label(edit_color, edit_text);
                ui.separator();

                // UI mode indicator
                if ui_mode {
                    ui.colored_label(egui::Color32::from_rgb(255, 200, 50), "UI Mode (RAlt)");
                    ui.separator();
                }

                // Grid
                ui.label(format!("Grid: {} (2^{})", editor.grid_size, editor.grid_power));
                ui.separator();

                // Selection info
                if editor.have_sel {
                    let sel = &editor.selection;
                    ui.label(format!(
                        "Sel: [{},{},{}] {}x{}x{}",
                        sel.origin[0], sel.origin[1], sel.origin[2],
                        sel.size[0], sel.size[1], sel.size[2],
                    ));
                    ui.separator();
                }

                // Texture slot info
                let num_slots = editor.tex_registry.num_slots();
                ui.label(format!("Slots: {}", num_slots));
                ui.separator();

                // Mesh stats
                ui.label(format!("{} verts, {} tris", vert_count, tri_count));
                ui.separator();

                // Map name
                let map_name = editor.current_map_path.as_deref()
                    .and_then(|p| std::path::Path::new(p).file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("untitled");
                ui.label(map_name);

                // FPS (right-aligned)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("FPS: {:.0}", fps));
                });
            });
        });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a small 3D crosshair positioned just in front of the camera.
/// This avoids needing a screen-space pipeline — we place a tiny cross
/// at (camera.pos + 0.5 * forward) in world space.
fn build_crosshair_mesh(camera: &FlyCamera, width: u32, height: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idxs = Vec::new();

    let fwd = camera.forward();
    let right = fwd.cross(glam::Vec3::Y).normalize_or_zero();
    let up = right.cross(fwd).normalize_or_zero();

    // Place crosshair just past the near plane (0.5) so it's always visible
    let dist = 0.6f32;
    let center = camera.pos + fwd * dist;

    // Size: ~15px at 900 height → in world units at dist=0.5, roughly:
    // size_world = dist * tan(fov/2) * (15/height*2) ≈ 0.5 * 0.577 * 0.033 ≈ 0.01
    let aspect = width as f32 / height as f32;
    let half_fov = (60.0f32 / 2.0).to_radians(); // 30° half-fov (60° total, matches camera)
    let pixel_size = dist * half_fov.tan() * 2.0 / height as f32;
    let cross_size = pixel_size * 12.0; // 12 pixels

    let color = [1.0f32, 1.0, 1.0, 0.9];

    // Horizontal line
    let h0 = center - right * cross_size;
    let h1 = center + right * cross_size;
    let base = verts.len() as u32;
    verts.push(Vertex::new(h0.to_array(), [0.0; 3], [0.0; 2], color));
    verts.push(Vertex::new(h1.to_array(), [0.0; 3], [0.0; 2], color));
    idxs.push(base); idxs.push(base + 1);

    // Vertical line
    let v0 = center - up * cross_size;
    let v1 = center + up * cross_size;
    let base = verts.len() as u32;
    verts.push(Vertex::new(v0.to_array(), [0.0; 3], [0.0; 2], color));
    verts.push(Vertex::new(v1.to_array(), [0.0; 3], [0.0; 2], color));
    idxs.push(base); idxs.push(base + 1);

    (verts, idxs)
}

// ── File operation handlers ───────────────────────────────────────────────────

fn handle_save(editor: &editor::EditorState) {
    use cube_world::save_ogz;

    let default_path = editor.current_map_path.as_deref().unwrap_or("untitled.ogz");
    let dialog = rfd::FileDialog::new()
        .set_title("Save OGZ Map")
        .set_file_name(
            std::path::Path::new(default_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled.ogz"),
        )
        .add_filter("Cube2 Map", &["ogz"]);

    if let Some(path) = dialog.save_file() {
        let path_str = path.display().to_string();
        match save_ogz(&path_str, &editor.edit_world.world) {
            Ok(()) => println!("Saved: {}", path_str),
            Err(e) => eprintln!("Save failed: {}", e),
        }
    }
}

fn handle_load(
    editor: &mut editor::EditorState,
    renderer: &mut Renderer,
) -> Option<bool> {
    let dialog = rfd::FileDialog::new()
        .set_title("Open OGZ Map")
        .add_filter("Cube2 Map", &["ogz"]);

    if let Some(path) = dialog.pick_file() {
        let path_str = path.display().to_string();
        match cube_world::load_ogz(&path_str) {
            Ok(world) => {
                println!("Loaded: {} ({} units), {} entities",
                    path.display(), world.world_size(), world.entities.len());
                editor.edit_world = cube_world::EditWorld::new(world);
                editor.current_map_path = Some(path_str.clone());
                editor.have_sel = false;
                editor.clipboard = None;
                // Load textures
                load_map_textures(&path_str, editor, renderer);
                Some(true) // needs rebuild
            }
            Err(e) => {
                eprintln!("Load failed: {}", e);
                None
            }
        }
    } else {
        None
    }
}

fn handle_newmap(editor: &mut editor::EditorState) {
    use cube_world::octree::{Cube, OctreeWorld};

    let world_scale = 10u32; // 1024 units, good default
    let mut root: [Cube; 8] = Default::default();
    for i in 0..8usize {
        let z_high = (i >> 2) & 1 == 1;
        root[i] = if z_high { Cube::empty() } else { Cube::solid() };
    }
    let world = OctreeWorld {
        root: Box::new(root),
        world_scale,
        entities: vec![],
        ogz_version: 33,
    };

    editor.edit_world = cube_world::EditWorld::new(world);
    editor.current_map_path = None;
    editor.packages_dir = None;
    editor.have_sel = false;
    editor.clipboard = None;
    editor.tex_registry = editor::EditorState::make_default_registry_static();
    println!("New map: {}×{} units", 1 << world_scale, 1 << world_scale);
}

fn handle_export_glb(editor: &editor::EditorState) {
    let default_name = editor.current_map_path.as_deref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");

    let dialog = rfd::FileDialog::new()
        .set_title("Export GLB")
        .set_file_name(&format!("{}.glb", default_name))
        .add_filter("glTF Binary", &["glb"]);

    if let Some(path) = dialog.save_file() {
        let path_str = path.display().to_string();
        let tex_base = editor.packages_dir.as_deref();
        match cube_world::export_glb(
            &path_str,
            &editor.edit_world.world,
            Some(&editor.tex_registry),
            tex_base,
        ) {
            Ok(()) => println!("Exported GLB: {}", path_str),
            Err(e) => eprintln!("Export failed: {}", e),
        }
    }
}

fn handle_export_fbx(editor: &editor::EditorState) {
    let default_name = editor.current_map_path.as_deref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");

    let dialog = rfd::FileDialog::new()
        .set_title("Export FBX")
        .set_file_name(&format!("{}.fbx", default_name))
        .add_filter("Autodesk FBX", &["fbx"]);

    if let Some(path) = dialog.save_file() {
        let path_str = path.display().to_string();
        let tex_base = editor.packages_dir.as_deref();
        match cube_world::export_fbx(
            &path_str,
            &editor.edit_world.world,
            Some(&editor.tex_registry),
            tex_base,
        ) {
            Ok(()) => println!("Exported FBX: {}", path_str),
            Err(e) => eprintln!("FBX export failed: {}", e),
        }
    }
}

/// Load map textures from the .cfg file alongside the .ogz.
/// Parses the config, loads JPEG/PNG images, uploads to a GPU texture array.
fn load_map_textures(
    ogz_path: &str,
    editor: &mut editor::EditorState,
    renderer: &mut Renderer,
) {
    use cube_world::load_texture_config;

    let ogz = std::path::Path::new(ogz_path);

    // Find the .cfg file (same name as .ogz)
    let cfg_path = ogz.with_extension("cfg");
    if !cfg_path.exists() {
        println!("No .cfg found at {} — skipping texture load", cfg_path.display());
        return;
    }

    // Find the packages root directory (parent of "base" typically)
    // Sauerbraten layout: .../packages/base/mapname.ogz
    // Textures are at:    .../packages/dg/texture.jpg
    let packages_dir = ogz.parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| std::path::Path::new("."));

    println!("Loading textures from: {}", packages_dir.display());
    editor.packages_dir = Some(packages_dir.display().to_string());

    // Read and parse the .cfg — handle `exec` includes
    let cfg_text = parse_cfg_with_exec(&cfg_path, packages_dir);

    // Reset registry and parse
    editor.tex_registry = cube_world::TextureRegistry::new();
    load_texture_config(&mut editor.tex_registry, &cfg_text);

    let num_slots = editor.tex_registry.num_slots();
    println!("Parsed {} texture slots from {}", num_slots, cfg_path.display());

    if num_slots <= 2 { return; } // only sky + default, nothing to load

    // Load images and build texture array layers
    const TEX_SIZE: u32 = 256;
    let mut layers: Vec<Vec<u8>> = Vec::new();
    let mut loaded_count = 0u32;
    let mut failed_count = 0u32;

    for slot_idx in 0..num_slots {
        let slot = &editor.tex_registry.slots[slot_idx];
        let diffuse_path = slot.textures.first().map(|t| t.path.clone()).unwrap_or_default();

        if diffuse_path.is_empty() {
            // No texture path — generate a solid color layer
            layers.push(generate_fallback_layer(TEX_SIZE, slot_idx));
            continue;
        }

        // Resolve the texture path relative to packages dir
        let full_path = packages_dir.join(&diffuse_path);

        match load_and_resize_image(&full_path, TEX_SIZE) {
            Some(rgba) => {
                let layer_idx = layers.len() as u32;
                layers.push(rgba);
                // Update the slot's layer index and mark as loaded
                editor.tex_registry.slots[slot_idx].textures[0].layer = layer_idx;
                editor.tex_registry.slots[slot_idx].textures[0].width = TEX_SIZE;
                editor.tex_registry.slots[slot_idx].textures[0].height = TEX_SIZE;
                editor.tex_registry.slots[slot_idx].loaded = true;
                loaded_count += 1;
            }
            None => {
                // Failed to load — use fallback
                let layer_idx = layers.len() as u32;
                layers.push(generate_fallback_layer(TEX_SIZE, slot_idx));
                editor.tex_registry.slots[slot_idx].textures[0].layer = layer_idx;
                editor.tex_registry.slots[slot_idx].loaded = true; // still mark loaded so it renders
                failed_count += 1;
            }
        }
    }

    println!("Textures loaded: {} ok, {} fallback, {} total layers",
        loaded_count, failed_count, layers.len());

    if !layers.is_empty() {
        renderer.upload_texture_array(TEX_SIZE, &layers);
    }
}

/// Parse a Cube2 .cfg file, recursively processing `exec` directives.
fn parse_cfg_with_exec(cfg_path: &std::path::Path, packages_dir: &std::path::Path) -> String {
    let text = match std::fs::read_to_string(cfg_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to read {}: {}", cfg_path.display(), e);
            return String::new();
        }
    };

    let mut result = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("exec ") || trimmed.starts_with("exec\t") {
            // exec "packages/dg/package.cfg"
            let path_str = trimmed[5..].trim().trim_matches('"');
            // Resolve relative to the sauerbraten root (parent of packages)
            let exec_path = if path_str.starts_with("packages/") {
                packages_dir.join(&path_str["packages/".len()..])
            } else {
                packages_dir.join(path_str)
            };
            if exec_path.exists() {
                let sub = parse_cfg_with_exec(&exec_path, packages_dir);
                result.push_str(&sub);
                result.push('\n');
            } else {
                eprintln!("  exec not found: {}", exec_path.display());
            }
        } else if trimmed == "texturereset" {
            // Skip texturereset — our registry handles slot numbering
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

/// Load an image file and resize to tex_size × tex_size RGBA.
fn load_and_resize_image(path: &std::path::Path, tex_size: u32) -> Option<Vec<u8>> {
    let img = image::open(path).ok()?;
    let resized = img.resize_exact(tex_size, tex_size, image::imageops::FilterType::Triangle);
    let rgba = resized.to_rgba8();
    Some(rgba.into_raw())
}

/// Generate a solid-color fallback texture layer for a missing texture.
fn generate_fallback_layer(tex_size: u32, slot_idx: usize) -> Vec<u8> {
    let npixels = (tex_size * tex_size) as usize;
    let mut data = vec![0u8; npixels * 4];

    // Use golden-ratio hue for distinct colors
    let hue = (slot_idx as f32 * 0.618033988749895) % 1.0;
    let h = hue * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let (r, g, b) = match i % 6 {
        0 => (0.7f32, 0.7 * f, 0.2),
        1 => (0.7 * (1.0 - f), 0.7, 0.2),
        2 => (0.2, 0.7, 0.7 * f),
        3 => (0.2, 0.7 * (1.0 - f), 0.7),
        4 => (0.7 * f, 0.2, 0.7),
        _ => (0.7, 0.2, 0.7 * (1.0 - f)),
    };

    // Checkerboard pattern
    for y in 0..tex_size {
        for x in 0..tex_size {
            let checker = ((x / 32) + (y / 32)) % 2 == 0;
            let scale = if checker { 1.0f32 } else { 0.7 };
            let idx = ((y * tex_size + x) * 4) as usize;
            data[idx]     = (r * scale * 255.0) as u8;
            data[idx + 1] = (g * scale * 255.0) as u8;
            data[idx + 2] = (b * scale * 255.0) as u8;
            data[idx + 3] = 255;
        }
    }
    data
}

fn make_test_world() -> OctreeWorld {
    let world_scale = 6u32;
    let mut root: [Cube; 8] = Default::default();
    for i in 0..8usize {
        let z_high = (i >> 2) & 1 == 1;
        root[i] = if z_high { Cube::empty() } else { Cube::solid() };
    }
    OctreeWorld { root: Box::new(root), world_scale, entities: vec![], ogz_version: 0 }
}
