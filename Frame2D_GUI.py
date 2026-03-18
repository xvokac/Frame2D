import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

from Frame_2D import Model


class Frame2DGui:
    PROBLEM_TYPES = (
        "Static",
        "Stability",
        "Dynamic - Natural frequencies and modes",
    )

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Frame2D - editor a solver")

        self.nodes = {}
        self.dof_nodes = {}
        self.sections = {}
        self.elements = {}
        self.supports = []
        self.nodal_loads = []
        self.element_loads = []
        self.problem_type = "Static"
        self.number_of_eigenvectors = 3
        self.mass = []

        self.next_node_id = 1
        self.next_dof_node_id = 1
        self.next_section_id = 1
        self.next_element_id = 1
        self.next_dof_id = 1

        self.mode = None
        self.pending_element_nodes = []
        self.pending_release_element = None

        self.view_scale = 40.0
        self.view_origin_x = 80.0
        self.view_origin_y = 620.0
        self.grid_unit = 1.0

        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Section", command=self.add_section_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Nodes", command=lambda: self.set_mode("add_node")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Elements", command=lambda: self.set_mode("add_element")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Supports", command=lambda: self.set_mode("add_support")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Nodal loads", command=lambda: self.set_mode("add_nodal_load")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Element load", command=lambda: self.set_mode("add_element_load")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Release rotation", command=lambda: self.set_mode("add_release_pick_element")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Edit inputs", command=self.open_data_editor).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Load JSON", command=self.load_json).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Save JSON", command=self.save_json).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Normalize ID", command=self.normalize_to_canvas).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Calculate", command=self.solve_and_plot).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Mode: none")
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor="w", padx=10)

        main_pane = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        canvas_frame = ttk.Frame(main_pane)
        self.canvas = tk.Canvas(canvas_frame, width=1000, height=700, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel_linux)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel_linux)

        reactions_frame = ttk.LabelFrame(main_pane, text="Results")
        reactions_frame.columnconfigure(0, weight=1)
        reactions_frame.rowconfigure(0, weight=1)

        self.reactions_text = tk.Text(reactions_frame, height=8, wrap=tk.NONE)
        y_scroll = ttk.Scrollbar(reactions_frame, orient=tk.VERTICAL, command=self.reactions_text.yview)
        x_scroll = ttk.Scrollbar(reactions_frame, orient=tk.HORIZONTAL, command=self.reactions_text.xview)
        self.reactions_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.reactions_text.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=(4, 0))
        y_scroll.grid(row=0, column=1, sticky="ns", padx=(2, 4), pady=(4, 0))
        x_scroll.grid(row=1, column=0, sticky="ew", padx=(4, 0), pady=(2, 4))

        self.reactions_text.insert("1.0", "The results will be displayed here after calculation.")
        self.reactions_text.configure(state=tk.DISABLED)

        main_pane.add(canvas_frame, weight=4)
        main_pane.add(reactions_frame, weight=1)

    def set_mode(self, mode: str):
        self.mode = mode
        self.pending_element_nodes = []
        self.pending_release_element = None
        self.status_var.set(f"Mode: {mode}")

    def to_canvas(self, x, y):
        return self.view_origin_x + x * self.view_scale, self.view_origin_y - y * self.view_scale

    def from_canvas(self, cx, cy):
        return (cx - self.view_origin_x) / self.view_scale, (self.view_origin_y - cy) / self.view_scale

    def zoom_at(self, cx, cy, factor):
        wx, wy = self.from_canvas(cx, cy)
        new_scale = self.view_scale * factor
        self.view_scale = max(8.0, min(300.0, new_scale))
        self.view_origin_x = cx - wx * self.view_scale
        self.view_origin_y = cy + wy * self.view_scale
        self.draw_scene()

    def on_mouse_wheel(self, event):
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self.zoom_at(event.x, event.y, factor)

    def on_mouse_wheel_linux(self, event):
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self.zoom_at(event.x, event.y, factor)

    def draw_scene(self):
        self.canvas.delete("all")
        self._draw_grid()
        mass_nodes = self._get_nodes_with_mass()

        for e in self.elements.values():
            ni = self.nodes[self.dof_nodes[e["i"]]["node"]]
            nj = self.nodes[self.dof_nodes[e["j"]]["node"]]
            xi, yi = self.to_canvas(ni["x"], ni["y"])
            xj, yj = self.to_canvas(nj["x"], nj["y"])
            self.canvas.create_line(xi, yi, xj, yj, fill="black", width=2)
            self._draw_release_symbol_if_needed(e, "i", xi, yi, xj, yj)
            self._draw_release_symbol_if_needed(e, "j", xj, yj, xi, yi)
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            self.canvas.create_text(mx, my - 8, text=f"E{e['id']}", fill="darkgreen")

        for n_id, node in self.nodes.items():
            x, y = self.to_canvas(node["x"], node["y"])
            if n_id in mass_nodes:
                mass_radius = 10
                self.canvas.create_oval(
                    x - mass_radius,
                    y - mass_radius,
                    x + mass_radius,
                    y + mass_radius,
                    outline="purple",
                    width=2,
                )
            r = 4
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="royalblue")
            self.canvas.create_text(x + 12, y - 10, text=f"N{n_id}", fill="blue")

        self._draw_supports()
        self._draw_nodal_loads()
        self._draw_element_loads()

    def _get_nodes_with_mass(self):
        nodes_with_mass = set()
        for item in self.mass:
            dof_id = item.get("dof_id")
            node_id = self._find_node_id_for_dof(dof_id)
            if node_id is not None:
                nodes_with_mass.add(node_id)
        return nodes_with_mass

    def _find_node_id_for_dof(self, dof_id):
        for dof_node in self.dof_nodes.values():
            if dof_id in (dof_node.get("ux"), dof_node.get("uy"), dof_node.get("rz")):
                return dof_node.get("node")
        return None

    def _draw_grid(self):
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        x0, y0 = self.from_canvas(0, height)
        x1, y1 = self.from_canvas(width, 0)

        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        unit = self.grid_unit
        max_lines = 350

        x_span_units = (x_max - x_min) / unit if unit > 0 else 0
        y_span_units = (y_max - y_min) / unit if unit > 0 else 0
        x_step_units = max(1, int(math.ceil(x_span_units / max_lines)))
        y_step_units = max(1, int(math.ceil(y_span_units / max_lines)))

        x_start = int(math.floor(x_min / unit))
        x_end = int(math.ceil(x_max / unit))
        y_start = int(math.floor(y_min / unit))
        y_end = int(math.ceil(y_max / unit))

        for xi in range(x_start, x_end + 1, x_step_units):
            wx = xi * unit
            cx, _ = self.to_canvas(wx, 0)
            if abs(wx) < 1e-12:
                color = "#8d8d8d"
                width_px = 2
                dash_pattern = (6, 4)
            elif xi % 5 == 0:
                color = "#d0d0d0"
                width_px = 1
                dash_pattern = (4, 4)
            else:
                color = "#ececec"
                width_px = 1
                dash_pattern = (2, 5)
            self.canvas.create_line(cx, 0, cx, height, fill=color, width=width_px, dash=dash_pattern)

        for yi in range(y_start, y_end + 1, y_step_units):
            wy = yi * unit
            _, cy = self.to_canvas(0, wy)
            if abs(wy) < 1e-12:
                color = "#8d8d8d"
                width_px = 2
                dash_pattern = (6, 4)
            elif yi % 5 == 0:
                color = "#d0d0d0"
                width_px = 1
                dash_pattern = (4, 4)
            else:
                color = "#ececec"
                width_px = 1
                dash_pattern = (2, 5)
            self.canvas.create_line(0, cy, width, cy, fill=color, width=width_px, dash=dash_pattern)

    def _fit_view_to_bounds(self, x_min, x_max, y_min, y_max):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1:
            width = int(self.canvas.cget("width"))
        if height <= 1:
            height = int(self.canvas.cget("height"))

        dx = max(float(x_max) - float(x_min), 1e-9)
        dy = max(float(y_max) - float(y_min), 1e-9)

        margin_px = 40.0
        usable_w = max(width - 2 * margin_px, 80.0)
        usable_h = max(height - 2 * margin_px, 80.0)

        self.view_scale = max(8.0, min(300.0, min(usable_w / dx, usable_h / dy)))
        self.view_origin_x = margin_px - float(x_min) * self.view_scale
        self.view_origin_y = margin_px + float(y_max) * self.view_scale

    def _draw_supports(self):
        for support in self.supports:
            dof_node = self.dof_nodes.get(support.get("node"))
            if not dof_node:
                continue
            node = self.nodes.get(dof_node["node"])
            if not node:
                continue
            x, y = self.to_canvas(node["x"], node["y"])
            ux = bool(support.get("ux"))
            uy = bool(support.get("uy"))
            rz = bool(support.get("rz"))

            if ux and uy and rz:
                self.canvas.create_rectangle(x - 8, y + 8, x + 8, y - 8, outline="firebrick", width=2)
            elif ux and uy:
                self.canvas.create_polygon(x, y , x - 10, y + 16, x + 10, y + 16, outline="firebrick", fill="", width=2)
            elif uy:
                self.canvas.create_polygon(x, y , x - 10, y + 12, x + 10, y + 12, outline="firebrick", fill="", width=2)
                self.canvas.create_oval(x - 9, y + 12, x - 3, y + 18, outline="firebrick", width=2)
                self.canvas.create_oval(x + 3, y + 12, x + 9, y + 18, outline="firebrick", width=2)
            else:
                self.canvas.create_polygon(x, y , x - 12, y + 10, x - 12, y - 10, outline="firebrick", fill="", width=2)
                self.canvas.create_oval(x - 12, y + 9, x - 18, y + 3, outline="firebrick", width=2)
                self.canvas.create_oval(x - 12, y - 3, x - 18, y - 9, outline="firebrick", width=2)

    def _draw_arrow(self, x1, y1, x2, y2, color="tomato", width=2):
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, arrow=tk.LAST)

    def _draw_nodal_loads(self):
        for load in self.nodal_loads:
            dof_node = self.dof_nodes.get(load.get("dof_node"))
            if not dof_node:
                continue
            node = self.nodes.get(dof_node["node"])
            if not node:
                continue

            x, y = self.to_canvas(node["x"], node["y"])
            fx = float(load.get("Fx", 0.0))
            fy = float(load.get("Fy", 0.0))
            mz = float(load.get("Mz", 0.0))

            if abs(fx) > 1e-12:
                sign = 1 if fx > 0 else -1
                self._draw_arrow(x , y , x + 36 * sign, y )
                self.canvas.create_text(x + 36 * sign, y - 10, text=f"Fx={np.abs(fx):g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(fy) > 1e-12:
                sign = 1 if fy > 0 else -1
                self._draw_arrow(x , y , x , y - 36 * sign)
                self.canvas.create_text(x , y - 46 * sign, text=f"Fy={np.abs(fy):g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(mz) > 1e-12:
                r = 12
                self.canvas.create_arc(x - r, y - r, x + r, y + r, start=30, extent=300, style=tk.ARC, outline="tomato", width=2)
                if mz < 0:
                    self._draw_arrow(x + 8, y - 8, x + 12, y )
                else:
                    self._draw_arrow(x + 8, y + 8, x + 12, y )
                self.canvas.create_text(x - 26, y - 18, text=f"Mz={np.abs(mz):g}", fill="tomato", font=("TkDefaultFont", 8))

    def _draw_element_loads(self):
        for load in self.element_loads:
            element = self.elements.get(load.get("element"))
            if not element:
                continue
            ni = self.nodes.get(self.dof_nodes.get(element["i"], {}).get("node"))
            nj = self.nodes.get(self.dof_nodes.get(element["j"], {}).get("node"))
            if not ni or not nj:
                continue

            x1, y1 = self.to_canvas(ni["x"], ni["y"])
            x2, y2 = self.to_canvas(nj["x"], nj["y"])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            ux = dx / length
            uy = dy / length
            nx = -uy
            ny = ux

            qx = float(load.get("qx", 0.0))
            qz = float(load.get("qz", 0.0))

            if abs(qx) > 1e-12:
                sign = 1 if qx > 0 else -1
                for t in (0.2, 0.4, 0.6, 0.8):
                    mx = x1 + t * dx
                    my = y1 + t * dy
                    self._draw_arrow(mx - 22 * ux * sign, my - 22 * uy * sign, mx + 22 * ux * sign, my + 22 * uy * sign)
                mx = x1 + .5 * dx
                my = y1 + .5 * dy
                self.canvas.create_text(mx + 10 * nx, my + 10 * ny, text=f"qx={np.abs(qx):g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(qz) > 1e-12:
                sign = 1 if qz > 0 else -1
                for t in (0.2, 0.4, 0.6, 0.8):
                    px = x1 + t * dx
                    py = y1 + t * dy
                    sx = px + 18 * nx * sign
                    sy = py + 18 * ny * sign
                    self._draw_arrow(sx, sy, px, py)
                tx = (x1 + x2) / 2 + 26 * nx * sign
                ty = (y1 + y2) / 2 + 26 * ny * sign
                self.canvas.create_text(tx, ty, text=f"qz={np.abs(qz):g}", fill="tomato", font=("TkDefaultFont", 8))

    def _get_primary_dof_node_id(self, node_id):
        candidates = [did for did, d in self.dof_nodes.items() if d["node"] == node_id]
        if not candidates:
            return None
        return min(candidates)

    def _draw_release_symbol_if_needed(self, element, end_key, x, y, ox, oy):
        did = element[end_key]
        dof = self.dof_nodes.get(did)
        if not dof:
            return
        primary = self._get_primary_dof_node_id(dof["node"])
        if primary is None or did == primary:
            return
        vx = ox - x
        vy = oy - y
        length = math.hypot(vx, vy)
        if length < 1e-9:
            return
        ux = vx / length
        uy = vy / length
        cx = x + ux * 10
        cy = y + uy * 10
        r = 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="white", outline="purple", width=2)

    def find_nearest_node(self, cx, cy, tol=12):
        best = None
        best_d = 1e9
        for nid, node in self.nodes.items():
            x, y = self.to_canvas(node["x"], node["y"])
            d = math.hypot(cx - x, cy - y)
            if d < best_d:
                best_d = d
                best = nid
        return best if best_d <= tol else None

    def find_nearest_element(self, cx, cy, tol=10):
        best = None
        best_d = 1e9
        for eid, elem in self.elements.items():
            ni = self.nodes[self.dof_nodes[elem["i"]]["node"]]
            nj = self.nodes[self.dof_nodes[elem["j"]]["node"]]
            x1, y1 = self.to_canvas(ni["x"], ni["y"])
            x2, y2 = self.to_canvas(nj["x"], nj["y"])
            d = self._point_to_segment_distance(cx, cy, x1, y1, x2, y2)
            if d < best_d:
                best_d = d
                best = eid
        return best if best_d <= tol else None

    @staticmethod
    def _point_to_segment_distance(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx, qy = x1 + t * dx, y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def add_section_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Nová section")

        labels = ["E", "A", "I"]
        entries = {}
        for i, lbl in enumerate(labels):
            ttk.Label(win, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ent = ttk.Entry(win)
            ent.grid(row=i, column=1, padx=6, pady=4)
            entries[lbl] = ent

        def save():
            try:
                E = float(entries["E"].get())
                A = float(entries["A"].get())
                I = float(entries["I"].get())
            except ValueError:
                messagebox.showerror("Error," "E, A, I must be numbers.")
                return

            sid = self.next_section_id
            self.next_section_id += 1
            self.sections[sid] = {"id": sid, "E": E, "A": A, "I": I}
            win.destroy()

        ttk.Button(win, text="Save", command=save).grid(row=4, column=0, columnspan=2, pady=8)

    def add_node_dialog(self, x, y):
        win = tk.Toplevel(self.root)
        win.title("Nový node")

        ttk.Label(win, text="x").grid(row=0, column=0, padx=6, pady=4)
        ex = ttk.Entry(win)
        ex.insert(0, f"{x:.3f}")
        ex.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(win, text="y").grid(row=1, column=0, padx=6, pady=4)
        ey = ttk.Entry(win)
        ey.insert(0, f"{y:.3f}")
        ey.grid(row=1, column=1, padx=6, pady=4)

        def save():
            try:
                nx = float(ex.get())
                ny = float(ey.get())
            except ValueError:
                messagebox.showerror("Error", "Coordinates must be numbers.")
                return

            nid = self.next_node_id
            self.next_node_id += 1
            self.nodes[nid] = {"id": nid, "x": nx, "y": ny}

            dof_id = self.next_dof_node_id
            self.next_dof_node_id += 1
            ux = self.next_dof_id
            uy = self.next_dof_id + 1
            rz = self.next_dof_id + 2
            self.next_dof_id += 3
            self.dof_nodes[dof_id] = {"id": dof_id, "node": nid, "ux": ux, "uy": uy, "rz": rz}

            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Save", command=save).grid(row=3, column=0, columnspan=2, pady=8)

    def add_element_dialog(self, n1, n2):
        if not self.sections:
            messagebox.showwarning("Section", "First enter at least one section.")
            return

        win = tk.Toplevel(self.root)
        win.title("Nový element")
        ttk.Label(win, text=f"Node {n1} -> Node {n2}").grid(row=0, column=0, columnspan=2, padx=6, pady=4)

        ttk.Label(win, text="Section").grid(row=1, column=0, padx=6, pady=4)
        cb = ttk.Combobox(win, values=[str(sid) for sid in self.sections.keys()], state="readonly")
        cb.current(0)
        cb.grid(row=1, column=1, padx=6, pady=4)

        def save():
            sid = int(cb.get())
            eid = self.next_element_id
            self.next_element_id += 1

            dn1 = self.get_primary_dof_node_for_node(n1)
            dn2 = self.get_primary_dof_node_for_node(n2)
            self.elements[eid] = {"id": eid, "i": dn1, "j": dn2, "section": sid}
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Save", command=save).grid(row=2, column=0, columnspan=2, pady=8)

    def get_primary_dof_node_for_node(self, node_id):
        for did, d in self.dof_nodes.items():
            if d["node"] == node_id:
                return did
        raise ValueError(f"DofNode for node {node_id} does not exist")

    def on_canvas_click(self, event):
        x, y = self.from_canvas(event.x, event.y)

        if self.mode == "add_node":
            self.add_node_dialog(x, y)

        elif self.mode == "add_element":
            n = self.find_nearest_node(event.x, event.y)
            if n is None:
                return
            self.pending_element_nodes.append(n)
            if len(self.pending_element_nodes) == 2:
                n1, n2 = self.pending_element_nodes
                self.pending_element_nodes = []
                if n1 == n2:
                    messagebox.showerror("Error", "The element must have 2 different nodes.")
                    return
                self.add_element_dialog(n1, n2)

        elif self.mode == "add_support":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None:
                self.add_support_dialog(n)

        elif self.mode == "add_nodal_load":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None:
                self.add_nodal_load_dialog(n)

        elif self.mode == "add_element_load":
            e = self.find_nearest_element(event.x, event.y)
            if e is not None:
                self.add_element_load_dialog(e)

        elif self.mode == "add_release_pick_element":
            e = self.find_nearest_element(event.x, event.y)
            if e is not None:
                self.pending_release_element = e
                self.mode = "add_release_pick_node"
                self.status_var.set(f"Mode: select the end of the element E{e}")

        elif self.mode == "add_release_pick_node":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None and self.pending_release_element is not None:
                self.apply_release(self.pending_release_element, n)
                self.pending_release_element = None
                self.mode = "add_release_pick_element"
                self.status_var.set("Mode: add_release_pick_element")

    def add_support_dialog(self, node_id):
        win = tk.Toplevel(self.root)
        win.title(f"Support node {node_id}")

        ux = tk.BooleanVar(value=False)
        uy = tk.BooleanVar(value=True)
        rz = tk.BooleanVar(value=False)

        ttk.Checkbutton(win, text="ux", variable=ux).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(win, text="uy", variable=uy).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(win, text="rz", variable=rz).pack(anchor="w", padx=8, pady=2)

        def save():
            dn = self.get_primary_dof_node_for_node(node_id)
            self.supports.append({"node": dn, "ux": ux.get(), "uy": uy.get(), "rz": rz.get()})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Save", command=save).pack(pady=8)

    def add_nodal_load_dialog(self, node_id):
        win = tk.Toplevel(self.root)
        win.title(f"Nodal load node {node_id}")

        names = ["Fx", "Fy", "Mz"]
        entries = {}
        for i, nm in enumerate(names):
            ttk.Label(win, text=nm).grid(row=i, column=0, padx=6, pady=4)
            e = ttk.Entry(win)
            e.insert(0, "0")
            e.grid(row=i, column=1, padx=6, pady=4)
            entries[nm] = e

        def save():
            try:
                Fx = float(entries["Fx"].get())
                Fy = float(entries["Fy"].get())
                Mz = float(entries["Mz"].get())
            except ValueError:
                messagebox.showerror("Error", "Fx, Fy, Mz must be numbers.")
                return

            dn = self.get_primary_dof_node_for_node(node_id)
            self.nodal_loads.append({"dof_node": dn, "Fx": Fx, "Fy": Fy, "Mz": Mz})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Save", command=save).grid(row=4, column=0, columnspan=2, pady=8)

    def add_element_load_dialog(self, element_id):
        win = tk.Toplevel(self.root)
        win.title(f"Element load E{element_id}")

        ttk.Label(win, text="qx").grid(row=0, column=0, padx=6, pady=4)
        ex = ttk.Entry(win)
        ex.insert(0, "0")
        ex.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(win, text="qz").grid(row=1, column=0, padx=6, pady=4)
        ez = ttk.Entry(win)
        ez.insert(0, "0")
        ez.grid(row=1, column=1, padx=6, pady=4)

        def save():
            try:
                qx = float(ex.get())
                qz = float(ez.get())
            except ValueError:
                messagebox.showerror("Error," "qx, qz must be numbers.")
                return

            self.element_loads.append({"element": element_id, "qx": qx, "qz": qz})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Save", command=save).grid(row=2, column=0, columnspan=2, pady=8)

    def apply_release(self, element_id, node_id):
        elem = self.elements[element_id]
        did_i = elem["i"]
        did_j = elem["j"]

        end = None
        if self.dof_nodes[did_i]["node"] == node_id:
            end = "i"
            base = self.dof_nodes[did_i]
        elif self.dof_nodes[did_j]["node"] == node_id:
            end = "j"
            base = self.dof_nodes[did_j]
        else:
            messagebox.showerror("Hinge", "The selected node is not at the end of the element.")
            return

        new_did = self.next_dof_node_id
        self.next_dof_node_id += 1
        new_rz = self.next_dof_id + 2
        self.next_dof_id += 3
        self.dof_nodes[new_did] = {
            "id": new_did,
            "node": node_id,
            "ux": base["ux"],
            "uy": base["uy"],
            "rz": new_rz,
        }

        if end == "i":
            elem["i"] = new_did
        else:
            elem["j"] = new_did

        self.draw_scene()

    def build_json_model(self):
        return {
            "nodes": list(self.nodes.values()),
            "dof_nodes": list(self.dof_nodes.values()),
            "sections": list(self.sections.values()),
            "elements": list(self.elements.values()),
            "supports": self.supports,
            "nodal_loads": self.nodal_loads,
            "element_loads": self.element_loads,
            "problem_type": self.problem_type,
            "number_of_eigenvectors": self.number_of_eigenvectors,
            "mass": self.mass,
        }

    def save_json(self):
        path = filedialog.asksaveasfilename(
            title="Save model", defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return

        data = self.build_json_model()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Saved", f"Model saved to:\n{path}")

    def load_json(self):
        path = filedialog.askopenfilename(title="Load model", filetypes=[("JSON", "*.json")])
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._load_data_dict(data, fit_view=True)
            messagebox.showinfo("Loaded", f"Model loaded from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Load JSON", f"Failed to load model:\n{exc}")

    def _load_data_dict(self, data, fit_view=False):
        self.nodes = {n["id"]: n for n in data.get("nodes", [])}
        self.dof_nodes = {
            d["id"]: {"id": d["id"], "node": d["node"], "ux": d["ux"], "uy": d["uy"], "rz": d["rz"]}
            for d in data.get("dof_nodes", [])
        }
        self.sections = {s["id"]: s for s in data.get("sections", [])}
        self.elements = {
            e["id"]: {"id": e["id"], "i": e["i"], "j": e["j"], "section": e["section"]}
            for e in data.get("elements", [])
        }
        self.supports = data.get("supports", [])
        self.nodal_loads = data.get("nodal_loads", [])
        self.element_loads = data.get("element_loads", [])
        self.problem_type = data.get("problem_type", "Static")
        if self.problem_type not in self.PROBLEM_TYPES:
            self.problem_type = "Static"
        self.number_of_eigenvectors = int(data.get("number_of_eigenvectors", 3))
        self.mass = data.get("mass", [])

        self.next_node_id = max(self.nodes.keys(), default=0) + 1
        self.next_dof_node_id = max(self.dof_nodes.keys(), default=0) + 1
        self.next_section_id = max(self.sections.keys(), default=0) + 1
        self.next_element_id = max(self.elements.keys(), default=0) + 1

        max_dof = 0
        for d in self.dof_nodes.values():
            max_dof = max(max_dof, d["ux"], d["uy"], d["rz"])
        self.next_dof_id = max_dof + 1

        if fit_view:
            bounds = None
            try:
                if all(k in data for k in ("x_min", "x_max", "y_min", "y_max")):
                    bounds = (float(data["x_min"]), float(data["x_max"]), float(data["y_min"]), float(data["y_max"]))
            except (TypeError, ValueError):
                bounds = None

            if bounds is None and self.nodes:
                xs = [node["x"] for node in self.nodes.values()]
                ys = [node["y"] for node in self.nodes.values()]
                bounds = (min(xs), max(xs), min(ys), max(ys))

            if bounds is not None:
                x_min, x_max, y_min, y_max = bounds
                self._fit_view_to_bounds(x_min, x_max, y_min, y_max)

        self.draw_scene()

    def normalize_to_canvas(self):
        if not self.nodes:
            messagebox.showwarning("Normalization," "There is nothing to normalize.")
            return

        tmp_path = "_frame2d_gui_temp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.build_json_model(), f, indent=2)

        try:
            model = Model.from_json(tmp_path)
            self._load_data_dict(model.to_json_data())
            messagebox.showinfo("Normalization", "The ID has been normalized and transferred to both the canvas and JSON data.")
        except Exception as exc:
            messagebox.showerror("Normalization", str(exc))

    def solve_and_plot(self):
        if not self.elements or not self.nodes:
            messagebox.showwarning("Solver", "The model must have at least nodes and elements.")
            return

        tmp_path = "_frame2d_gui_temp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.build_json_model(), f, indent=2)

        try:
            model = Model.from_json(tmp_path)

            if model.problem_type == "Dynamic - Natural frequencies and modes":
                model.solve_dynamic()
                self.show_reactions(model.format_dynamic_results())
                model.plot_all_mode_shapes(show=True)
            elif model.problem_type == "Stability":
                model.solve_stability()
                self.show_reactions(model.format_stability_results())
                model.plot_all_stability_shapes(show=True)
            else:
                model.solve()
                self._load_data_dict(model.to_json_data())
                self.show_reactions(model.format_reactions())
                model.plot_deformed_shape()
                model.plot_internal_forces("M")
                model.plot_internal_forces("V")
                model.plot_internal_forces("N")
                model.show_all_plots()
        except Exception as exc:
            messagebox.showerror("Solver error", str(exc))

    def show_reactions(self, text):
        self.reactions_text.configure(state=tk.NORMAL)
        self.reactions_text.delete("1.0", tk.END)
        self.reactions_text.insert("1.0", text)
        self.reactions_text.configure(state=tk.DISABLED)

    def open_data_editor(self):
        editor = tk.Toplevel(self.root)
        editor.title("Edit inputs")
        editor.geometry("980x560")

        notebook = ttk.Notebook(editor)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        schemas = {
            "sections": ["id", "E", "A", "I"],
            "nodes": ["id", "x", "y"],
            "elements": ["id", "i", "j", "section"],
            "supports": ["node", "ux", "uy", "rz"],
            "nodal loads": ["dof_node", "Fx", "Fy", "Mz"],
            "element load": ["element", "qx", "qz"],
            "dof nodes": ["id", "node", "ux", "uy", "rz"],
            "release rotation": ["element", "end", "node", "dof_node"],
        }

        data_tables = {
            "sections": [dict(v) for v in self.sections.values()],
            "nodes": [dict(v) for v in self.nodes.values()],
            "elements": [dict(v) for v in self.elements.values()],
            "supports": [dict(v) for v in self.supports],
            "nodal loads": [dict(v) for v in self.nodal_loads],
            "element load": [dict(v) for v in self.element_loads],
            "dof nodes": [dict(v) for v in sorted(self.dof_nodes.values(), key=lambda d: d["id"])],
            "release rotation": self._collect_release_rows(),
            "mass": [dict(v) for v in self.mass],
        }

        trees = {}

        for tab_name, columns in schemas.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=110, anchor="center")
            tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            trees[tab_name] = tree

            btns = ttk.Frame(frame)
            btns.pack(fill=tk.X, pady=6)
            if tab_name in {"dof nodes", "release rotation"}:
                ttk.Label(btns, text="The table is for informational purposes only (cannot be edited).").pack(side=tk.LEFT, padx=4)
            else:
                ttk.Button(btns, text="Add row", command=lambda n=tab_name: self._add_table_row(n, schemas[n], data_tables, trees)).pack(side=tk.LEFT, padx=4)
                ttk.Button(btns, text="Edit row", command=lambda n=tab_name: self._edit_table_row(n, schemas[n], data_tables, trees)).pack(side=tk.LEFT, padx=4)
                ttk.Button(btns, text="Delete row", command=lambda n=tab_name: self._delete_table_row(n, data_tables, trees)).pack(side=tk.LEFT, padx=4)

            self._refresh_tree(tree, columns, data_tables[tab_name])

        special_tab = ttk.Frame(notebook)
        notebook.add(special_tab, text="special")

        header = ttk.Frame(special_tab)
        header.pack(fill=tk.X, padx=4, pady=6)

        ttk.Label(header, text="problem_type").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        problem_type_var = tk.StringVar(value=self.problem_type)
        ttk.Combobox(
            header,
            textvariable=problem_type_var,
            values=self.PROBLEM_TYPES,
            state="readonly",
            width=42,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(header, text="number_of_eigenvectors").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        number_of_eigenvectors_var = tk.StringVar(value=str(self.number_of_eigenvectors))
        ttk.Entry(header, textvariable=number_of_eigenvectors_var, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(
            special_tab,
            text="Mass table (lumped masses in DOF for dynamic analysis)",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        mass_columns = ["dof_id", "mass"]
        mass_tree = ttk.Treeview(special_tab, columns=mass_columns, show="headings", height=10)
        for col in mass_columns:
            mass_tree.heading(col, text=col)
            mass_tree.column(col, width=140, anchor="center")
        mass_tree.pack(fill=tk.BOTH, expand=True, padx=8)
        self._refresh_tree(mass_tree, mass_columns, data_tables["mass"])

        mass_btns = ttk.Frame(special_tab)
        mass_btns.pack(fill=tk.X, pady=6, padx=6)
        ttk.Button(
            mass_btns,
            text="Add mass",
            command=lambda: self._add_table_row("mass", mass_columns, data_tables, {"mass": mass_tree}),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            mass_btns,
            text="Edit mass",
            command=lambda: self._edit_table_row("mass", mass_columns, data_tables, {"mass": mass_tree}),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            mass_btns,
            text="Delete mass",
            command=lambda: self._delete_table_row("mass", data_tables, {"mass": mass_tree}),
        ).pack(side=tk.LEFT, padx=4)

        def apply_changes():
            try:
                self._apply_editor_tables(
                    data_tables,
                    {
                        "problem_type": problem_type_var.get(),
                        "number_of_eigenvectors": number_of_eigenvectors_var.get(),
                    },
                )
                self.draw_scene()
                editor.destroy()
                messagebox.showinfo("Editing," "Data has been updated.")
            except Exception as exc:
                messagebox.showerror("Editace", str(exc))

        ttk.Button(editor, text="Apply changes", command=apply_changes).pack(pady=(0, 8))

    def _collect_release_rows(self):
        rows = []
        for element in self.elements.values():
            for end in ("i", "j"):
                did = element[end]
                dof = self.dof_nodes.get(did)
                if not dof:
                    continue
                primary = self._get_primary_dof_node_id(dof["node"])
                if primary is not None and did != primary:
                    rows.append({"element": element["id"], "end": end, "node": dof["node"], "dof_node": did})
        return rows

    def _refresh_tree(self, tree, columns, rows):
        tree.delete(*tree.get_children())
        for idx, row in enumerate(rows):
            tree.insert("", tk.END, iid=str(idx), values=[row.get(c, "") for c in columns])

    def _edit_row_dialog(self, title, columns, initial=None):
        win = tk.Toplevel(self.root)
        win.title(title)
        entries = {}
        initial = initial or {}
        for i, col in enumerate(columns):
            ttk.Label(win, text=col).grid(row=i, column=0, padx=6, pady=4, sticky="w")
            ent = ttk.Entry(win)
            ent.insert(0, str(initial.get(col, "")))
            ent.grid(row=i, column=1, padx=6, pady=4)
            entries[col] = ent

        result = {"row": None}

        def save_row():
            row = {col: entries[col].get().strip() for col in columns}
            result["row"] = row
            win.destroy()

        ttk.Button(win, text="Save", command=save_row).grid(row=len(columns), column=0, columnspan=2, pady=8)
        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)
        return result["row"]

    def _add_table_row(self, tab_name, columns, data_tables, trees):
        row = self._edit_row_dialog(f"Add: {tab_name}", columns)
        if row is None:
            return
        data_tables[tab_name].append(row)
        self._refresh_tree(trees[tab_name], columns, data_tables[tab_name])

    def _edit_table_row(self, tab_name, columns, data_tables, trees):
        tree = trees[tab_name]
        sel = tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        row = self._edit_row_dialog(f"Upravit: {tab_name}", columns, data_tables[tab_name][idx])
        if row is None:
            return
        data_tables[tab_name][idx] = row
        self._refresh_tree(tree, columns, data_tables[tab_name])

    def _delete_table_row(self, tab_name, data_tables, trees):
        tree = trees[tab_name]
        sel = tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        data_tables[tab_name].pop(idx)
        columns = tree["columns"]
        self._refresh_tree(tree, columns, data_tables[tab_name])

    @staticmethod
    def _to_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _apply_editor_tables(self, data_tables, special_config=None):
        def parse_int(row, key, default=None):
            value = row.get(key, default)
            if value is None or value == "":
                raise ValueError(key)
            return int(value)

        def parse_float(row, key, default=None):
            value = row.get(key, default)
            if value is None or value == "":
                raise ValueError(key)
            return float(value)

        sections = {
            parse_int(r, "id"): {
                "id": parse_int(r, "id"),
                "E": parse_float(r, "E"),
                "A": parse_float(r, "A"),
                "I": parse_float(r, "I"),
            }
            for r in data_tables["sections"]
        }
        nodes = {
            parse_int(r, "id"): {"id": parse_int(r, "id"), "x": parse_float(r, "x"), "y": parse_float(r, "y")}
            for r in data_tables["nodes"]
        }

        raw_dof_nodes = {}
        for did, dof in self.dof_nodes.items():
            node_id = dof.get("node")
            if node_id not in nodes:
                continue
            try:
                raw_dof_nodes[int(did)] = {
                    "id": int(did),
                    "node": int(node_id),
                    "ux": int(dof["ux"]),
                    "uy": int(dof["uy"]),
                    "rz": int(dof["rz"]),
                }
            except (KeyError, TypeError, ValueError):
                continue

        primary_dofs_by_node = {}
        for did, dof in raw_dof_nodes.items():
            nid = dof["node"]
            primary_dofs_by_node[nid] = min(did, primary_dofs_by_node.get(nid, did))

        elements = {}
        for r in data_tables["elements"]:
            try:
                eid = parse_int(r, "id")
                candidate = {
                    "id": eid,
                    "i": parse_int(r, "i"),
                    "j": parse_int(r, "j"),
                    "section": parse_int(r, "section"),
                }
            except (TypeError, ValueError):
                continue
            if candidate["section"] not in sections:
                continue
            if candidate["i"] not in raw_dof_nodes or candidate["j"] not in raw_dof_nodes:
                continue
            elements[eid] = candidate

        supports = []
        for r in data_tables["supports"]:
            try:
                dof_node_id = parse_int(r, "node")
            except (TypeError, ValueError):
                continue
            if dof_node_id not in raw_dof_nodes:
                continue
            supports.append(
                {
                    "node": dof_node_id,
                    "ux": self._to_bool(r.get("ux", False)),
                    "uy": self._to_bool(r.get("uy", False)),
                    "rz": self._to_bool(r.get("rz", False)),
                }
            )

        nodal_loads = []
        for r in data_tables["nodal loads"]:
            try:
                dof_node_id = parse_int(r, "dof_node")
                Fx = parse_float(r, "Fx", 0.0)
                Fy = parse_float(r, "Fy", 0.0)
                Mz = parse_float(r, "Mz", 0.0)
            except (TypeError, ValueError):
                continue
            if dof_node_id not in raw_dof_nodes:
                continue
            nodal_loads.append({"dof_node": dof_node_id, "Fx": Fx, "Fy": Fy, "Mz": Mz})

        element_loads = []
        for r in data_tables["element load"]:
            try:
                element_id = parse_int(r, "element")
                qx = parse_float(r, "qx", 0.0)
                qz = parse_float(r, "qz", 0.0)
            except (TypeError, ValueError):
                continue
            if element_id not in elements:
                continue
            element_loads.append({"element": element_id, "qx": qx, "qz": qz})

        valid_dof_ids = {
            dof_id
            for dof in raw_dof_nodes.values()
            for dof_id in (dof["ux"], dof["uy"], dof["rz"])
        }

        mass = []
        for r in data_tables.get("mass", []):
            try:
                dof_id = parse_int(r, "dof_id")
                lumped_mass = parse_float(r, "mass")
            except (TypeError, ValueError):
                continue
            if dof_id not in valid_dof_ids:
                continue
            mass.append({"dof_id": dof_id, "mass": lumped_mass})

        used_dofs = set(primary_dofs_by_node.values())
        for element in elements.values():
            used_dofs.add(element["i"])
            used_dofs.add(element["j"])
        for support in supports:
            used_dofs.add(support["node"])
        for load in nodal_loads:
            used_dofs.add(load["dof_node"])
        dof_nodes = {did: dof for did, dof in raw_dof_nodes.items() if did in used_dofs}

        # release rotation tab je odvozený pohled; změny zde se zatím neaplikují zpět
        self.sections = sections
        self.nodes = nodes
        self.dof_nodes = dof_nodes
        self.elements = elements
        self.supports = supports
        self.nodal_loads = nodal_loads
        self.element_loads = element_loads
        self.mass = mass

        if special_config is None:
            special_config = {}

        problem_type = special_config.get("problem_type", self.problem_type)
        if problem_type not in self.PROBLEM_TYPES:
            raise ValueError("Neplatná volba problem_type.")
        self.problem_type = problem_type

        try:
            number_of_eigenvectors = int(special_config.get("number_of_eigenvectors", self.number_of_eigenvectors))
        except (TypeError, ValueError) as exc:
            raise ValueError("number_of_eigenvectors musí být celé číslo.") from exc
        if number_of_eigenvectors < 1:
            raise ValueError("number_of_eigenvectors musí být >= 1.")
        self.number_of_eigenvectors = number_of_eigenvectors

        self.next_section_id = max(self.sections.keys(), default=0) + 1
        self.next_node_id = max(self.nodes.keys(), default=0) + 1
        self.next_element_id = max(self.elements.keys(), default=0) + 1
        self.next_dof_node_id = max(self.dof_nodes.keys(), default=0) + 1
        max_dof = max((max(d["ux"], d["uy"], d["rz"]) for d in self.dof_nodes.values()), default=0)
        self.next_dof_id = max_dof + 1


def main():
    root = tk.Tk()
    app = Frame2DGui(root)
    app.draw_scene()
    root.mainloop()


if __name__ == "__main__":
    main()
